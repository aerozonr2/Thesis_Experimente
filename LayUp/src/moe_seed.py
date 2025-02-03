import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader
import timm
from .backbone.adapter import add_adapters
from .backbone.ssf import add_ssf
from .backbone.vpt import add_vpt
from .backbone.util import call_in_all_submodules
from .approach.mvgb import ClassMemoryDataset, ClassDirectoryDataset
from .approach.gmm import GaussianMixture
from scipy.stats import wasserstein_distance


from .support_functions import check_gpu_memory

import copy
import random
import torch
from tqdm import tqdm

import numpy as np

from argparse import ArgumentParser
from itertools import compress
from torch import nn
from torch.utils.data import Dataset
from torch.distributions import MultivariateNormal

#from .mvgb import ClassMemoryDataset, ClassDirectoryDataset
#from .gmm import GaussianMixture
#from .incremental_learning import Inc_Learning_Appr

#torch.backends.cuda.matmul.allow_tf32 = False


class MoE_SEED(nn.Module):
    def __init__(self, args):
        super(MoE_SEED, self).__init__()
        self.backbone = timm.create_model(args.backbone, pretrained=True)
        self.finetune_method = args.finetune_method
        self.max_experts = args.moe_max_experts
        self.experts = []
        self.expert_heads = []
        self.experts_distributions = []
        self.finetune_epochs = args.finetune_epochs
        self.device = args.device
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.args = args
        self.backbone_param_names = []
        self.num_classes = None
        self.alpha = 0.99 # Knowledge distillation parameter for the loss function. 1.0 means no knowledge distillation. 0.99 is from SEED.
        self.gmms = args.gmms
        self.use_multivariate = args.use_multivariate
        self.selection_method = args.selection_method
        self.empty_expert = {}
        self.tau = 1.0 # Muss in args rein
        self.classification = args.classification # average/bayes
        self.kd = args.kd # muss am ende Weg?

    @torch.no_grad()
    def save_backbone_param_names(self):
        if len(self.backbone_param_names) == 0:
            for name, _ in self.backbone.named_parameters():
                self.backbone_param_names.append(name)
        else:
            print("Backbone parameter names already saved.")

    def add_expert(self):
        # Add Expert
        if self.finetune_method == 'ssf':
            self.backbone = add_ssf(self.backbone)
        elif self.finetune_method == 'vpt': 
            self.backbone = add_vpt(self.backbone, vpt_type="shallow")
        elif self.finetune_method == 'adapter':
            self.backbone = add_adapters(self.backbone)
        else: 
            raise ValueError('Invalid finetune method')

    def train_loop(self, t, train_dataset):
        choosen_expert_index = t
        if t < self.max_experts:
            print(f"Training expert {t} on task {t}:")
            self.experts_distributions.append([])
            self.train_expert(train_dataset)
        else:
            expert_index = self.choose_expert_to_finetune(train_dataset)
            print(f"Finetuning expert {expert_index} on task {t}:")
            self.finetune_expert(expert_index, train_dataset) 
            choosen_expert_index = expert_index
        
        print(f"Creating distributions for task {t}:")
        gmms = self.create_distributions(train_dataset=train_dataset, exp_index=choosen_expert_index)
        for gmm in gmms:
            self.experts_distributions[choosen_expert_index].append(gmm)

    def freeze_ViT_unfreeze_PEFT(self, model=None):
        # Freeze the backbone parameters based on names except for the head
        # Model as parameter
        if model is not None:
            for name, param in model.named_parameters():
                if name in self.backbone_param_names:
                    if name == 'head.weight' or name == 'head.bias':
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    param.requires_grad = True
            return model
        
        # Backbone
        for name, param in self.backbone.named_parameters():
            if name in self.backbone_param_names:
                if name == 'head.weight' or name == 'head.bias':
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True

    def train_expert(self, train_dataset=None):
        # initialise expert from empty_expert
        for name, param in self.backbone.named_parameters():
            if name in self.empty_expert:
                param.data.copy_(self.empty_expert[name])


        # Add a linear head at the end of the network
        num_features = self.backbone.num_features
        num_classes = self.num_classes
        self.backbone.head = nn.Linear(num_features, num_classes)

        # Freeze the backbone parameters based on names except for the head
        self.freeze_ViT_unfreeze_PEFT()

        model = self.backbone
        model.train()

        # GPU/CPU
        model.to(self.device)
        # Train model on task:
        optimizer, sceduler = self.get_optimizer(num_param=model.parameters())

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        for epoch in range(self.finetune_epochs):
            running_loss = 0.0
            num_train_loader = len(train_loader)
            pbar = tqdm(enumerate(train_loader), desc=f"Epoch: {epoch}", total=num_train_loader)
            # Logger.instance().add_backend(TQDMLogger(pbar)) # Ist is LayUp, weiß nicht ob notwendig
            for _, (inputs, labels) in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()



            print(f"Epoch [{epoch + 1}/{self.finetune_epochs}], Loss: {running_loss / len(train_loader)}")

            #
            print("#"*50)
            print(len(train_loader))
            print(f"input shape: {inputs.shape}")
            print(f"output shape: {outputs.shape}")
            print(outputs[0])
            print(f"target shape: {labels.shape}")
            print(labels[0])
            print("#"*10)
            for inputs, labels in train_loader:
                print(f"input shape: {inputs.shape}")
                print(f"target shape: {labels.shape}")
                print(labels)
                break
            print("#"*50)
            #

        # Save Expert parameters
        expert_parameters = {}
        for name, param in self.backbone.named_parameters():
            if name not in self.backbone_param_names:
                expert_parameters[name] = copy.deepcopy(param) 
            else:
                pass # Not saving backbone parameters
        self.experts.append(copy.deepcopy(expert_parameters))
        self.expert_heads.append(copy.deepcopy(self.backbone.head.state_dict()))

    def switch_to_expert(self, expert_index):
        # das mit dem head und identity muss ich nochmal testen und gucken obder head name wirklich gelöscht wird, oder ob ic das als eigenen parameter selbst implementieren muss.
        for name, param in self.experts[expert_index].items():
                self.backbone.state_dict()[name].copy_(param)
        if isinstance(self.backbone.head, nn.Identity):
            self.backbone.head = nn.Linear(self.backbone.num_features, self.num_classes)
        self.backbone.head.load_state_dict(self.expert_heads[expert_index])
        self.backbone.head.to(self.device)

    @torch.no_grad()
    def forward(self, x):
        if self.classification == 'average':
            return self.forward_average(x)
        elif self.classification == 'bayesian':
            return self.forward_bayes(x)
        else:
            raise ValueError('Invalid classification method')

    @torch.no_grad()
    def forward_average(self, x):
        features = []
        for expert_index in range(len(self.experts)):
            self.switch_to_expert(expert_index)
            out = self.backbone(x)
            features.append(out)

        # Stack the features to get a tensor of shape (1, num_experts, logits_of_classes)
        features = torch.stack(features, dim=1)        
        # Compute the mean along the num_experts dimension (dimension 1)
        average_logits = torch.mean(features, dim=1)
        # Squeeze the tensor to remove the singleton dimension
        average_logits = average_logits.squeeze(0)
        
        return average_logits

    @torch.no_grad()
    def forward_bayes(self, x):
        features = []
        for expert_index in range(len(self.experts)):
            self.switch_to_expert(expert_index)
            out = self.backbone(x)
            features.append(out)
        id = self.predict_class_bayes(features=torch.stack(features, dim=1))
        print(id)
        return id

    @torch.no_grad()
    def choose_expert_to_finetune(self, train_dataset):
        if self.selection_method == 'random':
            return self.selection_random()
        elif self.selection_method == 'eucld_dist':
            return self.selection_euclidean_distance(train_dataset)
        elif self.selection_method == 'kl_div':
            return self.selection_kl_divergence(train_dataset)
        elif self.selection_method == 'ws_div':
            return self.selection_ws_divergence(train_dataset)
        else:
            raise ValueError('Invalid selection method')
        # muss man bei beiden Divergenzen .argmin() oder .argmax() am Ende nehmen? In SEED wird bei KL Divergenz .argmax() genommen, aber was ist mit Wasserstein?
        # geiches gilt für Euclidean Distance. Copilot nimmt bei WS und Eucl .argmin(). Ich wegen SEED Logik erstmal .argmax().

    def selection_random(self):
        # Randomly choose an expert to finetune
        return random.randint(0, self.max_experts - 1)
    
    @torch.no_grad()
    def selection_euclidean_distance(self, train_dataset):
        # Euclidean distance between the current task distribution and the distributions of the experts
        experts_mean_euclidean_dist = torch.zeros(self.max_experts, device=self.device)

        for expert_index in range(self.max_experts):
            new_distributions = self.create_distributions(train_dataset, exp_index=expert_index)
            
            euclidean_matrix = torch.zeros((len(new_distributions), len(self.experts_distributions[expert_index])), device=self.device)
            for n, new_gmm in enumerate(new_distributions):
                
                new_gauss = new_gmm.mu.data[0][0]
                for o, old_gmm in enumerate(self.experts_distributions[expert_index]):
                    old_gauss = old_gmm.mu.data[0][0]
                    
                    euclidean_dist = torch.cdist(new_gauss.unsqueeze(0), old_gauss.unsqueeze(0), p=2).item()
                    euclidean_matrix[n, o] = euclidean_dist

            experts_mean_euclidean_dist[expert_index] = torch.mean(euclidean_matrix)
        print(experts_mean_euclidean_dist)
        exp_to_finetune = torch.argmax(experts_mean_euclidean_dist)  # Choose expert with the highest Euclidean distance
        return exp_to_finetune

    @torch.no_grad()
    def selection_kl_divergence(self, train_dataset):
        # KL divergence between the current task distribution and the distributions of the experts
        experts_mean_kl_div = torch.zeros(self.max_experts, device=self.device)

        for expert_index in range(self.max_experts): # min kann später weg
            new_distributions = self.create_distributions(train_dataset, expert_index)

            kl_matrix = torch.zeros((len(new_distributions), len(self.experts_distributions[expert_index])), device=self.device)
            for n, new_gmm in enumerate(new_distributions):

                new_gauss = MultivariateNormal(new_gmm.mu.data[0][0], covariance_matrix=new_gmm.var.data[0][0])
                for o, old_gmm in enumerate(self.experts_distributions[expert_index]):
                    old_gauss = MultivariateNormal(old_gmm.mu.data[0][0], covariance_matrix=old_gmm.var.data[0][0])

                    #
                    '''
                    new_gmm = torch.full((1, 1, 10), 10.)
                    new_cov = torch.eye(10) * 0.1
                    new_gauss = MultivariateNormal(new_gmm[0][0], covariance_matrix=new_cov)


                    old_gmm1 = torch.full((1, 1, 10), 5.)
                    old_cov1 = torch.eye(10) * 0.1
                    old_gauss1 = MultivariateNormal(old_gmm1[0][0], covariance_matrix=old_cov1)

                    old_gmm2 = torch.full((1, 1, 10), 10.)
                    old_cov2 = torch.eye(10) * 0.1
                    old_gauss2 = MultivariateNormal(old_gmm2[0][0], covariance_matrix=old_cov2)

                    old_gmm3 = torch.full((1, 1, 10), 15.)
                    old_cov3 = torch.eye(10) * 0.1
                    old_gauss3 = MultivariateNormal(old_gmm3[0][0], covariance_matrix=old_cov3)

                    old_gmm4 = torch.full((1, 1, 10), 20.)
                    old_cov4 = torch.eye(10) * 0.1
                    old_gauss4 = MultivariateNormal(old_gmm4[0][0], covariance_matrix=old_cov4)

                    kl_div1 = torch.distributions.kl_divergence(new_gauss, old_gauss1)
                    kl_div2 = torch.distributions.kl_divergence(new_gauss, old_gauss2)
                    kl_div3 = torch.distributions.kl_divergence(new_gauss, old_gauss3)
                    kl_div4 = torch.distributions.kl_divergence(new_gauss, old_gauss4)

                    kl_div5 = torch.distributions.kl_divergence(old_gauss1, new_gauss)
                    kl_div6 = torch.distributions.kl_divergence(old_gauss2, new_gauss)
                    kl_div7 = torch.distributions.kl_divergence(old_gauss3, new_gauss)
                    kl_div8 = torch.distributions.kl_divergence(old_gauss4, new_gauss)

                    print("KL Divergence(new, old)")
                    print(f"KL Divergence 1: {kl_div1}")
                    print(f"KL Divergence 2: {kl_div2}")
                    print(f"KL Divergence 3: {kl_div3}")
                    print(f"KL Divergence 4: {kl_div4}")

                    print("KL Divergence(old, new)")
                    print(f"KL Divergence 1: {kl_div5}")
                    print(f"KL Divergence 2: {kl_div6}")
                    print(f"KL Divergence 3: {kl_div7}")
                    print(f"KL Divergence 4: {kl_div8}")


                    assert False
                    '''
                    #

                    kl_div = torch.distributions.kl_divergence(new_gauss, old_gauss) # KL divergence between the current task distribution and the distributions of the experts (of each class)
                    kl_matrix[n, o] = kl_div

            experts_mean_kl_div[expert_index] = torch.mean(kl_matrix)
        exp_to_finetune = torch.argmax(experts_mean_kl_div) # Choose expert with the highest KL divergence
        return exp_to_finetune        

    @torch.no_grad()
    def selection_ws_divergence(self, train_dataset):
        # Wasserstein divergence between the current task distribution and the distributions of the experts    
        experts_mean_ws_div = torch.zeros(self.max_experts, device=self.device)

        for expert_index in range(self.max_experts):
            new_distributions = self.create_distributions(train_dataset, expert_index) # Create distributions for the current task

            ws_matrix = torch.zeros((len(new_distributions), len(self.experts_distributions[expert_index])), device=self.device)
            for n, new_gmm in enumerate(new_distributions):

                new_gauss = new_gmm.mu.data[0][0].cpu().numpy()
                for o, old_gmm in enumerate(self.experts_distributions[expert_index]):
                    old_gauss = old_gmm.mu.data[0][0].cpu().numpy()

                    ws_div = wasserstein_distance(new_gauss, old_gauss)
                    ws_matrix[n, o] = ws_div

            experts_mean_ws_div[expert_index] = torch.mean(ws_matrix)
        exp_to_finetune = torch.argmax(experts_mean_ws_div)  # Choose expert with the lowest Wasserstein divergence
        return exp_to_finetune

    def finetune_expert(self, expert_index, train_dataset):
        self.switch_to_expert(expert_index)
        old_model = copy.deepcopy(self.backbone)
        old_model.eval()

        # Add a linear head at the end of the network
        num_features = self.backbone.num_features
        num_classes = self.num_classes
        model = self.backbone
        model.head = nn.Linear(num_features, num_classes)
        
        self.freeze_ViT_unfreeze_PEFT()
        model.to(self.device)

        # Finetune model on task:
        optimizer, sceduler = self.get_optimizer(num_param=model.parameters())
# GPT für .item() runtime Verbesserung. Dadurch werden die batches küntstlich größer. und ich muss das auch in Train_expert machen. und switch expert!?!
# als näyhstes nochmal laufen lassen und cprofile vergleichen: --T 10!
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        for epoch in range(self.finetune_epochs):
            model.train()
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            
            running_loss = 0.0
            num_train_loader = len(train_loader)
            pbar = tqdm(enumerate(train_loader), desc=f"Epoch: {epoch}", total=num_train_loader)
            # Logger.instance().add_backend(TQDMLogger(pbar)) # Ist is LayUp, weiß nicht ob notwendig
            for _, (inputs, labels) in pbar:
                #bsz = inputs.shape[0]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                with torch.no_grad():
                        old_features = old_model.forward_features(inputs)
                outputs = model(inputs)
                features = model.forward_features(inputs)

                if self.kd:
                    loss = self.criterion(outputs, labels, features, old_features)
                else:
                    loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{self.finetune_epochs}], Loss: {running_loss / len(train_loader)}")

        # Save Expert parameters
        expert_parameters = {}
        for name, param in self.backbone.named_parameters():
            if name not in self.backbone_param_names:
                expert_parameters[name] = copy.deepcopy(param) 
            else:
                pass # Not saving backbone parameters
        self.experts[expert_index] = copy.deepcopy(expert_parameters)
        self.expert_heads[expert_index] = copy.deepcopy(self.backbone.head.state_dict())

    @torch.no_grad()
    def create_distributions(self, train_dataset, exp_index):
        """ 
        Create distributions for task t.
        One distribution for each class.
        The distributions are stored in self.experts_distributions[expert_index],
        but only for the expert who leared the Task
        """
        # Collect all images and labels needs much time
        all_images = []
        all_labels = []
        # Iterate over the DataLoader to collect images and labels
        for image, label in train_dataset:
            all_images.append(image)
            all_labels.append(label)

        # Convert into tensors
        all_images = torch.stack(all_images)
        all_labels = torch.tensor(all_labels)

        # Get expert
        self.switch_to_expert(exp_index)
        model = self.backbone
        model.head = nn.Identity()
        model.eval()
        model.to(self.device)

        gmms = []

        # Iterate over each class
        unique_labels = all_labels.unique()
        for class_label in unique_labels:
            # Get all images of the class
            class_indices = (all_labels == class_label).nonzero(as_tuple=True)[0]
            class_images = all_images[class_indices]
            class_loader = DataLoader(
                class_images,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )

            # Get expert features
            from_ = 0
            class_features = torch.full((2 * len(class_images), model.num_features), fill_value=-999999999.0, device=self.device)  
            # Process each image in the class and its flipped version
            for images in class_loader:
                bsz = images.shape[0]
                images = images.to(self.device)
                features = model(images)
                class_features[from_: from_+bsz, :] = features
                features = model(torch.flip(images, dims=(3,)))
                class_features[from_+bsz: from_+2*bsz, :] = features
                from_ += 2*bsz
            # Calculate distributions
            cov_type = "full" if self.use_multivariate else "diag"
            is_ok = False
            eps = 1e-8
            while not is_ok:
                try:
                    gmm = GaussianMixture(self.gmms, class_features.shape[1], covariance_type=cov_type, eps=eps).to(self.device)
                    gmm.fit(class_features, delta=1e-3, n_iter=100)
                except RuntimeError:
                    eps = 10 * eps
                else:
                    #print(f"WARNING: Covariance matrix is singular. Increasing eps from: {1e-8:.7f} to: {eps:.7f} but this may hurt results")   ###Muss später wieder rein
                    is_ok = True

            if len(gmm.mu.data.shape) == 2:
                gmm.mu.data = gmm.mu.data.unsqueeze(1)

            gmm_shape = gmm.mu.data.shape
            assert len(gmm_shape) == 3
            assert gmm_shape[0] == 1
            assert gmm_shape[1] == 1
            assert gmm_shape[2] == model.num_features

            gmms.append(gmm)
        return gmms

    @torch.no_grad()
    def eval(self):
        self.backbone.eval()
    
    @torch.no_grad()
    def calculate_metrics(self, features, targets, t):
        pass
    
    @torch.no_grad() 
    def predict_class_bayes(self, features): # Taskoffset und übersezung fehlen noch
        log_probs = torch.full((features.shape[0], len(self.experts_distributions), len(self.experts_distributions[0])), fill_value=-1e8, device=features.device)
        # shape: (batch_size, num_experts, num_classes/num_distr. learned by expert one) 
        mask = torch.full_like(log_probs, fill_value=False, dtype=torch.bool)

        ## ich muss in den Exp_distr einbauen ob ein Task nicht gelernt wurde.
        ## das wird in Seed über task_offset gemacht.
        ## ich mache das wahrscheinlich durch das hinzufügen einer leeren liste, gibt dann aber evtl. ein Problem mit selection_method, weil ich dann manchmal ein leeres array habe.
        for expert_index in range(len(self.experts_distributions)):
            for c, class_gmm in enumerate(self.experts_distributions[expert_index]):
                c += self.task_offset[expert_index] # welcher Task/Klasse? wurde von welchem Experten gelernt
                log_probs[:, expert_index, c] = class_gmm.score_samples(features[:, expert_index])
                mask[:, expert_index, c] = True # "diese Klasse wurde von diesem Experten gelernt"
        
        log_probs = torch.softmax(log_probs/self.tau, dim=2)
        confidences = torch.sum(log_probs, dim=1) / torch.sum(mask, dim=1)
        tag_class_id = torch.argmax(confidences, dim=1)
        # muss für evaluation was anderes returnen(logits).
        # Er kommt noch durcheinander weil nicht jeder Experte die gleiche (Anzahl) an Klassen gelernt hat.
        return tag_class_id

    def criterion(self, outputs, targets, features=None, old_features=None):
        """Returns the loss value"""
        ce_loss = nn.functional.cross_entropy(outputs, targets, label_smoothing=0.0)
        if old_features is not None:  # Knowledge distillation 
            kd_loss = nn.functional.mse_loss(features, old_features)
            total_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            return total_loss
        return ce_loss

    def freeze(self, fully=False):
        # Requires_grad = False for all PEFT parameters
        call_in_all_submodules(self.backbone, "freeze", fully=fully)

    def get_optimizer(self, num_param, milestones=[60, 120, 160]):
            """Returns the optimizer"""
            optimizer = torch.optim.SGD(num_param, lr=self.lr, momentum=0.9) # weight_decay=wd?
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
            return optimizer, scheduler