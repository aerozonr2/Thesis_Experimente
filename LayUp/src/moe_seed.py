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

# Required?
def softmax_temperature(x, dim, tau=1.0):
    return torch.softmax(x / tau, dim=dim)

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
        self.taskcla = []
        self.gmms = args.gmms
        self.use_multivariate = args.use_multivariate
        self.selection_method = args.selection_method
        self.empty_expert = {}

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
        self.taskcla.append([])
        choosen_expert_index = t
        if t < self.max_experts:
            print(f"Training expert {t} on task {t}:")
            self.experts_distributions.append([])
            self.train_expert(train_dataset)
        else:
            expert_index = self.choose_expert_to_finetune(t, train_dataset)
            print(f"Finetuning expert {expert_index} on task {t}:")
            self.finetune_expert(t, expert_index, train_dataset) 
            choosen_expert_index = expert_index
        
        #self.create_distributions(train_dataset=train_dataset, exp_index=choosen_expert_index)
                
        '''
        if t >= self.max_experts:
            expert_to_finetune = self.choose_expert_to_finetune(t, train_dataset)
            print(f"Finetuning expert {expert_to_finetune} on task {t}:")
            #self.finetune_expert(t=t, expert_index=expert_to_finetune, train_dataset)
        
        print(f"Creating distributions for task {t}:")
        self.create_distributions(t, train_dataset)
        '''

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
            pbar = tqdm(enumerate(train_loader), desc=f"Epoch: {epoch}", total=num_train_loader) #total=num_train_loader
            # Logger.instance().add_backend(TQDMLogger(pbar)) # Ist is LayUp, weiß nicht ob notwendig
            for batch_id, (inputs, labels) in pbar:
                #print(f'Epoch: {epoch}, Batch: {batch_id + 1}/{num_train_loader}')
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
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
        features = []
        for expert_index in range(len(self.experts)):
            self.switch_to_expert(expert_index)
            out = self.backbone(x)
            features.append(out)
        #return torch.stack(features, dim=1) # Stack the features to get a tensor of shape (?, num_experts, logits_of_classes) as in SEED


        # Stack the features to get a tensor of shape (1, num_experts, logits_of_classes)
        features = torch.stack(features, dim=1)
        
        # Compute the mean along the num_experts dimension (dimension 1)
        average_logits = torch.mean(features, dim=1)
        
        # Squeeze the tensor to remove the singleton dimension
        average_logits = average_logits.squeeze(0)
        
        return average_logits


        # was mache ich mit dem output?
        # Der muss entweder geaveraged werden oder ich mache das mit bayes wie in SEED
        # zweiteres wäre besser. Kann ich einfach die Funktion aus SEED kopieren?
        # Als Prototyp nehme ich erstmal den Average.
    
    @torch.no_grad()
    def choose_expert_to_finetune(self, t, train_dataset):
        if self.selection_method == 'random':
            return self.selection_random()
        elif self.selection_method == 'eucld_dist':
            return self.selection_euclidean_distance(train_dataset)
        elif self.selection_method == 'kl_div':
            return self.selection_kl_divergence(t, train_dataset)
        elif self.selection_method == 'ws_div':
            return self.selection_ws_divergence()
        else:
            raise ValueError('Invalid selection method')
    
    def selection_random(self):
        # Randomly choose an expert to finetune
        return random.randint(0, self.max_experts - 1)
    
    def selection_euclidean_distance(self, train_dataset):
        # Euclidean distance between the current task distribution and the distributions of the experts
        pass
    
    @torch.no_grad()
    def selection_kl_divergence(self, t, train_dataset):
        # KL divergence between the current task distribution and the distributions of the experts
        self.create_distributions(t, train_dataset)
        print("Calculating KL Divergence")

            # Debug prints
        print(f"t: {t}")
        print(f"self.taskcla: {self.taskcla}")
        print(f"len(self.taskcla): {len(self.taskcla)}")
        
        if t >= len(self.taskcla):
            raise IndexError(f"Index t={t} is out of range for self.taskcla with length {len(self.taskcla)}")
        
        expert_overlap = torch.zeros(self.max_experts, device=self.device)
        for expert_index in range(self.max_experts):
            classes_in_t = self.taskcla[t][1]
            new_distributions = self.experts_distributions[expert_index][-classes_in_t:]
            kl_matrix = torch.zeros((len(new_distributions), len(new_distributions)), device=self.device)
            for o, old_gauss_ in enumerate(new_distributions):
                old_gauss = MultivariateNormal(old_gauss_.mu.data[0][0], covariance_matrix=old_gauss_.var.data[0][0])
                for n, new_gauss_ in enumerate(new_distributions):
                    new_gauss = MultivariateNormal(new_gauss_.mu.data[0][0], covariance_matrix=new_gauss_.var.data[0][0])
                    kl_matrix[n, o] = torch.distributions.kl_divergence(new_gauss, old_gauss)
            expert_overlap[expert_index] = torch.mean(kl_matrix)
            self.experts_distributions[expert_index] = self.experts_distributions[expert_index][:-classes_in_t]
        print(f"Expert overlap: {expert_overlap}")
        expert_to_finetune = torch.argmax(expert_overlap)
        self.taskcla = self.taskcla[:-1]
        return int(expert_to_finetune)
        """
        expert_overlap = torch.zeros(self.max_experts, device=self.device)
        for expert_index in range(self.max_experts):
            self.switch_to_expert(expert_index)
            #expert_overlap[expert_index] = self.calculate_kl_divergences(t, expert_index)
            new_distributions = self.experts_distributions[expert_index]
            kl_matrix = torch.zeros((len(new_distributions), len(new_distributions)), device=self.device)
            
            for o, old_gauss_ in enumerate(new_distributions):
                old_gauss = MultivariateNormal(old_gauss_.mu.data[0][0], covariance_matrix=old_gauss_.var.data[0][0])
                for n, new_gauss_ in enumerate(new_distributions):
                    new_gauss = MultivariateNormal(new_gauss_.mu.data[0][0], covariance_matrix=new_gauss_.var.data[0][0])
                    kl_matrix[n, o] = torch.distributions.kl_divergence(new_gauss, old_gauss)
            
        
            expert_overlap[expert_index] = torch.mean(kl_matrix)

        expert_to_finetune = torch.argmax(expert_overlap)
        return int(expert_to_finetune)
        """

    def selection_ws_divergence(self, t):
        # Wasserstein divergence between the current task distribution and the distributions of the experts    
        pass

    def finetune_expert(self, t, expert_index, train_dataset):
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
            for batch_id, (inputs, labels) in pbar:
                bsz = inputs.shape[0]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                with torch.no_grad():
                        old_features = old_model(inputs) # Muss Identity head haben!
                outputs = model(inputs)
                features = model.forward_features(inputs)

                loss = self.criterion(outputs, labels)# , features, old_features) => Error
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
                    print(f"WARNING: Covariance matrix is singular. Increasing eps to: {eps:.7f} but this may hurt results")
                else:
                    is_ok = True

            if len(gmm.mu.data.shape) == 2:
                gmm.mu.data = gmm.mu.data.unsqueeze(1)
            self.experts_distributions[exp_index].append(gmm)
            print("gmm added")

    @torch.no_grad()
    def eval(self):
        self.backbone.eval()
    
    @torch.no_grad()
    def calculate_metrics(self, features, targets, t):
        pass
    
    @torch.no_grad() 
    def predict_class_bayes(self, t, features): # Etwas angefangen
        log_probs = torch.full((features.shape[0], len(self.experts_distributions), len(self.experts_distributions[0])), fill_value=-1e8, device=features.device)
        assert len(self.experts_distributions[0]) == self.num_classes
        assert len(self.experts_distributions) == self.max_experts
        mask = torch.full_like(log_probs, fill_value=False, dtype=torch.bool)

        #?
        for expert_index, _ in enumerate(self.experts_distributions):
            for c, class_gmm in enumerate(self.experts_distributions[expert_index]):
                c += self.model.task_offset[expert_index]
                log_probs[:, expert_index, c] = class_gmm.score_samples(features[:, expert_index])
                mask[:, expert_index, c] = True # "diese Klasse wurde von diesem Experten gelernt", glaube ich


        # Von Copilot
        for expert_index in range(len(self.experts_distributions)):
            for class_index in range(len(self.experts_distributions[expert_index])):
                gmm = self.experts_distributions[expert_index][class_index]
                if self.use_multivariate:
                    log_probs[:, expert_index, class_index] = gmm.log_prob(features)
                else:
                    log_probs[:, expert_index, class_index] = gmm.log_prob(features)   
    
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