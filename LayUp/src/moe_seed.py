import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader
import timm
from .backbone.adapter import add_adapters
from .backbone.ssf import add_ssf
from .backbone.vpt import add_vpt
from .backbone.util import call_in_all_submodules
#from .approach.mvgb import ClassMemoryDataset, ClassDirectoryDataset
from .approach.gmm import GaussianMixture
from scipy.stats import wasserstein_distance
from scipy.special import kl_div as kl_divergence

from .modules import CosineLinear
from .support_functions import check_gpu_memory

import copy
import random
import torch
from tqdm import tqdm
import time


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

def check_backbone_frozen(model: torch.nn.Module, i=None) -> None:
    """
    Checks if the backbone parameters of a timm ViT model are frozen.

    Args:
        model (torch.nn.Module): The timm ViT model.
    """

    backbone_modules = [model.patch_embed, model.blocks]  # Common backbone modules in ViT

    for name, param in model.named_parameters():
        is_backbone_param = False
        for backbone_module in backbone_modules:
            if any(module_name in name for module_name in [n for n, _ in backbone_module.named_modules()]):
                is_backbone_param = True
                break  # No need to check other backbone modules

        if is_backbone_param:
            if param.requires_grad:
                print(f"{i} ERROR: Backbone parameter {name} is trainable!")
            else:
                pass
        else:
            if param.requires_grad:
                print(f"{i} TRAINABLE: Non-backbone parameter {name} is trainable.")
            else:
                print(f"{i} FROZEN: Non-backbone parameter {name} is frozen (unexpected?).")

def check_expert_params(model: torch.nn.Module, i=None) -> None:
    """
    Prints first and last 10 parameters of the VPT tokens in a timm ViT model.
    Prints first and last 10 parameters of the head in a timm ViT model.

    Args:
        model (torch.nn.Module): The timm ViT model.
    """
    for name, param in model.named_parameters():
        if "vpt_prompt_tokens" in name:
            print(f"********** VPT tokens {name}: **********")
            print(f"First 10 parameters: {param.data[:10]}")
            print(f"Last 10 parameters: {param.data[-10:]}")
            print(f"Shape: {param.shape}")
            print(f"############# {i} #############")
        elif "head" in name:
            print(f"********** Head {name}: **********")
            print(f"First 10 parameters: {param.data[:10]}")
            print(f"Last 10 parameters: {param.data[-10:]}")
            print(f"Shape: {param.shape}")
            print(f"############# {i} #############")


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
        self.alpha = args.kd_alpha # Knowledge distillation parameter for the loss function. 1.0 means no knowledge distillation. 0.99 is from SEED.
        self.gmms = args.gmms
        self.use_multivariate = args.use_multivariate
        self.selection_method = args.selection_method
        self.empty_expert = {}
        self.tau = args.tau # Muss in args rein
        self.kd = args.kd # muss am ende Weg?
        self.logger = None
        self.selection_criterion = args.selection_criterion
        self.task_winning_expert = []
        self.use_adamw_and_cosinealing = args.use_adamw_and_cosinealing
        self.flipped_fetures = args.add_flipped_features
        self.accumulation_steps = args.accumulation_steps
        self.bottleneck_dim = args.bottleneck_dim
        self.cl_settig = None

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
            self.backbone = add_adapters(self.backbone, adapter_bottleneck=self.bottleneck_dim)
        else: 
            raise ValueError('Invalid finetune method')

    def train_loop(self, t, train_dataset):
        choosen_expert_index = t
        if t < self.max_experts:
            print(f"Training expert {t} on task {t}:")
            total_classes_learned_by_all_exp = 0
            if len(self.experts_distributions) != 0:
                total_classes_learned_by_all_exp = max(len(exp_distributions) for exp_distributions in self.experts_distributions)
            self.experts_distributions.append([[] for _ in range(total_classes_learned_by_all_exp)])

            self.train_expert(train_dataset)
        else:
            expert_index = self.choose_expert_to_finetune(train_dataset, t)
            #expert_index = 1 # Test
            #print("Experte index: expert_index = t mod self.max_experts")
            #expert_index = t % self.max_experts
            #print(f"Finetune Index is set to {expert_index}")
            print(f"Finetuning expert {expert_index} on task {t}:")
            self.finetune_expert(expert_index, train_dataset) 
            choosen_expert_index = expert_index

        self.logger.log({f"Expert {choosen_expert_index} learned task": t})

        print(f"Creating distributions for task {t}")
        gmms = self.create_distributions(train_dataset=train_dataset, exp_index=choosen_expert_index)
        for gmm in gmms:
            for expert_index in range(len(self.experts_distributions)):
                if expert_index == choosen_expert_index:
                    self.experts_distributions[choosen_expert_index].append(gmm)
                else:
                    self.experts_distributions[expert_index].append([])
        
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
        model.to(self.device, non_blocking=True)
        # Train model on task:
        optimizer, sceduler = self.get_optimizer(num_param=model.parameters())
        optimizer.zero_grad()


        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        for epoch in range(self.finetune_epochs):
            running_loss = 0.0
            #num_train_loader = len(train_loader)
            #pbar = tqdm(enumerate(train_loader), desc=f"Epoch: {epoch}", total=num_train_loader)


            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()


                if self.accumulation_steps > 1:
                    if (i + 1) % self.accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()

                running_loss += loss.item()

            sceduler.step()

            print(f"Epoch [{epoch + 1}/{self.finetune_epochs}], Loss: {running_loss / len(train_loader)}")
        
        del train_loader

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
        self.backbone.head.to(self.device, non_blocking=True)

    @torch.no_grad()
    def forward(self, x):
        return self.forward_bayes(x)

    @torch.no_grad()
    def forward_bayes(self, x):
        features = []
        for expert_index in range(len(self.experts)):
            self.switch_to_expert(expert_index)
            self.backbone.head = nn.Identity()
            self.backbone.to(self.device, non_blocking=True)            
            out = self.backbone(x)
            features.append(out)

        stacked_features = torch.stack(features, dim=1)
        #print(f"Stacked features shape: {stacked_features.shape}")
        if self.cl_settig == "CIL":
            synthetic_softmaxed_logits = self.predict_class_bayes_cil(features=stacked_features)
        elif self.cl_settig == "DIL":
            synthetic_softmaxed_logits = self.predict_class_bayes_dil(features=stacked_features)
        else:
            raise ValueError("Invalid CL setting. Must be either 'CIL' or 'DIL'.")
        return synthetic_softmaxed_logits

    def choose_expert_to_finetune(self, train_dataset, t):
        if self.selection_method == 'random':
            return self.selection_random()
        elif self.selection_method == 'around':
            return t % self.max_experts
        elif self.selection_method == 'eucld_dist':
            return self.selection_euclidean_distance(train_dataset)
        elif self.selection_method == 'inv_eucld_dist':
            return self.selection_euclidean_distance(train_dataset, inverted=True)
        elif self.selection_method == 'kl_div':
            return self.selection_kl_divergence(train_dataset)
        elif self.selection_method == 'inv_kl_div':
            return self.selection_kl_divergence(train_dataset, inverted=True)
        elif self.selection_method == 'ws_div':
            return self.selection_ws_divergence(train_dataset)
        elif self.selection_method == 'inv_ws_div':
            return self.selection_ws_divergence(train_dataset, inverted=True)
        elif self.selection_method == 'first':
            return 0
        else:
            raise ValueError('Invalid selection method')
        # muss man bei beiden Divergenzen .argmin() oder .argmax() am Ende nehmen? In SEED wird bei KL Divergenz .argmax() genommen, aber was ist mit Wasserstein?
        # geiches gilt für Euclidean Distance. Copilot nimmt bei WS und Eucl .argmin(). Ich wegen SEED Logik erstmal .argmax().

    def calc_selection_criterion(self, t):
        test = self.selection_criterion
        if test == 0: # normal
            return torch.mean(t)
        elif test == 1:
            return torch.min(t)
        elif test == 2:
            return torch.max(t)
        else:
            raise ValueError('Invalid selection criterium')

    def selection_random(self):
        # Randomly choose an expert to finetune
        return random.randint(0, self.max_experts - 1)
    
    def selection_euclidean_distance(self, train_dataset, inverted=False):
        # Euclidean distance between the current task distribution and the distributions of the experts
        experts_mean_euclidean_dist = torch.zeros(self.max_experts, device=self.device)

        for expert_index in range(self.max_experts):
            new_distributions = self.create_distributions(train_dataset, exp_index=expert_index)
            learned_distributions = [lst for lst in self.experts_distributions[expert_index] if lst] # Filter out empty lists
            
            euclidean_matrix = torch.zeros((len(new_distributions), len(learned_distributions)), device=self.device)
            for n, new_gmm in enumerate(new_distributions):
                
                new_gauss = new_gmm.mu.data[0][0]
                for o, old_gmm in enumerate(learned_distributions):
                    old_gauss = old_gmm.mu.data[0][0]
                    
                    euclidean_dist = torch.cdist(new_gauss.unsqueeze(0), old_gauss.unsqueeze(0), p=2).item()
                    euclidean_matrix[n, o] = euclidean_dist

            experts_mean_euclidean_dist[expert_index] = self.calc_selection_criterion(euclidean_matrix)
        if inverted:
            exp_to_finetune = torch.argmin(experts_mean_euclidean_dist)
        else:
            exp_to_finetune = torch.argmax(experts_mean_euclidean_dist)  # Choose expert with the highest Euclidean distance
        return exp_to_finetune

    def selection_kl_divergence_old(self, train_dataset, inverted=False):
        # KL divergence between the current task distribution and the distributions of the experts
        print("########### Kl_div ########")

        experts_mean_kl_div = torch.zeros(self.max_experts, device=self.device)

        for expert_index in range(self.max_experts):
            new_distributions = self.create_distributions(train_dataset, expert_index)
            learned_distributions = [lst for lst in self.experts_distributions[expert_index] if lst] # Filter out empty lists

            kl_matrix = torch.zeros((len(new_distributions), len(learned_distributions)), device=self.device)
            for n, new_gmm in enumerate(new_distributions):

                new_gauss = MultivariateNormal(new_gmm.mu.data[0][0], covariance_matrix=new_gmm.var.data[0][0])
                for o, old_gmm in enumerate(learned_distributions):
                    old_gauss = MultivariateNormal(old_gmm.mu.data[0][0], covariance_matrix=old_gmm.var.data[0][0])
                    kl_div = torch.distributions.kl_divergence(new_gauss, old_gauss) # KL divergence between the current task distribution and the distributions of the experts (of each class)
                    
                    kl_matrix[n, o] = kl_div
                    
            experts_mean_kl_div[expert_index] = self.calc_selection_criterion(kl_matrix)            
        
        if inverted:
            exp_to_finetune = torch.argmin(experts_mean_kl_div)
        else:
            exp_to_finetune = torch.argmax(experts_mean_kl_div) # Choose expert with the highest KL divergence
        return exp_to_finetune        

    def selection_kl_divergence(self, train_dataset, inverted=False):
        # KL divergence between the current task distribution and the distributions of the experts
        print("########### Kl_div ########")

        experts_mean_kl_div = torch.zeros(self.max_experts, device=self.device)

        for expert_index in range(self.max_experts):
            new_distributions = self.create_distributions(train_dataset, expert_index)
            learned_distributions = [lst for lst in self.experts_distributions[expert_index] if lst] # Filter out empty lists

            kl_matrix = torch.zeros((len(new_distributions), len(learned_distributions)), device=self.device)
            for n, new_gmm in enumerate(new_distributions):
                new_mean = new_gmm.mu.data[0][0]
                new_variance = new_gmm.var.data[0][0]
                new_stddev = torch.sqrt(new_variance)
                new_dist = torch.distributions.Independent(torch.distributions.Normal(new_mean, new_stddev), reinterpreted_batch_ndims=1)

                for o, old_gmm in enumerate(learned_distributions):
                    old_mean = old_gmm.mu.data[0][0]
                    old_variance = old_gmm.var.data[0][0]
                    old_stddev = torch.sqrt(old_variance)
                    old_dist = torch.distributions.Independent(torch.distributions.Normal(old_mean, old_stddev), reinterpreted_batch_ndims=1)

                    kl_div = torch.distributions.kl_divergence(new_dist, old_dist)
                    kl_matrix[n, o] = kl_div

                    
            experts_mean_kl_div[expert_index] = self.calc_selection_criterion(kl_matrix)            
        
        if inverted:
            exp_to_finetune = torch.argmin(experts_mean_kl_div)
        else:
            exp_to_finetune = torch.argmax(experts_mean_kl_div) # Choose expert with the highest KL divergence
        return exp_to_finetune  

    def selection_kl_divergence_full(self, train_dataset, inverted=False):
        # KL divergence between the current task distribution and the distributions of the experts
        print("########### Kl_div ########")

        experts_mean_kl_div = torch.zeros(self.max_experts, device=self.device)

        for expert_index in range(self.max_experts):
            new_distributions = self.create_distributions(train_dataset, expert_index)
            learned_distributions = [lst for lst in self.experts_distributions[expert_index] if lst] # Filter out empty lists

            kl_matrix = torch.zeros((len(new_distributions), len(learned_distributions)), device=self.device)
            for n, new_gmm in enumerate(new_distributions):
                new_mean = new_gmm.mu.data[0][0]
                # Versuche, die Dimensionen der Kovarianzmatrix anzupassen
                new_covariance = new_gmm.var.data[0].squeeze(0).squeeze(0)
                new_gauss = MultivariateNormal(new_mean, covariance_matrix=new_covariance)

                for o, old_gmm in enumerate(learned_distributions):
                    old_mean = old_gmm.mu.data[0][0]
                    # Versuche, die Dimensionen der Kovarianzmatrix anzupassen
                    old_covariance = old_gmm.var.data[0].squeeze(0).squeeze(0)
                    old_gauss = MultivariateNormal(old_mean, covariance_matrix=old_covariance)

                    kl_div = torch.distributions.kl_divergence(new_gauss, old_gauss)
                    kl_matrix[n, o] = kl_div

                    
            experts_mean_kl_div[expert_index] = self.calc_selection_criterion(kl_matrix)            
        
        if inverted:
            exp_to_finetune = torch.argmin(experts_mean_kl_div)
        else:
            exp_to_finetune = torch.argmax(experts_mean_kl_div) # Choose expert with the highest KL divergence
        return exp_to_finetune  

    def selection_ws_divergence(self, train_dataset, inverted=False):
        # Wasserstein divergence between the current task distribution and the distributions of the experts    
        experts_mean_ws_div = torch.zeros(self.max_experts, device=self.device)

        for expert_index in range(self.max_experts):
            new_distributions = self.create_distributions(train_dataset, expert_index) # Create distributions for the current task
            learned_distributions = [lst for lst in self.experts_distributions[expert_index] if lst] # Filter out empty lists

            ws_matrix = torch.zeros((len(new_distributions), len(learned_distributions)), device=self.device)
            for n, new_gmm in enumerate(new_distributions):

                new_gauss = new_gmm.mu.data[0][0].cpu().numpy()
                for o, old_gmm in enumerate(learned_distributions):
                    old_gauss = old_gmm.mu.data[0][0].cpu().numpy()

                    ws_div = wasserstein_distance(new_gauss, old_gauss)
                    ws_matrix[n, o] = ws_div

            experts_mean_ws_div[expert_index] = self.calc_selection_criterion(ws_matrix)
        if inverted:
            exp_to_finetune = torch.argmin(experts_mean_ws_div)
        else:
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
        model.to(self.device, non_blocking=True)

        # Finetune model on task:
        optimizer, sceduler = self.get_optimizer(num_param=model.parameters())
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        for epoch in range(self.finetune_epochs):
            print(f"Finetune epoch {epoch}")
            model.train()
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            
            running_loss = 0.0
            #num_train_loader = len(train_loader)
            #pbar = tqdm(enumerate(train_loader), desc=f"Epoch: {epoch}", total=num_train_loader)
            # Logger.instance().add_backend(TQDMLogger(pbar)) # Ist is LayUp, weiß nicht ob notwendig
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
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

                if self.accumulation_steps > 1:
                    if (i + 1) % self.accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()
                
                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{self.finetune_epochs}], Loss: {running_loss / len(train_loader)}")
        
        del train_loader
        
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
        The distributions are stored in self.experts_distributions[expert_index].
        """
        # Collect all images and labels needs much time
        all_images = []
        all_labels = []
        # Iterate over the DataLoader to collect images and labels
        #tbar = tqdm(train_dataset, desc="Collecting images and labels", total=len(train_dataset))
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
        model.to(self.device, non_blocking=True)

        gmms = []

        # Iterate over each class
        unique_labels = all_labels.unique()

        #pbar = tqdm(enumerate(unique_labels), desc=f"Compute distributions:", total=len(unique_labels))
        for class_label in unique_labels:
        #for _, class_label in pbar:
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

            if self.flipped_fetures:
                class_features = torch.full((2 * len(class_images), model.num_features), fill_value=-999999999.0, device=self.device)  
            else:
                class_features = torch.full((len(class_images), model.num_features), fill_value=-999999999.0, device=self.device)  

            # Process each image in the class (and its flipped version)
            for images in class_loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                features = model(images)
                class_features[from_: from_+bsz, : model.num_features] = features
                if self.flipped_fetures:
                    features = model(torch.flip(images, dims=(3,)))
                    class_features[from_+bsz: from_+2*bsz, :] = features
                    from_ += 2*bsz
                else:
                    from_ += bsz

            # Calculate distributions
            cov_type = "full" if self.use_multivariate else "diag"
            is_ok = False
            eps = 1e-8
            while not is_ok:
                try:
                    gmm = GaussianMixture(self.gmms, class_features.shape[1], covariance_type=cov_type, eps=eps).to(self.device, non_blocking=True)
                    gmm.fit(class_features, delta=1e-3, n_iter=100)
                except RuntimeError:
                    eps = 10 * eps
                    print(f"WARNING: Covariance matrix is singular. Increasing eps from: {1e-8:.8f} to: {eps:.8f} but this may hurt results")   ###Muss später wieder rein

                else:
                    is_ok = True
            #print("GMM fitted")
            if len(gmm.mu.data.shape) == 2:
                gmm.mu.data = gmm.mu.data.unsqueeze(1)

            gmm_shape = gmm.mu.data.shape
            assert len(gmm_shape) == 3
            assert gmm_shape[0] == 1
            assert gmm_shape[1] == 1
            assert gmm_shape[2] == model.num_features

            if self.cl_settig == "DIL":
                gmm.id = class_label.item()

            gmms.append(gmm)
        return gmms

    @torch.no_grad()
    def eval(self):
        self.backbone.eval()

    #@profile
    def predict_class_bayes_cil(self, features):
        #print("##### Classification (predict_class_bayes) ######")
        fill_value = -1e8
        log_probs = torch.full((features.shape[0], len(self.experts_distributions), len(self.experts_distributions[0])), fill_value=fill_value, device=features.device)
        # shape: (batch_size, num_experts, num_classes/num_distr.) 
        for expert_index in range(len(self.experts_distributions)):
            for c, class_gmm in enumerate(self.experts_distributions[expert_index]):

                if class_gmm == []: # Class not learned by this expert
                    continue
                else:
                    experts_class_features = features[:, expert_index]
                    probs = class_gmm.score_samples(experts_class_features)
                    #print(f"Probs Shape: {probs.shape}")
                    log_probs[:, expert_index, c] = probs
        #print("### Bayes ###")
        #print(f"Log_probs shape: {log_probs.shape}")

        output = []

        # Loop through each image in the batch
        for i in range(log_probs.shape[0]):
            # For each row in the slice (2D array), remove the fill_value (-1e8)
            row_result = []
            for j in range(log_probs.shape[2]):  # Loop through columns (2nd dimension)
                # Get the column values for the current column across all rows in the slice
                valid_values = log_probs[i, :, j][log_probs[i, :, j] != fill_value]

                # If there are valid values, get the first (they should only be one value per column)
                if len(valid_values) > 0:
                    row_result.append(valid_values[0])  # Keep only the first valid value for the column
            output.append(row_result)  # Add the processed row to the output
        filtered = torch.tensor(output)


        # Convert to a tensor
        #filtered = torch.tensor(output)
        #print(f"Filtered shape: {filtered.shape}")
        #
        #log_probs_softmaxed = torch.softmax(filtered/self.tau, dim=1).int() # tau? x= filtered/self.tau
        #padding = (0, self.num_classes - log_probs_softmaxed.shape[1])
        #synthetic_softmaxed_logits = torch.nn.functional.pad(log_probs_softmaxed, padding, "constant", 0).int()
        
        
        # which expert classifies:
        max_probs_per_expert, _ = log_probs.max(dim=2)  # Shape: [batchsize, experts]
        winning_expert = max_probs_per_expert.argmax(dim=1)  # Shape: [batchsize]
        winning_expert = winning_expert.to(torch.int64)
        self.task_winning_expert.append(winning_expert)

        return filtered

    def predict_class_bayes_dil(self, features):
        #print("##### Classification (predict_class_bayes) ######")
        fill_value = -1e8
           
        log_probs = torch.full((features.shape[0], len(self.experts_distributions), self.num_classes), fill_value=fill_value, device=features.device)
        # shape: (batch_size, num_experts, num_classes/num_distr.) 
        for expert_index in range(len(self.experts_distributions)):
            for c, class_gmm in enumerate(self.experts_distributions[expert_index]):

                if class_gmm == []: # from cil 
                    continue
                else:
                    experts_class_features = features[:, expert_index]
                    probs = class_gmm.score_samples(experts_class_features)
                    class_id = class_gmm.id
                    if torch.all(log_probs[:, expert_index, class_id] == fill_value):
                        log_probs[:, expert_index, class_id] = probs
                    else:
                        # ignore domain, use max prob for each class
                        log_probs[:, expert_index, class_id] = torch.max(log_probs[:, expert_index, class_id], probs)
        #print("### Bayes ###")
        #print(f"Log_probs shape: {log_probs.shape}")

        filtered = torch.max(log_probs, dim=1).values  # Shape: [batchsize, num_classes]

        
        # Convert to a tensor
        #filtered = torch.tensor(output)
        #print(f"Filtered shape: {filtered.shape}")
        #
        #log_probs_softmaxed = torch.softmax(filtered/self.tau, dim=1).int() # tau? x= filtered/self.tau
        #padding = (0, self.num_classes - log_probs_softmaxed.shape[1])
        #synthetic_softmaxed_logits = torch.nn.functional.pad(log_probs_softmaxed, padding, "constant", 0).int()
        
        
        # which expert classifies:
        max_probs_per_expert, _ = log_probs.max(dim=2)  # Shape: [batchsize, experts]
        winning_expert = max_probs_per_expert.argmax(dim=1)  # Shape: [batchsize]
        winning_expert = winning_expert.to(torch.int64)
        self.task_winning_expert.append(winning_expert)
        
        return filtered



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

    def save_experts_to_state_dict(self, destination):
        state_dict = {}
        for i, expert in enumerate(self.experts):
            for name, param in expert.items():
                state_dict[f"expert_{i}_{name}"] = param
            state_dict[f"expert_{i}_head"] = self.expert_heads[i]
        torch.save(state_dict, destination)
        return state_dict

    def load_experts_from_state_dict(self, state_dict):
        for i in state_dict.keys():
            parts = i.split('_')
            expert_index = int(parts[1])
            param_name = '_'.join(parts[2:])
            
            '''
            print("########## Before: ##########")
            print(f"Expert: {expert_index}")
            print(f"Param: {param_name}")
            print(f"Length of experts: {len(self.experts)}")
            print(f"Length of expert_heads: {len(self.expert_heads)}")
            '''

            # Experts does not exist
            if len(self.experts) <= expert_index or len(self.expert_heads) <= expert_index:
                self.backbone.head = nn.Identity() # Mismatch between head dimesnions with raw Backbone
                if param_name == 'head':
                    # self.expert_heads.append({param_name: state_dict[i]})
                    self.expert_heads.append(state_dict[i])
                    #print(state_dict[i])
                    #self.expert_heads[expert_index].load_state_dict(state_dict[i])

                else:
                    self.experts.append({param_name: state_dict[i]})
            # Experts already exists
            else:
                self.experts[expert_index][param_name] = state_dict[i]
                self.expert_heads[expert_index] = {param_name: state_dict[i]}
            '''
            print("########## After: ##########")
            print(f"Expert: {expert_index}")
            print(f"Param: {param_name}")
            print(f"Length of experts: {len(self.experts)}")
            print(f"Length of expert_heads: {len(self.expert_heads)}")
            print("#############################")
            '''

    def get_optimizer(self, num_param):
            """Returns the optimizer"""
            optimizer = None
            scheduler = None

            if self.use_adamw_and_cosinealing:
                optimizer = torch.optim.AdamW(num_param, lr=self.lr, weight_decay=self.args.weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.finetune_epochs)
                scheduler.last_epoch = -1
            else:
                optimizer = torch.optim.SGD(num_param, lr=self.lr, momentum=0.9, weight_decay=self.args.weight_decay) # weight_decay=wd? War vorher nicht drin               
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[60, 120, 160], gamma=0.1) #milestones=[60, 120, 160]
                

            return optimizer, scheduler
    
    def to(self, device=None):
        if device is None:
            device = self.device
        self.backbone.to(device, non_blocking=True)
        return self
    
    def children(self):
        return self.backbone.children()
