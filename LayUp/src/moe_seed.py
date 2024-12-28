import torch
from argparse import ArgumentParser
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader
import timm
from .backbone.adapter import add_adapters
from .backbone.ssf import add_ssf
from .backbone.vpt import add_vpt
from .backbone.util import call_in_all_submodules





import copy
import random
import torch

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
        self.experts_distributions = []
        self.device = args.device
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.finetune_epochs = args.finetune_epochs
        self.args = args

    def add_expert(self):
        # Add Expert
        if self.finetune_method == 'ssf':
            self.backbone = add_ssf(self.backbone)
        elif self.finetune_method == 'vpt': 
            self.backbone = add_vpt(self.backbone)
        elif self.finetune_method == 'adapter':
            self.backbone = add_adapters(self.backbone)
        else: 
            raise ValueError('Invalid finetune method')

    def train_loop(self, t, train_dataset):
        if t < self.max_experts:
            print(f"Training expert {t + 1} on task {t}:")
            self.experts_distributions.append([])
            self.train_expert(t, train_dataset)
        else:
            if t == self.max_experts:
                print("All experts trained.")  
        '''
        if t >= self.max_experts:
            expert_to_finetune = self.choose_expert_to_finetune(t, trn_loader, val_loader)
            print(f"Finetuning expert {expert_to_finetune} on task {t}:")
            self.finetune_expert(t, expert_to_finetune, trn_loader, val_loader)

        print(f"Creating distributions for task {t}:")
        self.create_distributions(t, trn_loader, val_loader)
        '''

    def train_expert(self, t=None, train_dataset=None):
        self.add_expert()

        # model.add_head() ? (last layer?)

        # Test
        test_expert_vpt = self.backbone.vpt_prompt_tokens.clone().detach()


        #moodel.train() ist aber wichtig. ich muss also das ViT freezen und denn rest so lassen
        # gut: 
        # model.train()
        # freeze_backbone()
        # doof:
        #self.freeze()
        #self.unfreeze_peft()



        # GPU/CPU
        model = self.backbone
        model.to(self.device)

        # Train model on task:
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        for epoch in range(self.finetune_epochs):
            model.train() # freeze() und unfreeze_peft sollten das übernehmen
            running_loss = 0.0
            num_train_loader = len(train_loader)
            for batch_id, (inputs, labels) in enumerate(train_loader):
                print(f'Epoch: {epoch}, Batch?: {batch_id + 1}/{num_train_loader}')
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{self.finetune_epochs}], Loss: {running_loss / len(train_loader)}")

        
        print("VPT Tokens vorher:")
        print(test_expert_vpt)
        print("VPT Tokens nachher:")
        print(self.backbone.vpt_prompt_tokens.clone().detach())
        print("VPT Tokens nachher2:")
        print(self.backbone.vpt_prompt_tokens)


    
    @torch.no_grad()
    def choose_expert_to_finetune(self, x):
        pass
    
    def finetune_expert(self, expert, trn_loader, val_loader):
        pass
    
    @torch.no_grad()
    def create_distributions(self, t, trn_loader, val_loader):
        pass
        
    @torch.no_grad()
    def eval(self, t, val_loader):
        pass
    
    @torch.no_grad()
    def calculate_metrics(self, features, targets, t):
        pass
    
    @torch.no_grad() 
    def predict_class_bayes(self, t, features):
        pass   
    
    def criterion(self, t, outputs, targets, features=None, old_features=None):
        # unnötig?
        pass
    
    def _get_optimizer(self, num, wd, milestones=[60, 120, 160]):
        # unnötig?
        pass

    def freeze(self, fully=False):
        call_in_all_submodules(self, "freeze", fully=fully)
    
    def unfreeze_peft(self):
        # wahrscheinlich mit model.layer.parameters[i].requires_grad = True/False
        if self.finetune_method == 'ssf':
            pass
        elif self.finetune_method == 'vpt':
            pass
        elif self.finetune_method == 'adapter':
            pass
        else:
            raise ValueError('Invalid finetune method')