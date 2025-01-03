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
        self.backbone_param_names = []
        self.num_classes = None
        self.alpha = 0.99 # Knowledge distillation parameter for the loss function. 1.0 means no knowledge distillation. 0.99 is from SEED.
        self.taskcla = []

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
            self.backbone = add_vpt(self.backbone)
        elif self.finetune_method == 'adapter':
            self.backbone = add_adapters(self.backbone)
        else: 
            raise ValueError('Invalid finetune method')

    def train_loop(self, t, train_dataset):
        self.taskcla.append([])

        if t < self.max_experts:
            print(f"Training expert {t + 1} on task {t}:")
            self.experts_distributions.append([])
            self.train_expert(train_dataset)
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

    def train_expert(self, train_dataset=None):
        self.add_expert()

        # Warum verändern sich die Tokens so wenig und warum sind die Zahken alle gleich?
        #test_expert_vpt = self.backbone.vpt_prompt_tokens.clone().detach()
        # will ich beim training einen Learning Rate Scheduler? SEED hat einen.
        # Welchen loss soll ich nehmen? LayUp oder SEED?

        # Add a linear head at the end of the network
        num_features = self.backbone.num_features
        num_classes = self.num_classes
        self.backbone.head = nn.Linear(num_features, num_classes)
 
        # Freeze the backbone parameters based on names except for the head
        for name, param in self.backbone.named_parameters():
            if name in self.backbone_param_names:
                param.requires_grad = False
            else:
                param.requires_grad = True
        for param in self.backbone.head.parameters():
            param.requires_grad = True

        # GPU/CPU
        model = self.backbone
        model.to(self.device)

        # Train model on task:
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        for epoch in range(self.finetune_epochs):
            model.train()
            running_loss = 0.0
            num_train_loader = len(train_loader)
            for batch_id, (inputs, labels) in enumerate(train_loader):
                print(f'Epoch: {epoch}, Batch?: {batch_id + 1}/{num_train_loader}')
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
                expert_parameters[name] = param # .clone() funktioniert nicht
            elif name == 'head.weight' or name == 'head.bias':
                expert_parameters[name] = param
            else:
                pass # Not saving backbone parameters except for the head
        self.experts.append(copy.deepcopy(expert_parameters))
        


        #print("VPT Tokens vorher:")
        #print(test_expert_vpt)
        #print("VPT Tokens nachher:")
        #print(self.backbone.vpt_prompt_tokens)


    def switch_to_expert(self, expert_index):
        for name, param in self.experts[expert_index].items():
            self.backbone.state_dict()[name].copy_(param)

    def forward(self, x):
        features = []
        for expert_index in range(len(self.experts)):
            self.switch_to_expert(expert_index)
            out = self.backbone(x)
            print(f'Feature Output shape: {self.backbone.num_features}')
            print(f'Label Output shape: {out.shape}')
            features.append(out)
        #return torch.stack(features, dim=1) # Stack the features to get a tensor of shape (?, num_experts, logits_of_classes) as in SEED


        # Stack the features to get a tensor of shape (1, num_experts, logits_of_classes)
        features = torch.stack(features, dim=1)
        
        # Compute the mean along the num_experts dimension (dimension 1)
        average_logits = torch.mean(features, dim=1)
        
        # Squeeze the tensor to remove the singleton dimension
        average_logits = average_logits.squeeze(0)
        
        print(f'Average Logits shape: {average_logits.shape}')
        print(average_logits)
        return average_logits


        # was ist mit dem head? Den speichere ich einfach mit. In SEED wird der glaube ich nicht gespeichert.
        # was mache ich mit dem output?
        # Der muss entweder geaveraged werden oder ich mache das mit bayes wie in SEED
        # zweiteres wäre besser. Kann ich einfach die Funktion aus SEED kopieren?
        # Als Prototyp nehme ich erstmal den Average.
    
    @torch.no_grad()
    def choose_expert_to_finetune(self, x):
        pass
    
    def finetune_expert(self, expert, trn_loader, val_loader):
        # Was ist mit dem Head?
        pass
    
    @torch.no_grad()
    def create_distributions(self, t, train_dataset):
        """ Create distributions for task t"""
        # benutze ich compose? bzw benutze ich für die Distributions die gleichen Transforms wie für das Training?
        self.backbone.eval()
        classes = train_dataset.num_classes

        
        #self.model.task_offset.append(self.model.task_offset[-1] + classes)
        
        #transforms = val_loader.dataset.transform
        transforms = train_dataset.transform
        print(transforms.transforms)
        print("-----------------")
        print(transforms.transforms[0])
        print("-----------------")
        dataloader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        '''
        for i, (x, y) in enumerate(dataloader):
            print(x.shape)
            print(y.shape)
            print(y)
            if i == 2:
                break
        return None
        '''
        '''
        # iterate over all experts
        for expert_index in range(min(self.max_experts, t+1)):
            eps = 1e-8
            self.switch_to_expert(expert_index)
            model = self.backbone
            for c in range(classes):
                #c = c + self.model.task_offset[t]
                train_indices = torch.tensor(trn_loader.dataset.labels) == c
                if isinstance(trn_loader.dataset.images, list):
                    train_images = list(compress(trn_loader.dataset.images, train_indices))
                    ds = ClassDirectoryDataset(train_images, transforms)
                else:
                    ds = trn_loader.dataset.images[train_indices]
                    ds = ClassMemoryDataset(ds, transforms)
                loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False) # num_workers=trn_loader.num_workers as parameter?
                from_ = 0
                class_features = torch.full((2 * len(ds), self.model.num_features), fill_value=-999999999.0, device=self.device)
                for images in loader:
                    bsz = images.shape[0]
                    images = images.to(self.device)
                    features = model(images)
                    class_features[from_: from_+bsz] = features
                    features = model(torch.flip(images, dims=(3,)))
                    class_features[from_+bsz: from_+2*bsz] = features
                    from_ += 2*bsz

                # Calculate distributions
                cov_type = "full" if self.use_multivariate else "diag"
                is_ok = False
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
                self.experts_distributions[expert_index].append(gmm)
        '''
    @torch.no_grad()
    def eval(self):
        self.backbone.eval()
    
    @torch.no_grad()
    def calculate_metrics(self, features, targets, t):
        pass
    
    @torch.no_grad() 
    def predict_class_bayes(self, t, features):
        pass   
    
    def criterion(self, outputs, targets, features=None, old_features=None):
        """Returns the loss value"""
        ce_loss = nn.functional.cross_entropy(outputs, targets, label_smoothing=0.0)
        if old_features is not None:  # Knowledge distillation loss on features
            kd_loss = nn.functional.mse_loss(features, old_features)
            total_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            return total_loss
        return ce_loss
    
    def _get_optimizer(self, num, wd, milestones=[60, 120, 160]):
        # unnötig?!
        pass
'''
    def freeze(self, fully=False):
        # Haben die einzelnen Blöcke eine Freeze Funktion?
        # NO !!
        call_in_all_submodules(self.backbone, "freeze", fully=fully)
'''    