import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import random
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import functools
import os
import cProfile
import copy
import time
import wandb
import timm
from torch.utils.data import DataLoader
from src.backbone import get_backbone
from src.modules import CosineLinear
from src.layup import LayUP
from src.moe_seed import MoE_SEED
from src.data import (
    CILDataManager,
    DILDataManager,
    get_dataset,
    DATASET_MAP,
    make_test_transform_from_args,
    make_train_transform_from_args,
    update_transforms,
)
from src.logging import Logger, WandbLogger, ConsoleLogger, TQDMLogger
from torch.utils.data import Subset
from src.support_functions import check_gpu_memory, shrink_dataset, display_profile




from xplique.concepts import CraftTorch as Craft
from xplique.concepts import DisplayImportancesOrder
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms







def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(args):
    Logger.instance().add_backend(ConsoleLogger())
    if args.wandb_project is not None:
        Logger.instance().add_backend(
            WandbLogger(args.wandb_project, args.wandb_entity, args)
        )


def update_args(args):
    assert args.k >= 1 and args.k <= 12
    args.intralayers = [f"blocks.{11 - i}" for i in range(args.k)]

    args.aug_normalize = bool(args.aug_normalize)

    args.target_size = 224

    return args


def eval_datamanager(model, data_manager: CILDataManager, up_to_task: int, args):
    num_samples = {}
    results = {}
    for i, test_dataset in enumerate(data_manager.test_iter(up_to_task)):
        print(f"########## {i} ##########")
        task_res = eval_dataset(model, test_dataset, args)
        results[i] = task_res
        num_samples[i] = len(test_dataset)

    # convert to formated string
    final_results = {
        "after_task": up_to_task,
    }
    for k, v in results.items():
        for kk, vv in v.items():
            final_results[f"task_{k}/{kk}"] = vv

    # add mean
    keys_to_mean = results[0].keys()
    for key in keys_to_mean:
        final_results[f"task_mean/{key}"] = np.mean(
            [v[key] for v in results.values()]
        ).item()

        # also use weighted mean
        final_results[f"task_wmean/{key}"] = np.average(
            [v[key] for v in results.values()],
            weights=[num_samples[k] for k in results],
        )

    return final_results


@torch.no_grad()
def eval_dataset(model, dataset, args):
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    predictions = []
    labels = []

    for x, y in tqdm(dataloader, desc="Evaluating"):
        x = x.to(args.device)
        y = y.to(args.device)
        y_hat = model(x)
        predictions.append(y_hat.cpu().numpy())
        labels.append(y.cpu().numpy())
    
    # Some batches have only one image wich causes a wrong shape
    predictions = [np.expand_dims(pred, axis=0) if pred.ndim == 1 else pred for pred in predictions]

    predictions = np.concatenate(predictions, axis=0)
    print("########## Labels: ##########")
    print(labels)
    print(f"len: {[len(i) for i in labels]}")
    labels = np.concatenate(labels, axis=0)
    print(labels)
    print("########## END ##########")
    acc = (predictions.argmax(1) == labels).mean().item()
    return {"acc": acc}


def use_layup(data_manager, train_transform, test_transform, args):
    pass


class CustomModuleN_to_2048(nn.Module):
    def __init__(self, output_dim=2048, device="cuda"):
        super(CustomModuleN_to_2048, self).__init__()
        self.output_dim = output_dim
        self.device = device
        self.fc = nn.Linear(1, output_dim).to(device)  # Initialize with dummy input size
        self.relu = nn.ReLU().to(device)  # Add ReLU activation

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        input_dim = x.size(1)
        if self.fc.in_features != input_dim:
            self.fc = nn.Linear(input_dim, self.output_dim).to(self.device)  # Reinitialize with correct input size
        x = self.fc(x)
        x = self.relu(x)  # Apply ReLU activation

        # Check if all values are positive
        if not torch.all(x >= 0):
            print("Warning: Some values are not positive")

        return x

class AddMinValue(nn.Module):
    def __init__(self, device="cuda"):
        super(AddMinValue, self).__init__()
        self.device = device

    def forward(self, x):
        min_value = torch.min(x)
        x = x + min_value
        return x

def craft_switch_models(model, moe_model, device):
    return moe_model
    print("\n######## START ##########\n")
    
    print("EXPERT 2:")
    print([i for i, _ in moe_model.named_children()])
    for name, param in moe_model.named_children():
        if name == "blocks":
            print(f"blocks: {param[0]}")
        else:
            print(f"{name}: {param}")
    print("\n")
    
    #begining = nn.Sequential(moe_model.patch_embed, moe_model.pos_drop)

    model.stem = nn.Sequential(moe_model.patch_embed, moe_model.pos_drop, moe_model.patch_drop, moe_model.norm_pre)
    model.stages = moe_model.blocks
    
    #model.head = nn.Linear(2048, 100)
    model.final_conv = nn.Identity()
    model.final_act = nn.Sequential(moe_model.norm, moe_model.fc_norm, moe_model.head_drop)
    model.head = nn.Sequential(moe_model.head, nn.Linear(100, 100))
    
    model.to(device)

    print("RESNET50:")
    print([i for i, _ in model.named_children()])
    for name, param in model.named_children():
        if name == "stages":
            print(f"stages: {param[0]}")
        else:
            print(f"{name}: {param}")


    print("\n######## END ##########\n")

    return model





def use_moe(data_manager, train_transform, test_transform, args): # test_transform muss noch integriert werden
    moe_model = MoE_SEED(args)
    moe_model.save_backbone_param_names()
    moe_model.num_classes = data_manager.num_classes
    moe_model.logger = Logger.instance()
    moe_model.add_expert()

    for param_name, _ in moe_model.backbone.named_parameters():
        if param_name not in moe_model.backbone_param_names:
            moe_model.empty_expert[param_name] = copy.deepcopy(moe_model.backbone.state_dict()[param_name])
    
    
    # Load experts
    print("Loading experts")
    state_dict = torch.load("local_experts/experts_test.pth")
    print(state_dict.keys())
    moe_model.load_experts_from_state_dict(state_dict)
    
    moe_model.switch_to_expert(2) # Welcher expert classifiziert die beste classe?
    device = args.device

    # nodel vr is changed
    moe_model = moe_model.backbone.to(args.device)
    moe_model.device = args.device


    # loading any timm model
    model = timm.create_model('nf_resnet50.ra2_in1k', pretrained=True)
    model = model.to(device)
    model.device = device



    model = craft_switch_models(model, moe_model, device)
    model.to(device)

    g = nn.Sequential(*(list(model.children())[:-1])).to(device) # input to penultimate layer
    h = nn.Sequential(*(list(model.children())[-1:])).to(device) # penultimate layer to logits

    #g = nn.Sequential(CustomModuleN_to_2048())
    #h = nn.Sequential(CustomModuleN_to_2048()) # h funktioniert wie es soll

    #g.add_module("onlyPositiveValues", AddMinValue())
    #h.add_module("onlyPositiveValues", AddMinValue()) # h funktioniert wie es soll




    # Instanciate CRAFT
    craft = Craft(input_to_latent_model = g,
                latent_to_logit_model = h,
                number_of_concepts = 10,
                patch_size = 80,
                batch_size = 32,
                device = args.device)
    
    # Keine Ahnung was das ist
    config = resolve_data_config({}, model=model)
    #config["input_size"] = (3, 224, 224) # passt besser zu meinem ViT ?
    transform = create_transform(**config)
    to_pil = transforms.ToPILImage()

    # Extract images and labels for the first two classes
    first_two_classes = [0, 1]
    images_class_0 = []
    class0_id = 0
    images_class_1 = []
    for idx, (train_dataset, _) in enumerate(data_manager):
        if idx in first_two_classes:
            for img, label in train_dataset:
                if label == 0:
                    images_class_0.append(img)
                elif label == 1:
                    images_class_1.append(img)
        if idx >= 0: # change for more classes. But you have to initiate the arrays first
            break

    # Convert images to tensors before applying transform
    images_class_0 = [transforms.ToTensor()(img) for img in images_class_0]
    images_class_1 = [transforms.ToTensor()(img) for img in images_class_1]

    # loading some images of ? !
    images = images_class_0
    images_preprocessed = torch.stack([transform(to_pil(img)) for img in images], 0)
    images_preprocessed = images_preprocessed.to(model.device)
    print("Images shape:")
    print(images_preprocessed.shape)

    
    '''
    #
    #
    #
    rabbit_class = 330 # imagenet class for rabbit

    # loading some images of rabbits !
    images = np.load('rabbit.npz')['arr_0'].astype(np.uint8)
    images_preprocessed = torch.stack([transform(to_pil(img)) for img in images], 0)

    images_preprocessed = images_preprocessed.to(device)
    print("++++++++++++++++")
    crops, crops_u, concept_bank_w = \
    craft.fit(images_preprocessed, class_id=rabbit_class)

    print(crops.shape, crops_u.shape, concept_bank_w.shape)
    exit(0)
    #
    #
    #
    '''


    crops, crops_u, concept_bank_w = \
    craft.fit(images_preprocessed, class_id=class0_id)

    print(crops.shape, crops_u.shape, concept_bank_w.shape)


    exit(0)


def main(args):
    # get dataset and augmentations
    train_transform = make_train_transform_from_args(args)
    test_transform = make_test_transform_from_args(args)
    train_base_dataset, test_base_dataset = get_dataset(
        args.dataset, path=args.data_root
    )
    update_transforms(test_base_dataset, transform=test_transform)

    # for faster testing reduce dataset
    fraction = float(args.reduce_dataset)
    train_base_dataset = shrink_dataset(train_base_dataset, fraction)
    test_base_dataset = shrink_dataset(test_base_dataset, fraction)
    print(f"Reduced dataset size. Fraction {fraction}")



    # get datamanager based on ds
    data_manager = None
    if DILDataManager.is_dil(str(train_base_dataset)):
        print("DIL")
        data_manager = DILDataManager(
            train_base_dataset,
            test_base_dataset,
        )
    else:
        print("CIL")
        data_manager = CILDataManager(
            train_base_dataset,
            test_base_dataset,
            T=args.T,
            num_first_task=None if args.dataset != "cars" else 16,
            shuffle=True,
            seed=args.seed,
        )
        # log datamanager info
        Logger.instance().log(
            {
                "class_order": data_manager.class_order,
            },
            blacklist_types=[WandbLogger],
        )
        
    Logger.instance().log(
        {
            "num_classes": data_manager.num_classes,
        }
    )
    # LayUp
    if args.approach == "layup":
        print("Using LayUp")
        use_layup(data_manager, train_transform, test_transform, args)
    else:
        print("Using MoE")
        use_moe(data_manager, train_transform, test_transform, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--early_stopping", type=int, default=5)

    # data
    parser.add_argument(
        "--dataset", type=str, default="cifar100", choices=DATASET_MAP.keys()
    )
    parser.add_argument("--T", type=int, default=50)

    # model
    parser.add_argument(
        "--backbone",
        type=str,
        default="vit_base_patch16_224",
        choices=["vit_base_patch16_224", "vit_base_patch16_224_in21k"],
    )
    parser.add_argument(
        "--finetune_method",
        type=str,
        default="vpt",
        choices=["none", "adapter", "ssf", "vpt"],
    )
    parser.add_argument("--finetune_epochs", type=int, default=1)
    parser.add_argument("--k", type=int, default=6)

    # misc
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument(
        "--data_root",
        type=str,
        default="./local_data",
        help="Root directory for datasets",
    )

    # Approach
    parser.add_argument("--approach", type=str, default='moe', choices=['layup', 'moe'])
    parser.add_argument("--moe_max_experts", type=int, default=2)
    parser.add_argument("--reduce_dataset", default=0.25, help="Reduce dataset size for faster testing", type=float)
    parser.add_argument('--gmms', help='Number of gaussian models in the mixture', type=int, default=1)
    parser.add_argument('--use_multivariate', help='Use multivariate distribution', action='store_true', default=True)
    parser.add_argument('--selection_method', help='Method for expert selection for finetuning on new task', default="kl_div", choices=["random", "eucld_dist", "kl_div", "ws_div"])
    parser.add_argument('--classification', type=str, default='bayesian', choices=['average', "bayesian"]) # kommt am ende weg?
    parser.add_argument('--kd', help='Use knowledge distillation', action='store_true', default=False)

    # augmentations
    parser.add_argument("--aug_resize_crop_min", type=float, default=0.7)
    parser.add_argument("--aug_resize_crop_max", type=float, default=1.0)
    parser.add_argument("--aug_random_rotation_degree", type=int, default=0)
    parser.add_argument("--aug_brightness_jitter", type=float, default=0.1)
    parser.add_argument("--aug_contrast_jitter", type=float, default=0.1)
    parser.add_argument("--aug_saturation_jitter", type=float, default=0.1)
    parser.add_argument("--aug_hue_jitter", type=float, default=0.1)
    parser.add_argument("--aug_normalize", type=int, default=0)

    # logging
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()
    args = update_args(args)
    set_seed(args.seed)

    setup_logger(args)

    print("T = 50")
    main(args)