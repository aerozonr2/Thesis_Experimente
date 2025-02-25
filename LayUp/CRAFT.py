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



def use_moe(data_manager, train_transform, test_transform, args): # test_transform muss noch integriert werden
    model = MoE_SEED(args)
    model.save_backbone_param_names()
    model.num_classes = data_manager.num_classes
    model.logger = Logger.instance()
    model.add_expert()

    for param_name, _ in model.backbone.named_parameters():
        if param_name not in model.backbone_param_names:
            model.empty_expert[param_name] = copy.deepcopy(model.backbone.state_dict()[param_name])
    
    state_dict = torch.load("local_experts/experts_good.pth")
    print("state_dict.keys():")
    print(state_dict.keys())
    model.load_experts_from_state_dict(state_dict)
    print("Experts loaded")
    
    
    model.switch_to_expert(0) # Welcher expert classifiziert die beste classe?

    '''
    for i, e in model.backbone.state_dict().items():
        print(i)
        if i == "vpt_prompt_tokens":
            print(e)
        print("***")

    assert False
    '''
    
    

    # Model var is changed!
    model = model.backbone
    # test ob .children() funktioniert!



    for i, e in model.named_children():
        print(i)
        print(e)
        print("***")
    assert False






    g = nn.Sequential(*(list(model.children())[:-1])) # input to penultimate layer
    h = nn.Sequential(*(list(model.children())[-1:])) # penultimate layer to logits
    # Instanciate CRAFT
    craft = Craft(input_to_latent_model = g,
                latent_to_logit_model = h,
                number_of_concepts = 10,
                patch_size = 80,
                batch_size = 64,
                device = args.device)
    
    #print(*(list(model.children())[:-1]))
    for i,  in list(model.named_children())[:-1]:
        print(i)
        print("***")
    
    
    #print(*(list(model.children())[-1:]))
    print("################")
    assert False

    # Keine Ahnung was das ist
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    to_pil = transforms.ToPILImage()



    rabbit_class = 1 # class with best accuracy

    # loading some images of rabbits !
    images = np.load('rabbit.npz')['arr_0'].astype(np.uint8)
    images_preprocessed = torch.stack([transform(to_pil(img)) for img in images], 0)

    images_preprocessed = images_preprocessed.to(model.device)

    print(images_preprocessed.shape)





    crops, crops_u, concept_bank_w = \
    craft.fit(images_preprocessed, class_id=rabbit_class)

    crops.shape, crops_u.shape, concept_bank_w.shape


    assert False
    # Trainloop for all tasks
    for t, (train_dataset, test_datatset) in enumerate(data_manager): 
        train_dataset.transform = train_transform
        print(f"# Task {t}")
        print(f"Train dataset: {len(train_dataset)}")
        print(f"Test dataset: {len(test_datatset)}")
        train_dataset.transform = train_transform
        model.train_loop(t=t, train_dataset=train_dataset)
        

        # eval on all tasks up to t
        model.eval()
        model.freeze(fully=True)
        eval_res = eval_datamanager(model, data_manager, t, args)
        # log results
        Logger.instance().log(eval_res)



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
    parser.add_argument("--T", type=int, default=10)

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


    main(args)