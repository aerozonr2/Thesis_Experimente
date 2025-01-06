import argparse
import random
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import functools
import os
import cProfile


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


def fsa(model, train_dataset, test_dataset, args):
    model.freeze(fully=False)

    fsa_head = CosineLinear(
        in_features=model.backbone.num_features,
        out_features=train_dataset.num_classes,
        sigma=30,
    ).to(args.device)

    # set forward to use fsa head (instead of ridge)
    model.forward = functools.partial(model.forward_with_fsa_head, head=fsa_head)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters()},
            {"params": fsa_head.parameters()},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.finetune_epochs, eta_min=0.0
    )
    scheduler.last_epoch = -1

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # sanity
    # print all trainable parameters
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    for name, param in fsa_head.named_parameters():
        if param.requires_grad:
            print("head." + name)

    best_model_state_dict = None
    best_acc = 0.0
    epochs_no_improvement = 0

    for epoch in range(args.finetune_epochs):
        fsa_head.train()
        model.train()
        pbar = tqdm(dataloader, desc=f"Finetuning epoch {epoch}")
        Logger.instance().add_backend(TQDMLogger(pbar))
        for x, y in pbar:
            x = x.to(args.device)
            y = y.to(args.device)

            y_hat = model(x)
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            Logger.instance().log(
                {
                    "loss": loss.item(),
                    "tacc": (y_hat.argmax(1) == y).float().mean().item(),
                },
                blacklist_types=[ConsoleLogger],
            )

        scheduler.step()

        Logger.instance().pop_backend(TQDMLogger)
        eval_res = eval_dataset(model, test_dataset, args)
        acc = eval_res["acc"]
        eval_res = {"fsa_eval_" + k: v for k, v in eval_res.items()}

        if acc > best_acc:
            epochs_no_improvement = 0
            best_acc = acc
            best_model_state_dict = model.state_dict()
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= args.early_stopping:
            break

        Logger.instance().log(eval_res)

    # load best model
    model.load_state_dict(best_model_state_dict)

    # reset back to ridge forward
    model.forward = model.forward_with_ridge


def eval_datamanager(model, data_manager: CILDataManager, up_to_task: int, args):
    num_samples = {}
    results = {}
    for i, test_dataset in enumerate(data_manager.test_iter(up_to_task)):
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

    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)

    acc = (predictions.argmax(1) == labels).mean().item()

    return {"acc": acc}


def use_layup(data_manager, train_transform, test_transform, args):
    backbone = get_backbone(args.backbone, finetune_method=args.finetune_method)
    model = LayUP(
        backbone=backbone,
        intralayers=args.intralayers,
        num_classes=data_manager.num_classes,
    )
    model.to(args.device)

    for t, (train_dataset, test_datatset) in enumerate(data_manager):
        print(f"Task {t}")
        print(f"Train dataset: {len(train_dataset)}")
        print(f"Test dataset: {len(test_datatset)}")


        model.freeze(fully=True)
        model.eval()

        train_dataset.transform = test_transform

        # train Ridge regression
        model.update_ridge(
            DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
        )

        # eval on all tasks up to t
        eval_res = eval_datamanager(model, data_manager, t, args)
        # log results
        Logger.instance().log(eval_res)



    """
    Print an organized summary of the important parts of a Vision Transformer.
    Includes patch embedding, transformer blocks, classification head,
    parameters, and activation functions.
    """
    print("Vision Transformer Summary:")
    print("=" * 40)
    
    # Patch embedding layer
    print("Patch Embedding Layer:")
    print(model.patch_embed)
    print("-" * 40)
    
    # Transformer blocks
    print("Transformer Blocks:")
    for i, block in enumerate(model.blocks):
        print(f"Block {i + 1}:")
        print(f"  Attention: {block.attn}")
        print(f"  MLP: {block.mlp}")
        
        # Extract activation functions in the MLP
        activation_functions = [
            layer.__class__.__name__ for layer in block.mlp.fc2.children() 
            if isinstance(layer, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh))
        ]
        if not activation_functions:  # If no activations are direct children, check deeper
            activation_functions = [
                layer.__class__.__name__ for layer in block.mlp.children()
                if isinstance(layer, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh))
            ]
        print(f"  Activation Functions: {', '.join(activation_functions) if activation_functions else 'None Found'}")
    print("-" * 40)
    
    # Classification head
    print("Classification Head:")
    print(model.head)
    print("-" * 40)
    
    # Parameters summary
    print("Parameters Summary:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
    print("-" * 40)
    
    # Activation functions
    print("Activation Functions:")
    activations = []
    def find_activation_functions(module, activations):
        for child in module.children():
            if isinstance(child, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh)):
                activations.append(child.__class__.__name__)
            else:
                find_activation_functions(child, activations)
        return activations
    
    activations = find_activation_functions(model)
    print(", ".join(set(activations)))
    print("=" * 40)

def load_model():
    model = MoE_SEED(args)
    load_path = os.path.join('.gitignore', 'model.pth')
    model.backbone.load_state_dict(torch.load(load_path))
    print(f"Model loaded from {load_path}")
    return model

def shrink_dataset(dataset, fraction=0.25):
    """
    Shrinks the dataset to a fraction of its original size.
    
    """
    # Calculate the number of samples to keep
    num_samples = int(len(dataset) * fraction)
    
    # Create a subset of the dataset
    indices = list(range(num_samples))
    subset = Subset(dataset, indices)
    
    return subset


def use_moe(data_manager, train_transform, test_transform, args):
    model = MoE_SEED(args)
    model.save_backbone_param_names()
    model.num_classes = data_manager.num_classes




    # Trainloop for all tasks
    for t, (train_dataset, test_datatset) in enumerate(data_manager):
        
        train_dataset.transform = train_transform
        print(f"Task {t}")
        print(f"Train dataset: {len(train_dataset)}")
        print(f"Test dataset: {len(test_datatset)}")
        train_dataset.transform = train_transform
        model.train_loop(t=t, train_dataset=train_dataset)
        #return None
        '''
        # Save model incase of crash
        save_path = os.path.join('model_checkpoints', 'model.pth')
        torch.save(model.backbone.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        '''

        # Freeze Model. kommt drauf an welche methoden ich in der Evaluation aus LayUp brauche
        # Wahrschenlich model.eval() und model.freeze(fully=True)
        # Und natürlich Foward
        try:
            model.backbone.eval() # Funktioniert auch
            print("model.backbone.eval()")    
        except:
            model.eval()
            print("model.eval()") 



        # Aus LayUp
        # eval on all tasks up to t
        #eval_res = eval_datamanager(model, data_manager, t, args)
        # log results
        #Logger.instance().log(eval_res)


    # Print model summary
    """
    print("Model Summary:")

    print("Vision Transformer Summary:")
    print("=" * 40)
    
    # Patch embedding layer
    print("Patch Embedding Layer:")
    print(model.backbone.patch_embed)
    print("-" * 40)
    
    # Transformer blocks
    print("Transformer Blocks:")
    for i, block in enumerate(model.backbone.blocks):
        print(f"Block {i + 1}:")
        print(f"  Attention: {block.attn}")
        print(f"  MLP: {block.mlp}")
        
        # Extract activation functions in the MLP
        activation_functions = [
            layer.__class__.__name__ for layer in block.mlp.fc2.children() 
            if isinstance(layer, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh))
        ]
        if not activation_functions:  # If no activations are direct children, check deeper
            activation_functions = [
                layer.__class__.__name__ for layer in block.mlp.children()
                if isinstance(layer, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh))
            ]
        print(f"  Activation Functions: {', '.join(activation_functions) if activation_functions else 'None Found'}")
    print("-" * 40)
    
    # Classification head
    print("Classification Head:")
    print(model.backbone.head)
    print("-" * 40)
    
    # Parameters summary
    print("Parameters Summary:")
    for name, param in model.backbone.named_parameters():
        print(f"{name}: {param.shape}")
    print("-" * 40)
    
    # Activation functions
    print("Activation Functions:")
    activations = []
    def find_activation_functions(module, activations):
        for child in module.children():
            if isinstance(child, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh)):
                activations.append(child.__class__.__name__)
            else:
                find_activation_functions(child, activations)
        return activations
    
    activations = find_activation_functions(model.backbone, activations)
    print(", ".join(set(activations)))
    print("=" * 40)
    """
    

    # VPT Experts
    '''
    model = MoE_SEED(args)
    # Only change are addition of vprompt tokens at the beginning
    print("-----------------")
    model = model.train_expert()
    print(model.vpt_prompt_tokens)

    print("--------change module at the end to identity--------")
    test_expert_vpt = model.vpt_prompt_tokens.clone().detach()
    with torch.no_grad():
        model.vpt_prompt_tokens.fill_(1.0)
    print(model.vpt_prompt_tokens)

    print("--------change module back to original--------")
    with torch.no_grad():
        model.vpt_prompt_tokens.copy_(test_expert_vpt)
    print(model.vpt_prompt_tokens)
    '''

    # Adapter Experts
    '''
    model = MoE_SEED(args)
    for name, module in model.backbone.blocks[0].named_children():
        print(f"{name}: {module}")

    print("-----------------")

    model = model.train_expert()
    for name, module in model.blocks[0].named_children():
        print(f"{name}: {module}")

    print("--------change module at the end to identity--------")
    test_expert_adaptmlp = model.blocks[0].adaptmlp
    model.blocks[0].adaptmlp = nn.Identity()
    for name, module in model.blocks[0].named_children():
        print(f"{name}: {module}")
    
    print("--------change module back to original--------")
    model.blocks[0].adaptmlp = test_expert_adaptmlp
    for name, module in model.blocks[0].named_children():
        print(f"{name}: {module}")
    '''

    # SSF Experts
    '''
    model = MoE_SEED(args)
    for name, param in model.backbone.blocks[0].named_children():
        print(f"Parameter name: {name}, Shape: {param}")

    print("-----------------")
    model = model.train_expert()
    for name, param in model.blocks[0].named_children():
        print(f"Parameter name: {name}, Shape: {param}")

    print("--------mlp layers at the end--------")
    test_expert_scale_layer = model.blocks[0].mlp.fc1_scale
    test_expert_shift_layer = model.blocks[0].mlp.fc1_shift
    print(model.blocks[0].mlp.fc1_scale)
    print(model.blocks[0].mlp.fc1_shift)

    print("--------replacement of layer with ones--------")
    model.blocks[0].mlp.fc1_scale = torch.nn.Parameter(torch.ones_like(model.blocks[0].mlp.fc1_scale))
    model.blocks[0].mlp.fc1_shift = torch.nn.Parameter(torch.ones_like(model.blocks[0].mlp.fc1_scale))
    print(model.blocks[0].mlp.fc1_scale)
    print(model.blocks[0].mlp.fc1_shift)

    print("--------replacement of layer with original--------")
    model.blocks[0].mlp.fc1_scale = test_expert_scale_layer
    model.blocks[0].mlp.fc1_shift = test_expert_shift_layer
    print(model.blocks[0].mlp.fc1_scale)
    print(model.blocks[0].mlp.fc1_shift)
    '''



def main(args):
    # get dataset and augmentations
    train_transform = make_train_transform_from_args(args)
    test_transform = make_test_transform_from_args(args)
    train_base_dataset, test_base_dataset = get_dataset(
        args.dataset, path=args.data_root
    )
    update_transforms(test_base_dataset, transform=test_transform)

    # for faster testing reduce dataset
    if args.reduce_dataset == "True":
        train_base_dataset = shrink_dataset(train_base_dataset)
        test_base_dataset = shrink_dataset(test_base_dataset)
        print("Reduced dataset size")

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
    parser.add_argument("--batch_size", type=int, default=48)
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
    parser.add_argument("--finetune_epochs", type=int, default=2)
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
    parser.add_argument("--reduce_dataset", default="True")
    parser.add_argument('--gmms', help='Number of gaussian models in the mixture', type=int, default=1)
    parser.add_argument('--use_multivariate', help='Use multivariate distribution', action='store_true', default=True)
    parser.add_argument('--selection_method', help='Method for expert selection for finetuning on new task', default="random", choices=["random", "eucld_dist", "kl_div", "ws_div"])
    parser.add_argument('--moe_train_epochs', help='Num training epochs for expert initialisation', default=2)
    parser.add_argument('--moe_finetune_epochs', help='Num finetune epochs for expert finetuning', default=2)


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
    # cProfile.run('main(args)', 'cProfile/profile_output.prof')
    # als nächstes die Methoden die lange dauern mit cProfile direkt untersuchen