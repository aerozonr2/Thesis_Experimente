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
import sys
import json
import subprocess


import wandb


'''
if torch.cuda.device_count() >= 2:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
'''

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

from src.support_functions import check_gpu_memory, shrink_dataset, display_profile, log_gpustat, optimize_args


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
        print(f"######## Evaluating task {i} ########")
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
        if key == "expert_percentages":
            continue
        # mean over all tasks
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

    for x, y in dataloader:
        x = x.to(args.device)
        y = y.to(args.device)
        y_hat = model(x)
        print(f"--- Real label: {y.tolist()[0]}")
        print("###############")

        predictions.append(y_hat.cpu().numpy())
        labels.append(y.cpu().numpy())
    
    # Some batches have only one image wich causes a wrong shape
    predictions = [np.expand_dims(pred, axis=0) if pred.ndim == 1 else pred for pred in predictions]

    predictions = np.concatenate(predictions, axis=0)
    #print("########## Labels: ##########")
    #print(labels)
    #print(f"len: {[len(i) for i in labels]}")
    
    #print("All Labels")
    labels = np.concatenate(labels, axis=0)
    #print(labels)
    #print("Pedictions:")
    #print(predictions)
    #print("########## END ##########")
    acc = (predictions.argmax(1) == labels).mean().item()

    #print(f"model.task_winning_expert: {model.task_winning_expert}")
    winning_experts = torch.cat(model.task_winning_expert, dim=0)
    # Count occurrences of each expert being the highest
    expert_counts = torch.bincount(winning_experts)
    # Convert to percentages
    expert_percentages = expert_counts.float() / winning_experts.shape[0] * 100
    # Format as a list of rounded percentages
    formatted_percentages = [f"{p:.1f}%" for p in expert_percentages.tolist()]
    print("Expert Contribution Percentages for task:", formatted_percentages)
    model.task_winning_expert = []
    
    #return {"acc": acc, "expert_percentages": formatted_percentages}    
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

        # first session adaptation
        if t == 0 and args.finetune_method != "none":
            train_dataset.transform = train_transform
            fsa(model, train_dataset, test_datatset, args)

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

    # Print model summary
    """
    Print an organized summary of the important parts of a Vision Transformer.
    Includes patch embedding, transformer blocks, classification head,
    parameters, and activation functions.
    """
    print("Vision Transformer Summary:")
    print("=" * 40)
    '''
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
    '''


def wandb_finish():
    if len(Logger.instance()._backends) > 1 and isinstance(Logger.instance()._backends[1], WandbLogger):
        Logger.instance()._backends[1].close()
    else:
        print("No wandb logger to close")

def use_moe(data_manager, train_transform, test_transform, args): # test_transform muss noch integriert werden
    model = MoE_SEED(args)
    model.save_backbone_param_names()
    model.num_classes = data_manager.num_classes
    model.logger = Logger.instance()
    model.add_expert()

    for param_name, _ in model.backbone.named_parameters():
        if param_name not in model.backbone_param_names:
            model.empty_expert[param_name] = copy.deepcopy(model.backbone.state_dict()[param_name])
    # Trainloop for all tasks
    for t, (train_dataset, test_datatset) in enumerate(data_manager): 
        if args.log_gpustat:
            log_gpustat()
        train_dataset.transform = train_transform
        print(f"# Task {t}")
        print(f"Train dataset: {len(train_dataset)}")
        print(f"Test dataset: {len(test_datatset)}")
        train_dataset.transform = train_transform
        model.train_loop(t=t, train_dataset=train_dataset)
        #if t == 9 and args.sweep_logging:
        #    Logger.instance().log({"GPU_memory": check_gpu_memory()})
        if args.log_gpustat:
            log_gpustat()
        

        # eval on all tasks up to t
        model.eval()
        model.freeze(fully=True)
        eval_res = eval_datamanager(model, data_manager, t, args)
        sorted_by_keys = dict(sorted(eval_res.items()))

        print("Sorted by keys:", sorted_by_keys)

        if args.log_gpustat:
            log_gpustat()
        # log results
        Logger.instance().log(eval_res)
        '''
        # Early stopping if accuracy is too low
        if float(eval_res["task_mean/acc"]) <= 0.20:
            wandb_finish()
            sys.exit()
        '''
        
        if args.exit_after_T != 0:
            if t == args.exit_after_T:
                print(f"Finished after T={args.exit_after_T}")
                wandb_finish()
                sys.exit()


    # Save experts
    #print("Saving experts")
    #model.save_experts_to_state_dict("local_experts/experts_good.pth")
    #model.save_experts_to_state_dict("local_experts/experts_test.pth")


        # VPT ist Deep wegen den 12 Layern, Es wird aber nur das erste benutzt. 12 Layer x 10 vpt_prompt_token_num x 768 embed_dim = VTP Tokens
        # ein experte so viel wie fsa benutzt 10 tokens, nicht 5, wie Kyra das meinte
        # Ich mache jetzt vpt_type="shallow", dadurch fällt der overhead weg, weil nur ein layer benutzt wird
        # Trotzdem ist der Experte immer noch doppelt so groß, 10 statt 5 tokens?
        # bayes optim für hyperparameter
        # gridsearch für alles andere


    # Print model summary
    
    # VPT                                                           !!!!!!!!! VPT
    """
    print(model.backbone.vpt_prompt_tokens) #                            
    weights_tensor = model.backbone.vpt_prompt_tokens.data
    # Count the number of weights that are 0
    num_zeros = torch.sum(weights_tensor == 0).item()
    # Count the number of weights that are not 0
    num_non_zeros = torch.sum(weights_tensor != 0).item()
    print(f"Number of weights that are 0: {num_zeros}")
    print(f"Number of weights that are not 0: {num_non_zeros}")
    """
    # Adapter                                                           !!!!!!!!! Adapter                                   
    """
    same_weights = torch.equal(test_adapter_weights, model.backbone.blocks[0].adaptmlp.down_proj.weight)
    print(f"Initial weights and modified weights are the same: {same_weights}")
    same_bias = torch.equal(test_adapter_bias, model.backbone.blocks[0].adaptmlp.down_proj.bias)
    print(f"Initial bias and modified bias are the same: {same_bias}")
    """
    # SSF                                                           !!!!!!!!! SSF
    """
    same_scale = torch.equal(test_ssf_scale, model.backbone.blocks[0].mlp.fc1_scale)
    print(f"Initial scale and modified scale are the same: {same_scale}")
    same_shift = torch.equal(test_ssf_shift, model.backbone.blocks[0].mlp.fc1_shift)
    print(f"Initial shift and modified shift are the same: {same_shift}")
    """



def main(args):
    # get dataset and augmentations
    train_transform = make_train_transform_from_args(args)
    test_transform = make_test_transform_from_args(args)
    train_base_dataset, test_base_dataset = get_dataset(
        args.dataset, path=args.data_root
    )
    update_transforms(test_base_dataset, transform=test_transform)

    # for faster testing reduce dataset
    num_images_per_class = 50
    fraction = float(args.reduce_dataset)
    train_base_dataset = shrink_dataset(train_base_dataset, fraction, num_images_per_class=num_images_per_class)
    test_base_dataset = shrink_dataset(test_base_dataset, fraction, num_images_per_class=num_images_per_class / 2)
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
    parser.add_argument("--moe_max_experts", type=int, default=5)
    parser.add_argument("--reduce_dataset", default=1.0, help="Reduce dataset size for faster testing", type=float)
    parser.add_argument('--gmms', help='Number of gaussian models in the mixture', type=int, default=1)
    parser.add_argument('--use_multivariate', help='Use multivariate distribution', action='store_true', default=True)
    parser.add_argument('--selection_method', help='Method for expert selection for finetuning on new task', default="kl_div", choices=["random", "around", "eucld_dist", "inv_eucld_dist", "kl_div", "inv_kl_div", "ws_div", "inv_ws_div"])
    parser.add_argument('--classification', type=str, default='bayesian', choices=['average', "bayesian"]) 
    parser.add_argument('--kd', help='Use knowledge distillation', default=False, type=bool)
    parser.add_argument('--kd_alpha', help='Alpha for knowledge distillation', default=0.99, type=float)
    parser.add_argument('--log_gpustat', help='Logging console -> gpustat', action='store_false', default=True)
    parser.add_argument('--sweep_logging', help='If you use a wandb sweep turn on for logging', default=False, type=bool)
    parser.add_argument('--exit_after_T', help='finish run after T=?', default=0, type=int)
    parser.add_argument('--kl_div_test', help='test', default=0, type=int, choices=[0, 1, 2])

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

    # For faster computation
    args = optimize_args(args)
    print(f"Args optimized: {args}")
    
    setup_logger(args)
    
    if args.sweep_logging:
        Logger.instance().add_backend(
                WandbLogger(args.wandb_project, args.wandb_entity, args)
            )
    

    if args.dataset == "vtab" and args.T > 50:
        print(f"Skipping run: dataset={args.dataset}, T={args.T} is too large")
        wandb_finish()
        exit(0)
    '''
    # Cifar100 wird zu oft gemacht
    if args.dataset == "cifar100":
        print(f"Skipping run: dataset={args.dataset}, should not be used")
        wandb_finish()
        exit(0)
    '''

    if torch.cuda.device_count() > 1:
        print("Specify GPU with: CUDA_VISIBLE_DEVICES=1/2 python main.py --...")
        sys.exit()



    #display_profile('cProfile/profile_output3.prof')
    #assert False
    #cProfile.run('main(args)', 'cProfile/runtime_optim.prof')
    #print("#################")
    #display_profile('cProfile/runtime.prof')
    #display_profile('cProfile/runtime_optim.prof')

    main(args)
    try:
        wandb_finish()
        print("WandB finished")
    except:
        print("Logger already closed")