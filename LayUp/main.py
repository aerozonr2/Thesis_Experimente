import sys
import subprocess
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
from torch import nn


import wandb


from torch.utils.data import DataLoader

from src.backbone import get_backbone
from src.modules import CosineLinear
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

def install_timm(version):
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"timm=={version}"])


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
    if args.accumulation_steps > 1:
        args.batch_size = int(args.batch_size / args.accumulation_steps)
        print(f"Computing batch size reduced to: {args.batch_size}")
    
    dataset_T_map = {
        "dil_imagenetr": {"T": 15, "moe_max_experts": 7},
        "limited_domainnet": {"T": 6, "moe_max_experts": 3},
        "vtab": {"T": 5, "moe_max_experts": 3},
        "cddb": {"T": 5, "moe_max_experts": 3},
    }

    if args.dataset in dataset_T_map.keys():
        args.T = dataset_T_map[args.dataset]["T"]
        #args.moe_max_experts = dataset_T_map[args.dataset]["moe_max_experts"] immer 5!
        print(f"Dataset {args.dataset} has T={args.T} and moe_max_experts={args.moe_max_experts}")

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
        start_time = time.time()

        task_res = eval_dataset(model, test_dataset, args)
        results[i] = task_res
        num_samples[i] = len(test_dataset)

        print(f"Num samples: {num_samples[i]}")
        end_time = time.time()
        runtime = round(end_time - start_time, 4)
        print(f"Runtime {runtime} seconds")
        Logger.instance().log({"Task evaluation runtime": runtime})

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
    """
    print("--- GMM Parameters:")
    for expert_index in range(len(model.experts_distributions)):
        for class_index in range(len(model.experts_distributions[expert_index])):
            if model.experts_distributions[expert_index][class_index] == []:
                continue
            print(f"Expert {expert_index}, Class {class_index}:")
            print(f"Mean mean: {model.experts_distributions[expert_index][class_index].mu.data[0][0].mean():.6f}")
            print(f"Mean shape: {model.experts_distributions[expert_index][class_index].mu.data.shape}")
            print(f"Covariance mean: {model.experts_distributions[expert_index][class_index].var.data[0][0].mean():.6f}")
            print(f"Covariance shape: {model.experts_distributions[expert_index][class_index].var.data.shape}")
    print("-------------")
    """
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
        #
        only_one_class = False
        max_probs, _ = torch.max(y_hat, dim=1)
        sorted_tensor, sorted_index = torch.sort(y_hat, dim=1, descending=True)
        try:
            second_largest = sorted_tensor[:, 1]
            sorted_index = sorted_index[:, 1]
        except:
            only_one_class = True
        #print(f"Batch size: {x.shape[0]}")
        unique_labels = torch.unique(y)

        mean_max_probs = {}
        for label in unique_labels:
            label_indices = (y == label).nonzero(as_tuple=True)[0]
            label_indices = label_indices.to(y_hat.device)  # Move indices to the same device as logits
            #print(f"Label indices: {label_indices}")

            probs_for_label = max_probs[label_indices]
            #print(f"Probs for label {label.item()}: {probs_for_label}")
            if not only_one_class:
                second_largest_for_label = second_largest[label_indices]
                sorted_index_for_label = sorted_index[label_indices]

            # Move back to CPU for the final mean calculation as dictionaries are CPU-based
            mean_prob = probs_for_label.cpu().mean().item() if probs_for_label.numel() > 0 else 0.0
            #print(f"Mean prob for label {label.item()}: {mean_prob}")
            if not only_one_class:
                mean_second_largest = second_largest_for_label.cpu().mean().item() if second_largest_for_label.numel() > 0 else 0.0

            mean_max_probs[label.item()] = mean_prob

            #print(f"Mean max prob for label {label.item()}: {mean_max_probs[label.item()]}")
            if not only_one_class:
                mean_max_probs["second highest prob."] = mean_second_largest
                mean_max_probs["second highest class"] = sorted_index_for_label.cpu().tolist()
        
        pred_classes = torch.argmax(y_hat, dim=1).tolist()
        if pred_classes != y.tolist():
            #print(y_hat[0].tolist())
            print("********")
            class_prob_means = torch.mean(y_hat, dim=0)
            print(y.tolist())
            print(pred_classes)
            #print(mean_max_probs)
            #print(f"Class prob. means{class_prob_means.tolist()}")
            # count occurence of means
            
            unique_values, counts = torch.unique(class_prob_means, return_counts=True)
            counts_dict = dict(zip(unique_values.tolist(), counts.tolist()))

            duplicates = {value: count for value, count in counts_dict.items() if count > 1}
            if duplicates:
                #print(f"Duplicates?: {duplicates}")
                pass
            
        #print(y_hat)
        #print(y_hat.shape)
        log_probs_softmaxed = torch.softmax(y_hat, dim=1).int()
        padding = (0, model.num_classes - log_probs_softmaxed.shape[1])
        synthetic_softmaxed_logits = torch.nn.functional.pad(log_probs_softmaxed, padding, "constant", 0).int()
        y_hat = synthetic_softmaxed_logits
        #print(f"--- Real label: {y.tolist()[0]}")
        #print("###############")
        
        #print("--- Batch End ---\n")

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
        for x, y in pbar:
            x = x.to(args.device)
            y = y.to(args.device)

            y_hat = model(x)
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        scheduler.step()

        #eval_res = eval_dataset(model, test_dataset, args)
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

    # load best model
    model.load_state_dict(best_model_state_dict)

    # reset back to ridge forward
    model.forward = model.forward_with_ridge


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
        train_dataset.transform = train_transform
        print(f"################## Task {t} ##################")
        print(f"Train dataset: {len(train_dataset)}")
        print(f"Test dataset: {len(test_datatset)}")
        train_dataset.transform = train_transform
        model.train_loop(t=t, train_dataset=train_dataset)


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

        
        if args.exit_after_T != 1000:
            if t == args.exit_after_T:
                print(f"Exited after T={args.exit_after_T}")
                wandb_finish()
                sys.exit()
        if args.exit_after_acc != 0:
            if float(eval_res["task_mean/acc"]) <= args.exit_after_acc:
                print(f"Exited after acc={args.exit_after_acc}")
                wandb_finish()
                sys.exit()


    try:
        sweep_id = wandb.run.sweep_id
        run_id = wandb.run.id

        weights_dir = os.path.join("local_experts", sweep_id)
        os.makedirs(weights_dir, exist_ok=True)

        model.save_experts_to_state_dict(f"{weights_dir}/{run_id}.pth")
        print(f"Saved experts to {weights_dir}/{run_id}.pth")
    except:
        print("No saving experts")
    """
    # Save experts
    if float(eval_res["task_mean/acc"]) >= 0.7:
        model.save_experts_to_state_dict(f"local_experts/gut_{args.dataset}_{args.backbone}.pth")
    elif float(eval_res["task_mean/acc"]) >= 0.6:
        print("Saving experts")
        #model.save_experts_to_state_dict("local_experts/experts_good.pth")
        model.save_experts_to_state_dict(f"local_experts/okay_{args.dataset}_{args.backbone}.pth")
    else:
        pass
        # lohnt sich nicht
    """
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
    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--k", type=int, default=6)

    # misc
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument(
        "--data_root",
        type=str,
        default="./local_data",
        help="Root directory for datasets",
    )

    # Approach
    parser.add_argument("--moe_max_experts", type=int, default=5)
    parser.add_argument("--reduce_dataset", default=1.0, help="Reduce dataset size for faster testing", type=float)
    parser.add_argument('--gmms', help='Number of gaussian models in the mixture', type=int, default=1)
    parser.add_argument('--use_multivariate', help='Use multivariate distribution',type=int, default=0)
    parser.add_argument('--selection_method', help='Method for expert selection for finetuning on new task', default="around", choices=["random", "around", "eucld_dist", "inv_eucld_dist", "kl_div", "inv_kl_div", "ws_div", "inv_ws_div"])
    parser.add_argument('--kd', help='Use knowledge distillation', default=False, type=bool)
    parser.add_argument('--kd_alpha', help='Alpha for knowledge distillation', default=0.99, type=float)
    parser.add_argument('--log_gpustat', help='Logging console -> gpustat', action='store_false', default=False)
    parser.add_argument('--sweep_logging', help='If you use a wandb sweep turn on for logging', default=False, type=bool)
    parser.add_argument('--exit_after_T', help='finish run after T=?', default=1000, type=int)
    parser.add_argument('--selection_criterion', help='mean, min, max', default=0, type=int, choices=[0, 1, 2])
    parser.add_argument('--tau', help='In softmax', default=1.0, type=float)
    parser.add_argument('--exit_after_acc', help='finish run after acc=?', default=0.0, type=float)
    parser.add_argument('--trash_var', help='Does nothing', default=0.0, type=float)
    parser.add_argument('--use_adamw_and_cosinealing', help='Use AdamW optimizer and cosine annealing', default=0, type=int)
    parser.add_argument('--add_flipped_features', help='Adding flipped features in gmm fitting', default=0, type=int)
    parser.add_argument('--accumulation_steps', help='computing_bs = batch_size / accumulation_steps', default=1, type=int, choices=[1, 2])
    parser.add_argument('--bottleneck_dim', help='Bottleneck dimension for Adapters', default=64, type=int)

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
    #args = optimize_args(args)
    #print(f"Args optimized: {args}")
    
    setup_logger(args)
    


    if args.sweep_logging:
        Logger.instance().add_backend(
                WandbLogger(args.wandb_project, args.wandb_entity, args)
            )
    

    if args.dataset == "vtab" and args.T > 50:
        print(f"Skipping run: dataset={args.dataset}, T={args.T} is too large")
        wandb_finish()
        exit(0)
    

    if torch.cuda.device_count() > 1:
        print("Specify GPU with: CUDA_VISIBLE_DEVICES=d python main.py --...")
        # print all cuda visible devices ids
        sys.exit()

    if args.selection_method == "around" and args.selection_criterion != 0:
        print("Selection method 'around' not compatible with selection_criterion")
        try:
            wandb_finish()
        except:
            pass
        exit(0)
        
    if args.selection_method == "random" and args.selection_criterion != 0:
        print("Selection method 'random' not compatible with selection_criterion")
        try:
            wandb_finish()
        except:
            pass
        exit(0)

    if args.moe_max_experts > args.T and args.dataset != "cddb":
        print(f"Skipping run: moe_max_experts={args.moe_max_experts} > T={args.T}")
        try:
            wandb_finish()
        except:
            pass
        exit(0)


    #display_profile('cProfile/vtab3.prof')
    #exit(0)
    #assert False
    #cProfile.run('main(args)', 'cProfile/vtab4.prof')
    #print("#################")
    #display_profile('cProfile/vtab1.prof')
    #display_profile('cProfile/runtime_optim.prof')
    

    # half compute batch size of sweep by id
    halfed_sweep_ids = [
        "mrkfsh8f"
    ]


    accumulation_steps = os.environ.get("ACCUMULATION_STEPS")
    if accumulation_steps:
        args.accumulation_steps = int(accumulation_steps)
        args = update_args(args)



    try:
        id = wandb.run.sweep_id
        if id in halfed_sweep_ids:
            args.accumulation_steps = 2
            args = update_args(args)
    except:
        pass

    main(args)
    try:
        wandb_finish()
        print("WandB finished")
    except:
        print("Logger already closed")