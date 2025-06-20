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
from torch import nn
from sklearn.model_selection import train_test_split
import timm
import pandas
import seaborn
import matplotlib.pyplot
import numpy as np
from tqdm import tqdm
from collections import Counter
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import csv



import wandb
from sklearn.metrics.pairwise import cosine_similarity

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
            #print("********")
            class_prob_means = torch.mean(y_hat, dim=0)
            print(y.tolist())
            print(pred_classes)
            print(mean_max_probs)
            print(f"Class prob. means{class_prob_means.tolist()}")
            # count occurence of means
            
            unique_values, counts = torch.unique(class_prob_means, return_counts=True)
            counts_dict = dict(zip(unique_values.tolist(), counts.tolist()))

            duplicates = {value: count for value, count in counts_dict.items() if count > 1}
            if duplicates:
                print(f"Duplicates?: {duplicates}")

            
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
        print(f"################## Task {t} ##################")
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




def calculate_similarity(feature1, feature2, metric='cosine'):
    """
    Calculates the similarity between two feature vectors.

    Args:
        feature1 (np.ndarray): First feature vector.
        feature2 (np.ndarray): Second feature vector.
        metric (str): The similarity metric to use ('cosine' or 'euclidean').

    Returns:
        float: The similarity score.
    """
    if metric == 'cosine':
        return cosine_similarity(feature1.reshape(1, -1), feature2.reshape(1, -1))[0][0]
    elif metric == 'euclidean':
        return -np.linalg.norm(feature1 - feature2) # Negative for consistency (higher value = more similar)
    else:
        raise ValueError(f"Unsupported similarity metric: {metric}")

def calculate_intra_class_similarity(features, labels, similarity_metric='cosine'):
    """
    Calculates the average intra-class similarity for a dataset.

    Args:
        features (np.ndarray): Array of feature vectors.
        labels (np.ndarray): Array of corresponding labels.
        similarity_metric (str): The similarity metric to use ('cosine' or 'euclidean').

    Returns:
        tuple: (dict, float) - A dictionary of per-class intra-class similarities
               and the overall average intra-class similarity.
    """
    intra_class_similarities = {}
    unique_labels = np.unique(labels)
    bar = tqdm(enumerate(unique_labels), desc="Calculating Intra-Class Similarity", total=len(unique_labels))
    for i, label in bar:
        class_features = features[labels == label]
        similarities = []
        for i in range(len(class_features)):
            for j in range(i + 1, len(class_features)):
                similarity = calculate_similarity(class_features[i], class_features[j], similarity_metric)
                similarities.append(similarity)
        if similarities:
            intra_class_similarities[label] = np.mean(similarities)
        else:
            intra_class_similarities[label] = 0.0

    overall_intra_class_similarity = np.mean(list(intra_class_similarities.values())) if intra_class_similarities else 0.0
    return intra_class_similarities, overall_intra_class_similarity

def calculate_inter_class_similarity(features, labels, similarity_metric='cosine'):
    """
    Calculates the average inter-class similarity for a dataset.

    Args:
        features (np.ndarray): Array of feature vectors.
        labels (np.ndarray): Array of corresponding labels.
        similarity_metric (str): The similarity metric to use ('cosine' or 'euclidean').

    Returns:
        float: The overall average inter-class similarity.
    """
    inter_class_similarities = []
    unique_labels = np.unique(labels)
    bar = tqdm(enumerate(unique_labels), desc="Calculating Inter-Class Similarity", total=len(unique_labels))
    # Iterate through all pairs of classes
    for i, _ in bar:
        for j in range(i + 1, len(unique_labels)):
            label1 = unique_labels[i]
            label2 = unique_labels[j]
            features_class1 = features[labels == label1]
            features_class2 = features[labels == label2]
            for feat1 in features_class1:
                for feat2 in features_class2:
                    similarity = calculate_similarity(feat1, feat2, similarity_metric)
                    inter_class_similarities.append(similarity)

    overall_inter_class_similarity = np.mean(inter_class_similarities) if inter_class_similarities else 0.0
    return overall_inter_class_similarity

def calculate_inter_class_similarity_vectorized(features, labels, similarity_metric='cosine'):
    """
    Calculates the average inter-class similarity for a dataset using vectorization.
    """
    inter_class_similarities = []
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    bar = tqdm(range(n_classes), desc="Calculating Inter-Class Similarity", total=n_classes)

    for i in bar:
        for j in range(i + 1, n_classes):
            label1 = unique_labels[i]
            label2 = unique_labels[j]
            features_class1 = features[labels == label1]
            features_class2 = features[labels == label2]

            if similarity_metric == 'cosine':
                similarity_matrix = cosine_similarity(features_class1, features_class2)
                inter_class_similarities.extend(similarity_matrix.flatten())
            elif similarity_metric == 'euclidean':
                # Calculate pairwise Euclidean distances and negate for consistency
                distances = np.linalg.norm(features_class1[:, np.newaxis, :] - features_class2[np.newaxis, :, :], axis=2)
                inter_class_similarities.extend((-distances).flatten())
            else:
                raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

    overall_inter_class_similarity = np.mean(inter_class_similarities) if inter_class_similarities else 0.0
    return overall_inter_class_similarity


def calculate_entropy(labels):
    """
    Calculates the entropy of a list or NumPy array of labels.

    Args:
        labels (list or np.ndarray): A list or array of labels.

    Returns:
        float: The entropy of the labels.
    """

    label_counts = Counter(labels)
    total_samples = len(labels)
    entropy = 0.0

    for count in label_counts.values():
        probability = count / total_samples
        entropy -= probability * math.log2(probability)

    return entropy

def calculate_feature_variance(features):
    """
    Calculates the variance of the features.

    Args:
        features (np.ndarray): Array of feature vectors.

    Returns:
        float: The variance of the features.
    """
    return np.var(features, axis=0).mean()  # Mean variance across all features

def visualize_csv_with_adjusted_size(csv_filepath, output_filepath="heatmap_adjusted.png"):
    try:
        df = pd.read_csv(csv_filepath, index_col=0)
    except FileNotFoundError:
        print(f"Fehler: Datei '{csv_filepath}' nicht gefunden.")
        return


    name_mapping = {
    "cars": "CARS",
    "cifar100": "CIFAR",
    "cub": "CUB",
    "imageneta": "IN-A",
    "imagenetr": "IN-R",
    "omnibenchmark": "OB",
    "dil_imagenetr": "IN-R (D)",
    "limited_domainnet": "S-DomainNet",
    "vtab": "VTAB",
    "cddb": "CDDB"
    }
    new_index_values = [name_mapping.get(idx, idx) for idx in df.index]
    df.index = new_index_values



    numeric_cols = df.select_dtypes(include=np.number).columns

    if not numeric_cols.empty:
        df_normalized = df[numeric_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
        cmap = LinearSegmentedColormap.from_list("mycmap", ["white", "lightblue", "darkblue"])

        # Erhöhe die Figurengröße, um die Kästchen größer zu machen
        plt.figure(figsize=(len(numeric_cols) * 2, len(df) * 1))

        sns.heatmap(df_normalized, annot=False, cmap=cmap, cbar=True, yticklabels=True)
        plt.title("Dataset metrics", fontsize=12) # Kleinere Schriftgröße für den Titel
        plt.xlabel("Metrics", fontsize=10) # Kleinere Schriftgröße für die X-Achse
        plt.ylabel("Datasets", fontsize=10) # Kleinere Schriftgröße für die Y-Achse
        plt.xticks(rotation=45, ha="right", fontsize=8) # Kleinere Schriftgröße für die X-Achsenbeschriftungen
        plt.yticks(fontsize=8) # Kleinere Schriftgröße für die Y-Achsenbeschriftungen
        plt.tight_layout()
        plt.savefig(output_filepath)
        print(f"Heatmap mit angepasster Größe und Schrift gespeichert als '{output_filepath}'.")
        plt.close()
    else:
        print("Keine numerischen Spalten zum Visualisieren gefunden.")

def visualize_csv_with_adjusted_size2(csv_filepath, output_filepath="heatmap_adjusted.png"):
    try:
        df = pd.read_csv(csv_filepath, index_col=0)
    except FileNotFoundError:
        print(f"Fehler: Datei '{csv_filepath}' nicht gefunden.")
        return

    # Mapping for y-axis (index)
    name_mapping = {
        "cars": "CARS",
        "cifar100": "CIFAR",
        "cub": "CUB",
        "imageneta": "IN-A",
        "imagenetr": "IN-R",
        "omnibenchmark": "OB",
        "dil_imagenetr": "IN-R (D)",
        "limited_domainnet": "S-DomainNet",
        "vtab": "VTAB",
        "cddb": "CDDB"
    }
    df.index = [name_mapping.get(idx, idx) for idx in df.index]

    # Mapping for x-axis (columns)
    column_mapping = {
        "label_entropy": "Label Entropy",
        "feature_entropy": "Feature Entropy",
        "feature_variance": "Feature Variance",
        "interclass_similarity": "Inter-Class Similarity",
        "intra_class_similarity": "Intra-Class Similarity",
        }
    df.columns = df.columns.str.strip()
    df.rename(columns=column_mapping, inplace=True)


    numeric_cols = df.select_dtypes(include=np.number).columns

    if not numeric_cols.empty:
        df_normalized = df[numeric_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
        cmap = LinearSegmentedColormap.from_list("mycmap", ["white", "lightblue", "darkblue"])

        plt.figure(figsize=(len(numeric_cols) * 2, len(df) * 1))
        sns.heatmap(df_normalized, annot=False, cmap=cmap, cbar=True,
                    yticklabels=True, xticklabels=True)

        # Labeling
        plt.title("Overview: Heatmap of different dataset metrics", fontsize=16)
        #plt.xlabel("Metrics", fontsize=14)
        #plt.ylabel("Datasets", fontsize=14)

        # Tick label styling
        plt.xticks(rotation=45, ha="right", fontsize=11)
        plt.yticks(fontsize=11)

        # Remove tick marks (but keep labels)
        plt.tick_params(axis='both', which='both', length=0)

        plt.tight_layout()
        plt.savefig(output_filepath)
        print(f"Heatmap gespeichert als '{output_filepath}'.")
        plt.close()
    else:
        print("Keine numerischen Spalten zum Visualisieren gefunden.")





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

    #use_moe(data_manager, train_transform, test_transform, args)

    
    values = [None] * 6
    values[0] = args.dataset

    feature_extractor = timm.create_model(args.backbone, pretrained=True).to(args.device)
    feature_extractor.head = nn.Identity()
    feature_extractor.eval()
    train_features = []
    train_labels = []
    bar = tqdm(enumerate(data_manager), desc="Extracting Features", total=len(data_manager))
    for i, (train_dataset, _) in bar:
        train_dataset.transform = train_transform
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        for images, labels in train_loader:  # Iterate through all batches in the loader
            images = images.to(args.device)
            features = feature_extractor(images)
            train_features.append(features.cpu().detach().numpy())
            train_labels.append(labels.cpu().detach().numpy())

    del feature_extractor

    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    print("Features shape:", train_features.shape)

    label_entropy = calculate_entropy(train_labels)
    print(f"Label Entropy: {label_entropy:.4f}")
    values[1] = label_entropy
    feature_entropy = calculate_entropy(train_features.flatten())
    print(f"Feature Entropy: {feature_entropy:.4f}")
    values[2] = feature_entropy

    # 4. Calculate intra-class similarity on the training set
    intra_class_similarities, overall_intra_similarity = calculate_intra_class_similarity(train_features, train_labels) # Use train_labels here
    print(f"Intra-Class Similarities per class: {intra_class_similarities}")
    print(f"Overall Intra-Class Similarity: {overall_intra_similarity:.4f}")
    values[5] = overall_intra_similarity
    
    # 5. Calculate inter-class similarity on the training set
    overall_inter_similarity = calculate_inter_class_similarity_vectorized(train_features, train_labels) # Use train_labels here
    print(f"Overall Inter-Class Similarity: {overall_inter_similarity:.4f}")
    values[4] = overall_inter_similarity

    # 6. Calculate feature variance
    feature_variance = calculate_feature_variance(train_features)
    print(f"Feature Variance: {feature_variance:.4f}")
    values[3] = feature_variance


    # Saving values
    save_path = "local_data/dataset_metrics.csv"
    with open(save_path, 'a', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(values)


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
    parser.add_argument('--use_adamw', help='Use AdamW optimizer', default=0, type=int)
    parser.add_argument('--use_cosine_annealing', help='Use cosine annealing', default=0, type=int)

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
    if args.finetune_method == "ssf":
        args.batch_size = 28


    #display_profile('cProfile/vtab3.prof')
    #exit(0)
    #assert False
    #cProfile.run('main(args)', 'cProfile/vtab4.prof')
    #print("#################")
    #display_profile('cProfile/vtab1.prof')
    #display_profile('cProfile/runtime_optim.prof')

    # Beispielhafte Verwendung:
    dateipfad = 'local_data/dataset_metrics.csv'  # Ersetze dies durch den Pfad zu deiner CSV-Datei
    ausgabepfad = './local_saved_graphics/dataset_metrics2.png' # Optional: Gib einen spezifischen Dateinamen an
    visualize_csv_with_adjusted_size2(dateipfad, ausgabepfad)
    exit(0)

    main(args)
    try:
        wandb_finish()
        print("WandB finished")
    except:
        print("Logger already closed")