import argparse
import random
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import sys
from torch import nn
import timm
import numpy as np
from tqdm import tqdm
import pandas as pd
import numpy as np


from torch.utils.data import DataLoader

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



# For saving the dataset features, not any metric computation




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

def wandb_finish():
    if len(Logger.instance()._backends) > 1 and isinstance(Logger.instance()._backends[1], WandbLogger):
        Logger.instance()._backends[1].close()
    else:
        print("No wandb logger to close")




def save_features_to_csv(features_list, labels_list, dataset_name, output_dir="local_data"):
    """
    Save features and labels to a CSV file.
    """

    # Stelle sicher, dass features_list eine Python-Liste ist
    if not isinstance(features_list, list):
        features_list = features_list.tolist()  # Konvertiere zu Liste, falls es ein NumPy Array ist

    # Annahme: Jede innere Liste hat die gleiche Länge und entspricht einer Zeile
    # und die Elemente der inneren Liste sollen separate Spalten werden.

    # Erstelle Spaltennamen für die Features
    num_features = len(features_list[0]) if features_list else 0
    feature_columns = [f'feature_{i}' for i in range(num_features)]

    # Erstelle ein Dictionary für den DataFrame
    data = {'dataset': [dataset_name] * len(labels_list),
            'label': [item[0] if isinstance(item, list) else item for item in labels_list]} # Annahme: Label ist das erste Element der inneren Liste
    for i, col in enumerate(feature_columns):
        data[col] = [item[i] if isinstance(item, list) and len(item) > i else None for item in features_list]

    df = pd.DataFrame(data)

    # Speichern als CSV-Datei
    csv_filename = f"./{output_dir}/{dataset_name}_class_features.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Daten erfolgreich als '{csv_filename}' gespeichert.")


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

    return train_features, train_labels





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

    # create features and labels
    features, labels = main(args)
    

    # Save features and labels to CSV
    save_features_to_csv(features, labels, args.dataset)







    try:
        wandb_finish()
        print("WandB finished")
    except:
        print("Logger already closed")