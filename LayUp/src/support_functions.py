import torch 
from torch.utils.data import Subset
import pstats
import subprocess
import csv
import os
from datetime import datetime
import time
import json
'''
print(torch.version.cuda)  # Shows the CUDA version PyTorch was built with
print(torch.backends.cudnn.version())  # Shows the cuDNN version
print(torch.__version__)
print(torch.__config__.show())
print(torch.cuda.is_available())
'''

def display_profile(file_path="cProfile/profile_output.prof", sort_by="cumulative", lines=10):
    """
    Displays the contents of a .prof file using pstats.

    Parameters:
        file_path (str): Path to the .prof file.
        sort_by (str): Sorting criteria (e.g., 'cumulative', 'time', 'calls').
        lines (int): Number of top entries to display.
    """
    try:
        stats = pstats.Stats(file_path)
        stats.strip_dirs()  # Clean up file paths for readability
        stats.sort_stats(sort_by)
        stats.print_stats(lines)  # Print the top N lines
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def optimize_args(args):
    # Batchsize for 12 GB GPU
    if args.T == 10 and args.moe_max_experts <= 5 and args.batch_size == 32:
        optimized_batch_sizes = {
            "cifar100": 40,
            "imagenetr": 32,
            "cub": 48,
            "dil_imagenetr": 32,
            "imageneta": 48,
            "vtab": 32, # Sometimes 40
            "cars": 48,
            "omnibenchmark": 32,
            "limited_domainnet": 24,
            "cddb": 32 # not checked yet
        }
        args.batch_size = optimized_batch_sizes[args.dataset]
    else:
        print("Batch size can't be optimized for the current configuration.")

    # GPU memory 16GB:
    """
    gpu_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    if gpu_memory >= 16 * 1000**3:
        args.batch_size += 8
        if args.dataset in ["vtab"]:
            args.batch_size += 8
    """
    # Backbones
    opm_backbone = {
        "cifar100": "vit_base_patch16_224",
        "imagenetr": "vit_base_patch16_224_in21k",
        "cub": "vit_base_patch16_224_in21k",
        "dil_imagenetr": "vit_base_patch16_224_in21k",
        "imageneta": "vit_base_patch16_224_in21k",
        "vtab": "vit_base_patch16_224",
        "cars": "vit_base_patch16_224",
        "omnibenchmark": "vit_base_patch16_224",
        "limited_domainnet": "vit_base_patch16_224",
        "cddb": "vit_base_patch16_224" # not checked yet
    }
    #args.backbone = opm_backbone[args.dataset]

    
    return args


def shrink_dataset(dataset, fraction=0.25, num_images_per_class=50, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], DIL=False):
    """
    Shrinks the dataset to a fraction of its original size.
    
    """
    if float(fraction) == 1.0:
        print("Dataset size is unchanged.")
        return dataset
    
    if fraction > 1.0:
        class_counts = {cls: 0 for cls in classes}
        filtered_dataset = []
        print("+++++++")
        for image, label in dataset:
            # images per class per task
            if label in classes and class_counts[label] < num_images_per_class:
                filtered_dataset.append((image, label))
                class_counts[label] += 1

            # Stop if we have enough images for all classes
            if all(count >= num_images_per_class for count in class_counts.values()):
                break
        print("-------")
        return filtered_dataset
     

    # Calculate the number of samples to keep
    num_samples = int(len(dataset) * fraction)
    
    # Create a subset of the dataset
    indices = list(range(num_samples))
    subset = Subset(dataset, indices)
    
    return subset

def check_gpu_memory(i=None, extra_info=False):
    if torch.cuda.is_available():
        if i is not None:
            print(f"i: {i}")
        gpu_id = torch.cuda.current_device()
        if extra_info:
            print(f"GPU Name: \t{torch.cuda.get_device_name(gpu_id)}")
            print(f"Max Memory Allocated: \t{torch.cuda.max_memory_allocated(gpu_id) / 1e6:.2f} MB")
            print(f"Max Memory Cached: \t{torch.cuda.max_memory_reserved(gpu_id) / 1e6:.2f} MB")
        
        print(f"Memory Allocated: \t{torch.cuda.memory_allocated(gpu_id) / 1e6:.2f} MB")
        print(f"Memory Cached: \t\t{torch.cuda.memory_reserved(gpu_id) / 1e6:.2f} MB")
        print("-" * 50)
    else:
        print("CUDA is not available. Check your GPU setup.")
    
    
    return float(f"{torch.cuda.memory_allocated(torch.cuda.current_device()) / 1e6:.2f}")

def move_large_tensor_to_gpu():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a larger tensor (e.g., 1000x1000)
    tensor = torch.randn(1000, 1000)  # Example tensor of shape (1000, 1000)
    
    # Move the tensor to the GPU (or keep it on CPU if CUDA is unavailable)
    tensor = tensor.to(device, non_blocking=True)
    check_gpu_memory()

    
    # Print the tensor's shape and its device
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor is on device: {tensor.device}")


def log_gpustat(log_dir="local_logs", log_file="gpustat_log.csv"):
    try:
        # Create the directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Specify the full path to the log file
        log_path = os.path.join(log_dir, log_file)

        # Run gpustat and capture the output
        result = subprocess.run(["gpustat", "--json"], capture_output=True, text=True, check=True)
        
        # Parse JSON output
        data = json.loads(result.stdout)
        
        # Extract relevant information
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        gpu_data = []
        for gpu in data["gpus"]:
            if gpu["index"] == 2:
                continue
            gpu_info = {
                "Time": timestamp,
                "GPU ID": gpu["index"],
                "GPU Name": gpu["name"],
                "Temperature (°C)": gpu["temperature.gpu"],
                "Memory Used (MB)": gpu["memory.used"],
                "Memory Total (MB)": gpu["memory.total"],
                "GPU Utilization (%)": gpu["utilization.gpu"],
                "Power Draw (W)": gpu["power.draw"],
            }
            gpu_data.append(gpu_info)

        # Check if file exists
        file_exists = os.path.exists(log_path)
        
        # Log data to CSV
        with open(log_path, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=gpu_data[0].keys())

            # Write header if the file is new
            if not file_exists:
                writer.writeheader()

            # Write GPU data
            for gpu_info in gpu_data:
                writer.writerow(gpu_info)

    except subprocess.CalledProcessError as e:
        print(f"Error running gpustat: {e}")
    except json.JSONDecodeError as e:
        print(f"Error parsing gpustat output: {e}")


if __name__ == "__main__":
    pass