import torch 
from torch.utils.data import Subset
import pstats
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

def move_large_tensor_to_gpu():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a larger tensor (e.g., 1000x1000)
    tensor = torch.randn(1000, 1000)  # Example tensor of shape (1000, 1000)
    
    # Move the tensor to the GPU (or keep it on CPU if CUDA is unavailable)
    tensor = tensor.to(device)
    check_gpu_memory()

    
    # Print the tensor's shape and its device
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor is on device: {tensor.device}")

if __name__ == "__main__":
    pass