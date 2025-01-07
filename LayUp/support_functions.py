import torch 
from torch.utils.data import Subset

print(torch.version.cuda)  # Shows the CUDA version PyTorch was built with
print(torch.backends.cudnn.version())  # Shows the cuDNN version
print(torch.__version__)
print(torch.__config__.show())
print(torch.cuda.is_available())


def check_gpu_memory():
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        print(f"GPU Name: {torch.cuda.get_device_name(gpu_id)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(gpu_id) / 1e6:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(gpu_id) / 1e6:.2f} MB")
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(gpu_id) / 1e6:.2f} MB")
        print(f"Max Memory Cached: {torch.cuda.max_memory_reserved(gpu_id) / 1e6:.2f} MB")
    else:
        print("CUDA is not available. Check your GPU setup.")


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