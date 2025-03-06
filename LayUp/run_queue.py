import multiprocessing
import os
import time

# Define job configurations (dataset and corresponding T values)
jobs = [
    {"dataset": "cifar"},
    {"dataset": "imagenet"},
    {"dataset": "cifar"},
    {"dataset": "imagenet"}
]

def train(gpu, job_queue):
    """Process function that continuously runs jobs on the specified GPU."""
    while not job_queue.empty():
        job = job_queue.get()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = f"CUDA_VISIBLE_DEVICES={gpu} python main.py --dataset {job['dataset']}"
        print(f"Starting on GPU {gpu}: {cmd}")
        os.system(cmd)  # Run the training script
        time.sleep(2)  # Short delay before picking the next job (optional)

# Create a job queue
job_queue = multiprocessing.Queue()
for job in jobs:
    job_queue.put(job)

# Start processes only on GPU 1 and GPU 2
p1 = multiprocessing.Process(target=train, args=(1, job_queue))
p2 = multiprocessing.Process(target=train, args=(2, job_queue))

p1.start()
p2.start()

p1.join()
p2.join()

print("All jobs have finished.")
