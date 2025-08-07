import torch
import numpy as np
import random


def adjust_num_batches(worker_id: int):
    worker_info = torch.utils.data.get_worker_info()

    num_batches = worker_info.dataset.num_batches
    adjusted_num_batches = num_batches // worker_info.num_workers
    print(
        f"Adjusting worker {worker_id} num_batches from {num_batches} to {adjusted_num_batches}."
    )
    worker_info.dataset.num_batches = adjusted_num_batches

    # Randomly seeds each worker differently to ensure different random data loading
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed)
    random.seed(base_seed)
