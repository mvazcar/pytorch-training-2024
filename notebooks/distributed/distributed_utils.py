import os
import torch
import torch.distributed as dist


def setup():
    """
    Initializes torch.distributed using environment variables (SLURM compatible).
    Assumes:
        - MASTER_ADDR and MASTER_PORT are set in environment.
        - RANK and WORLD_SIZE are set by SLURM (or manually).
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")  # Automatically picks up env variables
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()
