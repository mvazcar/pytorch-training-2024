#!/bin/bash

#SBATCH --job-name="testrun"
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH -A csstaff

### OUTPUT ###
###SBATCH --output=/capstor/scratch/cscs/boeschf/HiRAD-Gen/logs/diffusion.log
###SBATCH --error=/capstor/scratch/cscs/boeschf/HiRAD-Gen/logs/diffusion.err
#SBATCH --output=/iopsstor/scratch/cscs/boeschf/pytorch-training/notebooks/distributed/train.log
##SBATCH --error=/iopsstor/scratch/cscs/${USER}/pytorch-training/notebooks/distributed/train.err

# Get master node.
# Get IP for hostname.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_ADDR="$(getent ahosts "$MASTER_ADDR" | awk '{ print $1; exit }')"

# Choose method to initialize dist in pytorch
export DISTRIBUTED_INITIALIZATION_METHOD=SLURM
export MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_RANK=$SLURM_LOCALID

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 
#export TRITON_HOME=/dev/shm/
export MPICH_GPU_SUPPORT_ENABLED=0
export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=64

srun -ul --environment=/iopsstor/scratch/cscs/${USER}/pytorch-training/notebooks/distributed/edf.toml bash -c "
    python main.py --method ddp --epochs 15
"


