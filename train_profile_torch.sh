#!/bin/bash
#SBATCH -A ltc@a100
#SBATCH --job-name=PROFILE_TORCH
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=%x_%A.out

module purge
module load arch/a100
module load pytorch-gpu/py3/2.3.0

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun python -u train_profile_torch.py
