#!/bin/bash
#SBATCH -A ltc@a100
#SBATCH --job-name=TRAIN_FLOW_TORCH
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00                 
#SBATCH --output=%x_%A.out       

module purge
module load arch/a100
module load pytorch-gpu/py3/2.3.0 
module load nvidia-nsight-systems/2024.1.1.59

export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1

ln -sfn $JOBSCRATCH /tmp/nvidia

which nsys
nsys --version

srun nsys profile \
    -t cuda,nvtx,osrt,cudnn,cublas,nccl \
    --ncll-trace=all \
    --stats=true \
    -o "report_rank%q{SLURM_PROCID}" \
    python -u train_torch.py 
