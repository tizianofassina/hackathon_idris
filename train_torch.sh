#!/bin/bash
#SBATCH -A ltc@h100
#SBATCH --job-name=TRAIN_FLOW_SINGLE
#SBATCH --partition=gpu_p6
#SBATCH --constraint="h100&prof2"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0-4

module purge
module load arch/h100
module load pytorch-gpu/py3/2.3.1

export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

BATCH_SIZES=(256 512 1024 2048 4096)
BS=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "[BASELINE SINGLE GPU] Array task $SLURM_ARRAY_TASK_ID — batch size = $BS"
echo "============================================================"

srun python -u train_torch.py --batch_size $BS