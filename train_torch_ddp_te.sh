#!/bin/bash
#SBATCH -A ltc@h100
#SBATCH --job-name=TRAIN_FLOW_DDP_PROF
#SBATCH --partition=gpu_p6
#SBATCH --constraint="h100&prof2"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0-4

module purge
module load arch/h100
module load pytorch-gpu/py3/2.3.1
module load nvidia-nsight-systems/2024.7.1.84

export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

NUM_GPUS=2
BATCH_SIZES=(256 512 1024 2048 4096 8192)

BS=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}


ln -sfn $JOBSCRATCH /tmp/nvidia

# Create report directory
mkdir -p "$SLURM_SUBMIT_DIR/report"

srun nsys profile -t cuda,nvtx,osrt,cudnn,cublas \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --stop-on-exit=true \
    --force-overwrite=true \
    --stats=true \
    -o report/report_2gpus_te${BS} torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS train_torch_ddp_te.py --batch_size $BS
