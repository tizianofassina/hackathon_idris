#!/bin/bash
#SBATCH -A ltc@a100
#SBATCH --job-name=TRAIN_FLOW_TORCH
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=%x_%A.out

module purge
module load arch/h100
module load pytorch-gpu/py3/2.3.0
module load nvidia-nsight-systems/2024.7.1.84

export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ln -sfn $JOBSCRATCH /tmp/nvidia

mkdir -p "$SLURM_SUBMIT_DIR/report"

nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv \
    -l 1 \
    > "$SLURM_SUBMIT_DIR/report/standard_training_gpu_metrics_${SLURM_JOB_ID}.csv" &
NVIDIA_SMI_PID=$!

cleanup() {
    if [[ -n "$NVIDIA_SMI_PID" ]] && kill -0 "$NVIDIA_SMI_PID" 2>/dev/null; then
        kill "$NVIDIA_SMI_PID"
    fi
}
trap cleanup EXIT

srun nsys profile -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o "$SLURM_SUBMIT_DIR/report/standard_run_${SLURM_JOB_ID}" python -u train_torch.py
