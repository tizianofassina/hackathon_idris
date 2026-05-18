#!/bin/bash
#SBATCH -A ltc@a100
#SBATCH --job-name=TRAIN_FLOW_DDP
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1                       # torchrun
#SBATCH --gres=gpu:2                     
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=%x_%A.out

module purge
module load arch/a100
module load pytorch-gpu/py3/2.3.0
module load nvidia-nsight-systems/2024.7.1.84

export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ln -sfn $JOBSCRATCH /tmp/nvidia

mkdir -p logs

which nsys
nsys --version

# ============================================================
# nvidia-smi monitoring in background
# ============================================================
nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv \
    -l 5 \
    > logs/ddp_gpu_${SLURM_JOB_ID}.csv &
NVIDIA_SMI_PID=$!

cleanup() {
    if [[ -n "$NVIDIA_SMI_PID" ]] && kill -0 "$NVIDIA_SMI_PID" 2>/dev/null; then
        echo "Stopping nvidia-smi monitoring (PID $NVIDIA_SMI_PID)"
        kill "$NVIDIA_SMI_PID"
    fi
}
trap cleanup EXIT
# ============================================================
# Directory Cleanup and Preparation
# ============================================================
# Remove the old local cache if present to start fresh
rm -rf .nsys_cache_ddp
mkdir -p .nsys_cache_ddp

# ============================================================
# Profiling + Training (DDP) Configuration
# ============================================================
NUM_GPUS=$SLURM_GPUS_ON_NODE
echo "Launching DDP training with $NUM_GPUS GPUs under Nsys Profiler..."

# Set the cache directory via environment variable to keep nsys happy
export NSYS_CACHE_DIR="./.nsys_cache_ddp"
mkdir -p ./report

# Launch torchrun directly wrapped inside nsys profile
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=./report/ddp_profile_report \
    --capture-range=cudaProfilerApi \
    --sample=cpu \
    --force-overwrite=true \
    torchrun \
        --standalone \
        --nproc_per_node=$NUM_GPUS \
        train_torch_ddp.py
