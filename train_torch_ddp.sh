#!/bin/bash
#SBATCH -A ltc@a100
#SBATCH --job-name=TRAIN_FLOW_DDP
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1                      # Controlled internally by torchrun
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

# Create directories and clean local nsys cache
mkdir -p ./report
rm -rf .nsys_cache_ddp
mkdir -p .nsys_cache_ddp
export NSYS_CACHE_DIR="./.nsys_cache_ddp"

# ============================================================
# GPU Monitoring in background (Synchronized metrics)
# ============================================================
nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv \
    -l 1 \
    > ./report/ddp_training_gpu_metrics_${SLURM_JOB_ID}.csv &
NVIDIA_SMI_PID=$!

cleanup() {
    if [[ -n "$NVIDIA_SMI_PID" ]] && kill -0 "$NVIDIA_SMI_PID" 2>/dev/null; then
        kill "$NVIDIA_SMI_PID"
    fi
}
trap cleanup EXIT

# ============================================================
# Critical Intercept Wrapper to isolate profiling to Rank 0
# ============================================================
python_nsys_wrapper() {
    if [ "${LOCAL_RANK:-0}" -eq 0 ]; then
        exec nsys profile \
            -t cuda,nvtx,osrt,cudnn,cublas \
            --sample=cpu \
            --capture-range=cudaProfilerApi \
            --force-overwrite=true \
            -o "./report/ddp_profile_report_rank0" \
            python "$@"
    else
        exec python "$@"
    fi
}
export -f python_nsys_wrapper

# ============================================================
# Execution Launch via torchrun
# ============================================================
NUM_GPUS=$SLURM_GPUS_ON_NODE
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    --role python_nsys_wrapper \
    train_torch_ddp.py
