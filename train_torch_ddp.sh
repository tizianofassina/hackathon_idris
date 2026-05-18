#!/bin/bash
#SBATCH -A ltc@a100
#SBATCH --job-name=TRAIN_FLOW_DDP
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1                      # Controlled by torchrun
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

# Ensure output report directory exists
mkdir -p ./report
rm -rf .nsys_cache_ddp
mkdir -p .nsys_cache_ddp
export NSYS_CACHE_DIR="./.nsys_cache_ddp"

which nsys
nsys --version

# ============================================================
# Start background nvidia-smi monitoring
# ============================================================
nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv \
    -l 1 \
    > ./report/ddp_training_gpu_metrics_${SLURM_JOB_ID}.csv &
NVIDIA_SMI_PID=$!
echo "Started nvidia-smi monitoring (PID $NVIDIA_SMI_PID)"

cleanup() {
    if [[ -n "$NVIDIA_SMI_PID" ]] && kill -0 "$NVIDIA_SMI_PID" 2>/dev/null; then
        echo "Stopping nvidia-smi monitoring (PID $NVIDIA_SMI_PID)"
        kill "$NVIDIA_SMI_PID"
    fi
}
trap cleanup EXIT

# ============================================================
# Create a local bash function to intercept python on Rank 0
# ============================================================
python_nsys_wrapper() {
    if [ "${LOCAL_RANK:-0}" -eq 0 ]; then
        echo "Profiling Rank 0 with Nsight Systems..."
        exec nsys profile \
            -t cuda,nvtx,osrt,cudnn,cublas \
            --sample=cpu \
            --capture-range=cudaProfilerApi \
            --force-overwrite=true \
            -o "./report/ddp_profile_report_rank0" \
            python "$@"
    else
        echo "Launching Rank ${LOCAL_RANK} without profiling..."
        exec python "$@"
    fi
}
export -f python_nsys_wrapper

# ============================================================
# Execution Launch via torchrun
# ============================================================
NUM_GPUS=$SLURM_GPUS_ON_NODE
echo "Launching DDP training with $NUM_GPUS GPUs..."

torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    --role python_nsys_wrapper \
    train_torch_ddp.py
