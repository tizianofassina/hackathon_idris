#!/bin/bash
#SBATCH -A ltc@a100
#SBATCH --job-name=TRAIN_FLOW_FSDP
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1                       # torchrun spawns the workers itself
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
    > logs/train_fsdp_gpu_${SLURM_JOB_ID}.csv &
NVIDIA_SMI_PID=$!

cleanup() {
    if [[ -n "$NVIDIA_SMI_PID" ]] && kill -0 "$NVIDIA_SMI_PID" 2>/dev/null; then
        echo "Stopping nvidia-smi monitoring (PID $NVIDIA_SMI_PID)"
        kill "$NVIDIA_SMI_PID"
    fi
}
trap cleanup EXIT

# ============================================================
# Intercept Python on Rank 0 for FSDP Profiling
# ============================================================
python_nsys_fsdp_wrapper() {
    # If torchrun sets LOCAL_RANK to 0, prepend nsys to the command
    if [ "${LOCAL_RANK:-0}" -eq 0 ]; then
        echo "Profiling FSDP Rank 0 with Nsight Systems..."
        exec nsys profile \
            -t cuda,nvtx,osrt,cudnn,cublas \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            --force-overwrite=true \
            --stats=true \
            -o "report_fsdp_${SLURM_JOB_ID}_rank0" \
            python "$@"
    else
        # Other ranks execute the python command normally without nsys overhead
        echo "Launching FSDP Rank ${LOCAL_RANK} without profiling..."
        exec python "$@"
    fi
}

# Export the function so it is inherited by the sub-shells spawned by torchrun
export -f python_nsys_fsdp_wrapper

# ============================================================
# Profiling + Training (FSDP)
# ============================================================
NUM_GPUS=$SLURM_GPUS_ON_NODE
echo "Launching FSDP training with $NUM_GPUS GPUs"

torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    --role python_nsys_fsdp_wrapper \
    train_torch_fsdp.py
