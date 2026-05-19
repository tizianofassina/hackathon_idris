#!/bin/bash
#SBATCH -A ltc@a100
#SBATCH --job-name=TRAIN_FLOW_DDP
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
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
export OMP_NUM_THREADS=1 # deepen them 
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ln -sfn $JOBSCRATCH /tmp/nvidia

# Create report directory
mkdir -p "$SLURM_SUBMIT_DIR/report"

# ============================================================
# GPU Monitoring in background
# ============================================================
nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv \
    -l 1 \
    > "$SLURM_SUBMIT_DIR/report/ddp_training_gpu_metrics_${SLURM_JOB_ID}.csv" &
NVIDIA_SMI_PID=$!

cleanup() {
    if [[ -n "$NVIDIA_SMI_PID" ]] && kill -0 "$NVIDIA_SMI_PID" 2>/dev/null; then
        kill "$NVIDIA_SMI_PID"
    fi
}
trap cleanup EXIT

NUM_GPUS=$SLURM_GPUS_ON_NODE

srun nsys profile -t cuda,nvtx,osrt,cudnn,cublas \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --stop-on-exit=true \
    --force-overwrite=true \
    -o "$SLURM_SUBMIT_DIR/report/ddp_run_${SLURM_JOB_ID}" torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS train_torch_ddp.py \




nsys profile -t cuda,nvtx,osrt,cudnn,cublas \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --stop-on-exit=true \
    --force-overwrite=true \
    --stats=true \
    -o report/report_static torchrun \
    --standalone \
    --nproc_per_node=2 train_torch_ddp.py 
