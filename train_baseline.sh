#!/bin/bash
#SBATCH -A ltc@a100
#SBATCH --job-name=TRAIN_FLOW_BASELINE
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --output=%x_%A.out

module purge
module load arch/a100
module load pytorch-gpu/py3/2.3.0
module load nvidia-nsight-systems/2024.1.1.59


export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ln -sfn "$JOBSCRATCH" /tmp/nvidia || true

WORKERS_LIST=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)

for NUM_WORKERS in "${WORKERS_LIST[@]}"; do
    if [ "$NUM_WORKERS" -gt "$SLURM_CPUS_PER_TASK" ]; then
        echo "Skip: NUM_WORKERS=$NUM_WORKERS > SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
        continue
    fi

    echo "======================================"
    echo "NUM_WORKERS=$NUM_WORKERS"
    echo "======================================"

    NUM_WORKERS="$NUM_WORKERS" srun python -u train_baseline.py 
done
