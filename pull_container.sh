#!/bin/bash
#SBATCH -A ltc@h100
#SBATCH --job-name=TRAIN_FLOW_DDP_PROF
#SBATCH --partition=gpu_p6
#SBATCH --constraint="h100&prof2"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0-5



singularity pull transformer_engine.sif docker://nvcr.io/nvidia/pytorch:26.02-py3