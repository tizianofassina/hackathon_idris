#!/bin/bash
#SBATCH -A ltc@cpu
#SBATCH --job-name=PULL_SING
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=%x_%A_%a.out


module load singularity/3.8.5  

singularity pull transformer_engine.sif docker://nvcr.io/nvidia/pytorch:26.02-py3