#!/bin/bash
#SBATCH --job-name=gen_bounds2_0
#SBATCH --output=gen_bounds2_0_%j.out
#SBATCH --error=gen_bounds2_0_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

echo "Starting job for gen_bounds2_0"
date

# Activate conda environment
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate fenics_env

# Run with timing
/usr/bin/time -v python run_case.py gen_bounds2_0.csv

echo "Job finished"
date
