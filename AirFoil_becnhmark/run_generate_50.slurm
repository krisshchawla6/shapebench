#!/bin/bash
#SBATCH --job-name=gen50
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/LLM_Evolve/logs/gen50_%j.out
#SBATCH --error=/scratch/LLM_Evolve/logs/gen50_%j.err

set -e

# Set up paths
export PATH="$HOME/miniconda3/bin:/usr/bin:/bin:/usr/local/bin:$PATH"
REPO_DIR="/scratch/LLM_Evolve"
ENV_DIR="$REPO_DIR/AirFoil_becnhmark/modified_env"
OUTPUT_DIR="$REPO_DIR/AirFoil_becnhmark/benchmark_action_generate"

# Create directories
mkdir -p "$REPO_DIR/logs"
mkdir -p "$OUTPUT_DIR"

# Move to environment directory
cd "$ENV_DIR"

# Activate conda environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate fenics_env

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "Action: generate"
echo "Iterations: 50"
echo "Max Inspirations: 10 (grows from 0)"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Run the benchmark
PYTHON_EXEC="$HOME/miniconda3/envs/fenics_env/bin/python"
PYTHONNOUSERSITE=1 "$PYTHON_EXEC" run_benchmark_action.py --action generate --iterations 50 --inspirations 10 --output "$OUTPUT_DIR"

echo "=========================================="
echo "Job completed successfully"
echo "=========================================="
