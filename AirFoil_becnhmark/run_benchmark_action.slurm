#!/bin/bash
#SBATCH --job-name=bench_%j
#SBATCH --output=/scratch/LLM_Evolve/AirFoil_becnhmark/logs/slurm_%j.out
#SBATCH --error=/scratch/LLM_Evolve/AirFoil_becnhmark/logs/slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=compute

# Get the action from command line argument
ACTION=$1
ITERATIONS=${2:-5}

# Create logs directory
mkdir -p /scratch/LLM_Evolve/AirFoil_becnhmark/logs

echo "=============================================="
echo "Starting benchmark for action: $ACTION"
echo "Iterations: $ITERATIONS"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=============================================="

# Activate conda environment
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate fenics_env

# Set PKG_CONFIG_PATH for FEniCS
export PKG_CONFIG_PATH="/home/ubuntu/miniconda3/envs/fenics_env/lib/pkgconfig:$PKG_CONFIG_PATH"

# Run from modified_env directory (required for imports)
cd /scratch/LLM_Evolve/AirFoil_becnhmark/modified_env

# Run the benchmark script with specified action
python ../run_benchmark_action.py --action $ACTION --iterations $ITERATIONS --inspirations 2 --debug

echo "=============================================="
echo "Benchmark finished for action: $ACTION"
echo "End time: $(date)"
echo "=============================================="
