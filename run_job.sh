#!/bin/bash
#SBATCH --job-name=llm_evolve
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Exit on error
set -e

# Arguments
PYTHON_SCRIPT=${1:-""}
CSV_INPUT=${2:-""}
OUTPUT_DIR_NAME=${3:-"hpc_test"}

if [ -z "$PYTHON_SCRIPT" ]; then
    echo "Error: No Python script specified"
    exit 1
fi

# Set up paths
export PATH="$HOME/miniconda3/bin:/usr/bin:/bin:/usr/local/bin:$PATH"
REPO_DIR="/scratch/LLM_Evolve"
ENV_DIR="$REPO_DIR/AirFoil_becnhmark/modified_env"
OUTPUT_DIR="$REPO_DIR/$OUTPUT_DIR_NAME"

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
echo "Script: $PYTHON_SCRIPT"
echo "Input: $CSV_INPUT"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Run the Python script
# We pass the CSV input to the script
# Explicitly use the conda env python to avoid path issues
PYTHON_EXEC="$HOME/miniconda3/envs/fenics_env/bin/python"
PYTHONNOUSERSITE=1 "$PYTHON_EXEC" "$PYTHON_SCRIPT" "$CSV_INPUT"

# Move results to the requested output folder
echo "Moving results to $OUTPUT_DIR..."
# The environment saves results in 'save/' inside its directory
if [ -d "save" ]; then
    cp -r save/* "$OUTPUT_DIR/"
    echo "Results moved."
else
    echo "Warning: 'save' directory not found."
fi

echo "=========================================="
echo "Job completed successfully"
echo "=========================================="
