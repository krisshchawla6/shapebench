#!/bin/bash
#SBATCH --job-name=gen50_i10_gaussain
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/Testing_paramters/AirFoil_becnhmark_change_gemini/logs/gen50_i10_gaussain_%j.out
#SBATCH --error=/scratch/Testing_paramters/AirFoil_becnhmark_change_gemini/logs/gen50_i10_gaussain_%j.err

set -e

# Set up paths
export PATH="$HOME/miniconda3/bin:/usr/bin:/bin:/usr/local/bin:$PATH"
REPO_DIR="/scratch/Testing_paramters/AirFoil_becnhmark_change_gemini"
ENV_DIR="$REPO_DIR/modified_env"
OUTPUT_DIR="$REPO_DIR/experiment_generate_gaussain"

# Create unique working directory for this job (prevents race conditions)
WORK_DIR="$REPO_DIR/work_${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"

# Create directories
mkdir -p "$REPO_DIR/logs"
mkdir -p "$OUTPUT_DIR"

# Activate conda environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate fenics_env

# Set PKG_CONFIG_PATH for FEniCS
export PKG_CONFIG_PATH="$HOME/miniconda3/envs/fenics_env/lib/pkgconfig:$PKG_CONFIG_PATH"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "Action: gaussain"
echo "Iterations: 80"
echo "Max Inspirations: 10 (grows 0→1→2...→10, then stays at 10)"
echo "Working Directory: $WORK_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Copy environment files to unique working directory (prevents race conditions)
echo "Setting up isolated workspace..."
cp -r "$ENV_DIR"/* "$WORK_DIR/"

# Move to working directory
cd "$WORK_DIR"

# Run the benchmark with gaussain action and 10 inspirations
echo "Starting benchmark run with debug mode..."
PYTHON_EXEC="$HOME/miniconda3/envs/fenics_env/bin/python"
PYTHONNOUSERSITE=1 "$PYTHON_EXEC" "$REPO_DIR/run_benchmark_action.py" \
    --action gaussain \
    --iterations 80 \
    --inspirations 5 \
    --debug

echo "=========================================="
echo "Job completed successfully"
echo "Results saved to: $OUTPUT_DIR"
echo "End time: $(date)"
echo "=========================================="

# Cleanup working directory
echo "Cleaning up workspace..."
cd "$REPO_DIR"
rm -rf "$WORK_DIR"
echo "Workspace cleaned."
