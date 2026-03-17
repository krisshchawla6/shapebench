#!/bin/bash
#SBATCH --job-name=reflection_bench
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=17:00:00
#SBATCH --output=/scratch/Higher_dims/Islands_visual_stratedgy/testing_reflection/logs/reflection_bench_%j.out
#SBATCH --error=/scratch/Higher_dims/Islands_visual_stratedgy/testing_reflection/logs/reflection_bench_%j.err

set -e

# Set up paths
export PATH="$HOME/miniconda3/bin:/usr/bin:/bin:/usr/local/bin:$PATH"
REPO_DIR="/scratch/Higher_dims/Islands_visual_stratedgy/testing_reflection"
OUTPUT_DIR="$REPO_DIR/benchmark_results_gaussain"

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
echo "Action: gaussain (with reflection loop)"
echo "Iterations: 50"
echo "Inspirations: 5"
echo "Working Directory: $WORK_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Copy environment files to unique working directory (prevents race conditions)
echo "Setting up isolated workspace..."
cp -r "$REPO_DIR"/* "$WORK_DIR/" 2>/dev/null || true

# Move to working directory
cd "$WORK_DIR"

# Run the benchmark with reflection
echo "Starting reflection benchmark run..."
PYTHON_EXEC="$HOME/miniconda3/envs/fenics_env/bin/python"
PYTHONNOUSERSITE=1 "$PYTHON_EXEC" run_benchmark_action.py \
    --action gaussain \
    --iterations 50 \
    --inspirations 5 \
    --output "$OUTPUT_DIR"

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
