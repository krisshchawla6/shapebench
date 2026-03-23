#!/bin/bash
#SBATCH --job-name=3d_vortexnet_bench
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/logs/3d_bench_%j.out
#SBATCH --error=/home/%u/logs/3d_bench_%j.err

set -e

export PATH="/home/ubuntu/miniconda3/bin:/usr/bin:/bin:/usr/local/bin:$PATH"
export PYTHONNOUSERSITE=1

REPO_DIR="/scratch/ShapeEvolve"
ENV_DIR="$REPO_DIR/environments/vlm_3d"
OUTPUT_DIR="$REPO_DIR/environments/vlm_3d/results/benchmark_results_3d_gaussain"

WORK_DIR="$REPO_DIR/work_${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
mkdir -p "$REPO_DIR/logs"
mkdir -p "$OUTPUT_DIR"

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate vortexnet

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "Action: gaussain"
echo "Iterations: 80"
echo "Max Inspirations: 5 (grows 0->1->2...->5)"
echo "Initialize N Sample: 6"
echo "AoA: 0 deg, Mach: 0.3"
echo "Working Directory: $WORK_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

echo "Setting up isolated workspace..."
cp -r "$ENV_DIR"/LLM_Actions "$WORK_DIR/"
cp "$ENV_DIR"/LLM_Actions/.env "$WORK_DIR/LLM_Actions/" 2>/dev/null || true

cd "$WORK_DIR"

python "$REPO_DIR/run_benchmark.py" \
    --action gaussain \
    --iterations 80 \
    --inspirations 5 \
    --initialize_n_sample 6 \
    --aoa 0 \
    --mach 0.3 \
    --debug

echo "=========================================="
echo "Job completed successfully"
echo "Results saved to: $OUTPUT_DIR"
echo "End time: $(date)"
echo "=========================================="

echo "Cleaning up workspace..."
cd "$REPO_DIR"
rm -rf "$WORK_DIR"
echo "Workspace cleaned."
