#!/bin/bash
#SBATCH --job-name=v3_drvr_n24_a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/ShapeEvolve/logs/%u/v3_drvr_n24_a2_%j.out
#SBATCH --error=/scratch/ShapeEvolve/logs/%u/v3_drvr_n24_a2_%j.err

set -e

REPO_DIR="/scratch/ShapeEvolve"
OUTPUT_DIR="$REPO_DIR/environments/DrivAer_Star/results/run_v3_dynamic_optimizer_cd_only_drivaer_star_attempt_2_flash_2_5_n2400"

mkdir -p "$OUTPUT_DIR"
mkdir -p "/scratch/ShapeEvolve/logs/$USER"

echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Date:         $(date)"
echo "Framework:    v3_dynamic_optimizer"
echo "Environment:  DrivAer_Star"
echo "Reward:       cd_only"
echo "Model:        gemini-2.5-flash"
echo "Iterations:   240 (n_evals=2400)"
echo "Attempt:      2"
echo "Output:       $OUTPUT_DIR"
echo "=========================================="

source "$REPO_DIR/venv/bin/activate"
cd "$REPO_DIR"

python run_benchmark.py \
    --config "/scratch/ShapeEvolve/sbatch/environments/DrivAer_Star/v3_dynamic_optimizer/v3_flash2_5_cd_only_drivaer_star_n2400.config.json" \
    --output-dir "$OUTPUT_DIR"

echo "=========================================="
echo "Job completed successfully"
echo "End time: $(date)"
echo "=========================================="
