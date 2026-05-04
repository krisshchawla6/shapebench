#!/bin/bash
#SBATCH --job-name=bn_cmaes_tdcon_rerun
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/scratch/ShapeEvolve/environments/BlendedNet/results/logs_total_drag_constrained_runs/slurm_cmaes_rerun_%j.log

# CMA-ES constrained rerun — seeds 1 and 8 only (original runs crashed at 10 evals).
MAX_JOBS=2
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
source /scratch/ShapeEvolve/slurm_jobs/total_drag_constrained/_common.sh

X0_DESIGN=/scratch/ShapeEvolve/slurm_jobs/total_drag_constrained/cmaes_neutral_feasible_x0.json

for seed in 1 8; do
  launch "run_cmaes_shapebench_5_total_drag_constrained_seed${seed}_n1000" \
    --framework cmaes \
    --reward shapebench_5_total_drag_constrained \
    --n_calls 1000 \
    --sigma0 0.3 \
    --random_state "$seed" \
    --x0-design "$X0_DESIGN"
done

wait
echo "CMA-ES constrained rerun done."
