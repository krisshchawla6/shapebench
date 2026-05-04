#!/bin/bash
#SBATCH --job-name=bn_pso_tdcon
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/scratch/ShapeEvolve/environments/BlendedNet/results/logs_total_drag_constrained_runs/slurm_pso_%j.log

# GA_parallel (PSO) — 20 runs (10 seeds × 2 constrained total-drag rewards)
# AR ≥ 2.5 constraint blocks surrogate-exploitation corner.
# Each run spawns 20 worker processes; 4 concurrent × 20 workers × 1 thread = 80
MAX_JOBS=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
source /scratch/ShapeEvolve/slurm_jobs/total_drag_constrained/_common.sh

for REWARD in shapebench_5_total_drag_constrained; do
  for seed in $(seq 0 9); do
    launch "run_GA_parallel_${REWARD}_seed${seed}_20p_200i" \
      --framework GA_parallel \
      --reward "$REWARD" \
      --n_particles 20 \
      --n_iterations 200
  done
done

wait
echo "PSO constrained node done."
