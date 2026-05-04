#!/bin/bash
#SBATCH --job-name=bn_bo_tdcon
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/scratch/ShapeEvolve/environments/BlendedNet/results/logs_total_drag_constrained_runs/slurm_bo_%j.log

# BO_torch — 20 runs (10 seeds × 2 constrained total-drag rewards)
# AR ≥ 2.5 constraint blocks surrogate-exploitation corner (B3=200/C2=785).
# 128-vCPU node: 16 concurrent × 8 threads = 128
MAX_JOBS=16
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
source /scratch/ShapeEvolve/slurm_jobs/total_drag_constrained/_common.sh

for REWARD in shapebench_5_total_drag_constrained; do
  for seed in $(seq 0 9); do
    launch "run_BO_torch_${REWARD}_seed${seed}_n1000" \
      --framework BO_torch \
      --reward "$REWARD" \
      --n_calls 1000 \
      --n_initial 30 \
      --random_state "$seed" \
      --gradient_infeasible
  done
done

wait
echo "BO constrained node done."
