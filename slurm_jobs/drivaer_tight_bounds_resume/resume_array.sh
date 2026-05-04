#!/bin/bash
#SBATCH --job-name=drvr_tb_resume
#SBATCH --partition=compute
#SBATCH --array=1-10
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=compute-dy-hpc-128vcpu-[3-7]
#SBATCH --output=/scratch/ShapeEvolve/environments/DrivAer_Star/results/logs_tight_bounds_resume/slurm_%A_%a.log

# Resume DrivAer tight_bounds v3 runs, one attempt per node.
# Excludes nodes 3-7 (occupied by BlendedNet jobs 9445-9449).
# Each node gets 128 dedicated vCPUs; single process per node.

export OMP_NUM_THREADS=128
export MKL_NUM_THREADS=128

source /home/jack/venv_torch210/bin/activate

ATTEMPT=${SLURM_ARRAY_TASK_ID}
REPO=/scratch/ShapeEvolve
OUTDIR=$REPO/environments/DrivAer_Star/results/run_v3_dynamic_optimizer_cd_only_tight_bounds_drivaer_star_vtk_E_attempt_${ATTEMPT}_flash_2_5_n10000

echo "Resuming attempt ${ATTEMPT} from $(python3 -c "import json; print(json.load(open('$OUTDIR/checkpoint.json'))['last_completed_iter'])" 2>/dev/null || echo 'unknown') ..."

python "$REPO/run_benchmark.py" \
    --framework v3_dynamic_optimizer \
    --environment DrivAer_Star \
    --reward cd_only \
    --iterations 1000 \
    --batch_size 10 \
    --sampler_model gemini-2.5-flash \
    --inspirations 10 \
    --action gaussain \
    --sampler_max_retries 3 \
    --pw_alpha 3.0 \
    --base_vtk "$REPO/environments/DrivAer_Star/data/vtk_E/00000.vtk" \
    --bounds_override '{"car_size": [0.9, 1.1], "diffusor_angle": [-4.0, 4.0]}' \
    --rho 1.25 \
    --u 40.0 \
    --area_ref 2.37 \
    --resume \
    --output-dir "$OUTDIR"

echo "Attempt ${ATTEMPT} done (exit $?)."
