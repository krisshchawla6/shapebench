#!/bin/bash
#SBATCH --job-name=drvr_stb_v3
#SBATCH --partition=compute
#SBATCH --array=1-10
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/scratch/ShapeEvolve/environments/DrivAer_Star/results/logs_super_tight_bounds/slurm_v3_%A_%a.log

# ShapeEvolve v3 — 10 attempts, vtk_E, super-tight bounds ablation.
# All 17 high-saturation parameters tightened by 50% vs tight_bounds.
# One attempt per dedicated 128-vCPU node to avoid thread contention.

export OMP_NUM_THREADS=128
export MKL_NUM_THREADS=128

source /home/jack/venv_torch210/bin/activate

ATTEMPT=${SLURM_ARRAY_TASK_ID}
REPO=/scratch/ShapeEvolve
OUTDIR=$REPO/environments/DrivAer_Star/results/run_v3_dynamic_optimizer_cd_only_super_tight_bounds_drivaer_star_vtk_E_attempt_${ATTEMPT}_flash_2_5_n10000

BOUNDS_OVERRIDE='{"car_size":[0.9,1.1],"car_len":[-0.05,0.05],"ramp_angle":[-4.0,4.0],"front_bumper_length":[-0.05,0.05],"wind_screen_x":[-0.025,0.025],"wind_screen_z":[-0.025,0.025],"side_mirrors_x":[-0.025,0.025],"side_mirrors_z":[-0.025,0.025],"rear_window_x":[-0.025,0.025],"rear_window_z":[-0.025,0.025],"trunklid_angle":[-4.0,4.0],"trunklid_x":[-0.025,0.025],"trunklid_z":[-0.025,0.025],"diffusor_angle":[-2.0,2.0],"car_front_hood_angle":[-4.0,4.0],"car_air_intake_angle":[-4.0,4.0],"tires_diameter":[-0.0065,0.0065],"tires_width":[-0.0075,0.0075]}'

mkdir -p "$OUTDIR"
echo "Starting attempt ${ATTEMPT} ..."

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
    --bounds_override "$BOUNDS_OVERRIDE" \
    --rho 1.25 \
    --u 40.0 \
    --area_ref 2.37 \
    --output-dir "$OUTDIR"

echo "Attempt ${ATTEMPT} done (exit $?)."
