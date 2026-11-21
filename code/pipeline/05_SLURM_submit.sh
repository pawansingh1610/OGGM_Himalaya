#!/bin/bash
#SBATCH --job-name=glacier_v3
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=72:00:00
#SBATCH --output=glacier_v3_%j.out
#SBATCH --error=glacier_v3_%j.err
#SBATCH --mail-type=END,FAIL
# (file names kept: 05_SLURM_submit.sh == submit_calibration_v2.sh)
#
# Test subset first:   sbatch --export=ALL,OGGM_V3_TEST=200 submit_calibration_v2.sh
# Full run:            sbatch submit_calibration_v2.sh

echo "========================================"
echo "  GLACIER CALIBRATION v3 — SLURM JOB"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $SLURMD_NODENAME"
echo "  Tasks    : $SLURM_NTASKS"
echo "  Test mode: ${OGGM_V3_TEST:-off}"
echo "  Start    : $(date)"
echo "========================================"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate oggm

# SLURM copies the script to a spool dir, so $0 is not the code dir;
# use the submit directory instead.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
echo "  Working dir: $(pwd)"

echo "Running precheck..."
python precheck.py
if [ $? -ne 0 ]; then
    echo "PRECHECK FAILED — aborting job."
    exit 1
fi

echo "Precheck passed — starting calibration..."
python main_calibration_parallel_v2.py
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Calibration finished — compiling..."
    python compile_v2.py
fi

echo "========================================"
echo "  Finished : $(date)"
echo "  Exit code: $EXIT_CODE"
echo "========================================"
exit $EXIT_CODE
