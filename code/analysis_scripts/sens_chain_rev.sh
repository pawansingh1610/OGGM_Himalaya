#!/bin/bash
# Reverse hindcast on the 681-glacier test subset: calibrate to Hugonnet 2010-2019 only, predict 2000-2009 (new run name run_s07_rev_hindcast).
# Same settings as the other sensitivity runs (sigma_T = 4, no area-consistent target, corrected ERA5-Land precipitation); model-code copy
# OGGM_Complete_sens. CHELSA-W5E5 (ends 2016) cannot be calibrated to 2010-2019 and is not run.
set -u
OGGM=${HMA_OGGM_ROOT:?set HMA_OGGM_ROOT}
CODE=${HMA_SENS_CODE_DIR:?set HMA_SENS_CODE_DIR (code/pipeline/provenance + the other pipeline files)}
PY=${HMA_PYTHON:-python}
ST=${HMA_STATUS_DIR:-.}/SENS_STATUS.md
say() { echo "$(date '+%F %T') $*" | tee -a $OGGM/sens_chain.log >> $ST; }
export OGGM_V3_ONLY_IDS=$(cat $OGGM/subset681_ids.txt) OGGM_V3_WORKERS=${OGGM_V3_WORKERS:-6} OGGM_V3_ONLY_FORCING=ERA5,ERA5-Land,HARv2,CRU_TS4
name=run_s07_rev_hindcast
cd $CODE
if [ -d $OGGM/compiled_v3_$name/per_forcing ]; then say "skip $name (already compiled)"; exit 0; fi
say "start $name (OGGM_V3_CALIB_ONLY_PERIOD=2010-01-01_2020-01-01, forcings $OGGM_V3_ONLY_FORCING)"
env OGGM_V3_CALIB_ONLY_PERIOD=2010-01-01_2020-01-01 OGGM_V3_RUN=$name $PY 01_MAIN_PIPELINE_run_this.py >> $OGGM/run_$name.log 2>&1 || { say "FAILED: $name model"; exit 1; }
env OGGM_V3_RUN=$name $PY 02_COMPILE_RESULTS_run_after.py >> $OGGM/compile_$name.txt 2>&1 || { say "FAILED: $name compile"; exit 1; }
say "done $name"
say "REVERSE HINDCAST FINISHED"
