#!/bin/bash
# Unattended chain for run_07 (ERA5-Land with corrected Sep 2022 - Feb 2024 precipitation).
# 1 wait for the CDS downloads  2 verify + patch the file  3 re-run ERA5-Land  4 compile  5 merge into compiled_v3_run_07_tpfix
# Stops at the first failure and writes STATUS to $HMA_STATUS_DIR/RUN07_STATUS.md.
set -u
ROOT=${HMA_PAPER_ROOT:?set HMA_PAPER_ROOT (working dir with scripts/, analysis/, tables/)}
HMA_STATUS_DIR=${HMA_STATUS_DIR:-$ROOT}
CODE=${HMA_CODE_DIR:?set HMA_CODE_DIR (code/pipeline)}
OGGM=${HMA_OGGM_ROOT:?set HMA_OGGM_ROOT}
PY=${HMA_PYTHON:-python}
STATUS=$HMA_STATUS_DIR/RUN07_STATUS.md
LOG=$OGGM/run07_chain.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG" >> "$STATUS"; }
fail() { say "FAILED: $*"; exit 1; }
echo "# run_07 status (written by scripts/run07_chain.sh)" > "$STATUS"
say "chain started"
# 1 downloads
while pgrep -f "era5land_tp_fix.py download" >/dev/null; do sleep 30; done
n=$(ls $OGGM/download_cache/era5_land_data/raw/fix/*.nc 2>/dev/null | wc -l)
[ "$n" -ge 22 ] || fail "only $n of 22 corrected files downloaded"
say "downloads complete ($n files)"
# 2 verify + patch
cd $ROOT
$PY scripts/era5land_tp_fix.py patch >> "$LOG" 2>&1 || fail "verify/patch (see $LOG)"
tail -n 30 "$LOG" | grep -E "control|affected|VERIFY|wrote" >> "$STATUS"
# 3 re-run ERA5-Land only (same settings as run_06)
cd $CODE
export OGGM_V3_ONLY_FORCING=ERA5-Land OGGM_V3_RUN=run_07_era5l_tpfix OGGM_V3_AREA_CONSISTENT=1 \
       OGGM_V3_AREA_FACTORS=$OGGM/calibration_output_v3/area_factors_run_04_full_fix.csv OGGM_V3_WORKERS=12
say "model re-run started (ERA5-Land, all regions)"
$PY 01_MAIN_PIPELINE_run_this.py >> $OGGM/run_v3_run07.log 2>&1 || fail "model run (see $OGGM/run_v3_run07.log)"
say "model re-run finished"
# 4 compile
$PY 02_COMPILE_RESULTS_run_after.py >> $OGGM/compile_run07.txt 2>&1 || fail "compile"
say "compile finished"
# 5 merge
cd $ROOT
$PY scripts/merge_run07.py >> "$LOG" 2>&1 || fail "merge (asserts; see $LOG)"
tail -n 12 "$LOG" >> "$STATUS"
say "merge finished -> compiled_v3_run_07_tpfix; next: analysis (06_ANALYSIS_v3.py), switch scripts/paths.py RUN, regenerate"
