# HMA glacier reconstruction with five climate forcings: code

Model code and analysis scripts for the manuscript *"Sensitivity of High Mountain Asia glacier mass balance
reconstructions (1950-2025) to different climate forcings"* (Pawan Singh; manuscript for The Cryosphere).

All 95,536 RGI 6.0 glaciers of High Mountain Asia (RGI regions 13, 14, 15) are modelled with five climate products
(ERA5, ERA5-Land, CRU TS4, CHELSA-W5E5, HARv2). OGGM 1.6.3 supplies the preprocessing and the monthly temperature-index
mass-balance model only; volume and area evolve with volume-area scaling anchored to the Farinotti et al. (2019)
consensus volume. Every glacier and forcing is calibrated to the Hugonnet et al. (2021) geodetic mass balance.

**Data** (per-glacier annual output, calibration results, all tables behind the paper, area-factor input) are in a separate
Zenodo data record (CC BY 4.0); see its README for the file formats. This repository holds code only (MIT licence).

## 1. Contents

```
code/
  config.py                   all machine-specific locations (environment variables, section 2)
  pipeline/                   model code: calibration, geometry, compile, analysis; environment files
    provenance/               code variants that produced individual runs (section 3)
  analysis_scripts/           scripts that make every table, figure and number of the paper
    figures/                  one script per figure (make_all.py runs them all)
  manifests/                  run_manifest_*.json: settings, prior, bounds and SHA-256 of the code of each run
  calibration_inputs/         where the area-factor file from the data record goes
legacy/durung_drung/          earlier single-glacier pipeline (Durung Drung), unrelated to the paper; GPL-3.0
```

## 2. Re-running

1. Create the conda environment from `code/pipeline/oggm_environment_minimal.yml` (OGGM 1.6.3; the `full` file is a complete export).
2. Obtain the third-party inputs (section 4) into `$HMA_OGGM_ROOT/download_cache` (not included in this archive).
3. Set the locations, then run the pipeline from `code/pipeline`:
   ```
   export HMA_OGGM_ROOT=/path/to/OGGM            # holds download_cache/, calibration_output_v3/, compiled_v3_<run>/
   export HMA_CODE_DIR=$PWD/code/pipeline         # model code
   export HMA_PYTHON=$(which python)              # for the *.sh chains
   cd code/pipeline && python precheck.py && OGGM_V3_RUN=run_06_areacons OGGM_V3_AREA_CONSISTENT=1 \
     OGGM_V3_AREA_FACTORS=../calibration_inputs/area_factors_run_04_full_fix.csv python main_calibration_parallel_v2.py
   OGGM_V3_RUN=run_06_areacons python compile_v2.py
   ```
   (`01_/02_/03_*` and `main_calibration_parallel_v2.py` / `compile_v2.py` / `precheck.py` are identical pairs; the second names are the ones other modules import.)
   `config.json.example` shows the optional config file; `HMA_DOWNLOAD_CACHE` overrides the cache location.
4. The analysis scripts expect the layout `scripts/` (= `code/analysis_scripts`), `analysis/` (= `data/analysis_csv` of the Zenodo data record), `tables/`
   (= `data/table_and_figure_csv` + `gridded_maps` of the data record) in one working folder; `scripts/paths.py` selects the run. Set `HMA_OGGM_ROOT`.
   `*_chain.sh` additionally need `HMA_PAPER_ROOT`, `HMA_CODE_DIR` or `HMA_SENS_CODE_DIR`, and optionally `HMA_STATUS_DIR`.

All absolute paths of the author's machine were replaced by these variables (`code/config.py`). The edits are **path-only**
(candidate cache-path lists replaced by environment variables); no model logic was changed. The published versions of the
following files differ from the files that ran only in file paths: `pipeline/climate_loader_unified.py`, `pipeline/precheck.py`,
`pipeline/03_PRECHECK_run_first.py`, `pipeline/provenance/climate_loader_unified_run06.py` (plus the analysis/shell scripts).
SHA-256 of the **original** (unedited) versions, as recorded in `manifests/`:

| original file | SHA-256 |
|---|---|
| `03_PRECHECK_run_first.py` | `2ffce753e2996a32799e5a6ee4621522f41dce82e8adc4aefc7aba3e1d338201` |
| `climate_loader_unified.py` | `902fc7a3c8187c3aa8d74e28833e428c150e64028427cf41657b5c4dd9dba015` |
| `climate_loader_unified_run06.py` | `b0baa853dd5a0e4d9bd6c30858991f634279bf066a40ca559d702e43da592b1c` |
| `precheck.py` | `2ffce753e2996a32799e5a6ee4621522f41dce82e8adc4aefc7aba3e1d338201` |

(`climate_loader_unified.py` = run_07 loader; `climate_loader_unified_run06.py` = run_06 loader; `precheck.py` and `03_PRECHECK_run_first.py` are identical.)
`01_MAIN_PIPELINE_run_this.py` and `calibration_v3.py` are released unchanged.

## 3. Runs and provenance

| run | content |
|---|---|
| `run_06_areacons` | full run, area-consistent targets, sigma_T = 4 everywhere (superseded only for ERA5-Land 2022-2024) |
| `run_07_era5l_tpfix` | ERA5-Land re-run with corrected precipitation |
| `run_s01_base` ... `run_s07_rev_hindcast` | 681-glacier test subset (ids in `manifests/run_manifest_run_s*.json`, key `only_ids`), five forcings (s07: four, no CHELSA): s01 baseline; s02 calibrated 2000-2009 only (hindcast of 2010-2019); s03/s04 melt factor +20 %/-20 %; s05/s06 lapse rate +0.001/-0.001 K m-1; s07 calibrated 2010-2019 only (reverse hindcast) |

The sensitivity runs were made with a copy of the pipeline that adds the switches `OGGM_V3_MELT_SCALE`, `OGGM_V3_LAPSE_DELTA` and
`OGGM_V3_CALIB_ONLY_PERIOD`: `code/pipeline/provenance/01_MAIN_PIPELINE_sensitivity_runs_s01-s07.py` (use it in place of
`01_MAIN_PIPELINE_run_this.py` together with the other pipeline files). `provenance/climate_loader_unified_run06.py` is the climate loader of run_06
(before the ERA5-Land fix). `provenance/*.diff` are the patches introduced for run_05 and run_06.

## 4. Third-party data

The model needs inputs that are not redistributed here: ERA5 and ERA5-Land (Copernicus Climate Change Service), CRU TS4
(Harris et al. 2020), CHELSA-W5E5 (Karger et al. 2023), HARv2 (Wang et al. 2021), RGI 6.0, the Farinotti et al. (2019)
consensus ice thickness, the Hugonnet et al. (2021) geodetic mass balance, SRTM, the WGMS Fluctuations of Glaciers
database (version 2026-02-10), GlaMBIE, GHS-POP R2023A and HydroBASINS. `code/analysis_scripts/download_external.sh`
documents the sources of the population and basin inputs.

## 5. Licence and citation

Code: MIT (`LICENSE`), except `legacy/`, which keeps its own GPL-3.0 licence. Please cite the archived version of this
repository (see `CITATION.cff`), the Zenodo data record and the manuscript.
