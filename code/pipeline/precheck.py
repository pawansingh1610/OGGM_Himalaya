"""
================================================================================
PRECHECK — run this before anything else on any machine   (v3)
(file names kept: 03_PRECHECK_run_first.py == precheck.py)
================================================================================
What it does:
  1. Finds or asks for the OGGM data root, writes/updates config.json
     (v3: output_root -> calibration_output_v3; the old v2 path is kept as
      'output_root_v2' so v2 checkpoints are never reused by mistake)
  2. Python environment + OGGM version
  3. Required code files
  4. Climate data files for all 5 forcings
  5. NEW: grid-cell OROGRAPHY files for all 5 forcings + the elevation
     offset v2 ignored (glacier Zmed − grid-cell elevation)
  6. RGI shapefiles, field MB data
  7. Climate loader functional test (now also gap check + grid elevation)
  8. Disk space, OGGM init
  Exit code 0 only if everything needed for v3 is present.

Run with:
    conda activate oggm
    python precheck.py
================================================================================
"""

import sys
import json
import os
import time
import importlib
from pathlib import Path

GREEN, YELLOW, RED, BOLD, RESET = '\033[92m', '\033[93m', '\033[91m', '\033[1m', '\033[0m'
if not sys.stdout.isatty():
    GREEN = YELLOW = RED = BOLD = RESET = ''


def ok(msg):   print(f"  {GREEN}✓ PASS{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠ WARN{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗ FAIL{RESET}  {msg}")
def info(msg): print(f"  {BOLD}  INFO{RESET}  {msg}")
def hdr(msg):  print(f"\n{BOLD}{msg}{RESET}")


SCRIPT_DIR  = Path(__file__).resolve().parent
CONFIG_FILE = SCRIPT_DIR / 'config.json'
FORCINGS = ['ERA5', 'ERA5-Land', 'CRU_TS4', 'CHELSA', 'HARv2']


# ─────────────────────────────────────────────────────────────────────────────
# 1. config.json
# ─────────────────────────────────────────────────────────────────────────────

def find_or_create_config() -> dict:
    if CONFIG_FILE.exists():
        cfg = json.loads(CONFIG_FILE.read_text())
        info(f"config.json found: {CONFIG_FILE}")
        # Moved to another machine? Repair stale paths automatically.
        if not Path(cfg.get('data_root', '')).exists():
            sys.path.insert(0, str(SCRIPT_DIR))
            from climate_loader_unified import _CANDIDATE_ROOTS, _CACHE_MARKER
            here = SCRIPT_DIR
            cands = list(_CANDIDATE_ROOTS) + [
                p / sub for p in [here] + list(here.parents)
                for sub in ['download_cache', 'OGGM/download_cache']]
            found = next((c for c in cands
                          if c.exists() and (c / _CACHE_MARKER).exists()), None)
            if found is None:
                fail(f"data_root {cfg.get('data_root')} does not exist on this "
                     "machine and no download_cache was found — edit config.json")
                sys.exit(1)
            warn(f"data_root {cfg.get('data_root')} not found on this machine "
                 f"→ using {found}")
            cfg['data_root'] = str(found)
            cfg['output_root'] = str(found.parent / 'calibration_output_v3')
            cfg['output_root_v2'] = str(found.parent / 'calibration_output_v2')
            cfg['updated_paths'] = time.strftime('%Y-%m-%d %H:%M:%S')
            CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
        if not Path(cfg.get('output_root', '/nonexistent')).parent.exists():
            cfg['output_root'] = str(Path(cfg['data_root']).parent /
                                     'calibration_output_v3')
            CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
            warn(f"output_root parent missing → {cfg['output_root']}")
        out = Path(cfg.get('output_root', ''))
        if 'v2' in out.name or not cfg.get('output_root'):
            new = Path(cfg['data_root']).parent / 'calibration_output_v3'
            cfg['output_root_v2'] = str(out)
            cfg['output_root'] = str(new)
            cfg['updated_v3'] = time.strftime('%Y-%m-%d %H:%M:%S')
            CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
            warn(f"output_root switched to v3 folder: {new}")
            info(f"  (old v2 results stay in {out}; key 'output_root_v2')")
        info(f"  data_root   = {cfg.get('data_root')}")
        info(f"  output_root = {cfg.get('output_root')}")
        return cfg

    print(f"\n{BOLD}config.json not found — let's create it.{RESET}")
    _root = Path(os.environ.get("HMA_OGGM_ROOT", Path.home() / "OGGM"))
    candidates = [
        Path(os.environ["HMA_DOWNLOAD_CACHE"]) if os.environ.get("HMA_DOWNLOAD_CACHE") else _root / "download_cache",
        Path("/scratch/OGGM/download_cache"),
    ]
    auto = next((c for c in candidates
                 if c.exists() and (c / 'chelsa_w5e5').exists()), None)
    data_root = None
    if auto:
        print(f"Auto-detected data root: {auto}")
        if input("Use this? [Y/n]: ").strip().lower() in ('', 'y', 'yes'):
            data_root = auto
    while data_root is None:
        p = Path(input("Enter full path to OGGM download_cache directory: ")
                 .strip()).expanduser()
        if p.exists():
            data_root = p
        else:
            print(f"  Path not found: {p}")
    output_root = data_root.parent / 'calibration_output_v3'
    print(f"\nOutput root: {output_root}")
    if input("Use this? [Y/n]: ").strip().lower() not in ('', 'y', 'yes'):
        output_root = Path(input("Custom output root: ").strip()).expanduser()
    cfg = {'data_root': str(data_root), 'output_root': str(output_root),
           'created': time.strftime('%Y-%m-%d %H:%M:%S')}
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    ok(f"config.json written: {CONFIG_FILE}")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Registries
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_FILES = [
    ('ERA5 temperature',   ['ERA5/era5_t2m.nc'], 'ERA5', 100),
    ('ERA5 precipitation', ['ERA5/era5_tp.nc'],  'ERA5', 100),
    ('ERA5-Land temperature',
     ['era5_land_data/raw/era5_land_t2m_1950_2025.nc',
      'era5_land_data/raw/era5_land_t2m_1950_2026.nc'], 'ERA5-Land', 500),
    ('ERA5-Land precipitation',
     ['era5_land_data/raw/era5_land_tp_1950_2025.nc',
      'era5_land_data/raw/era5_land_tp_1950_2026.nc'], 'ERA5-Land', 500),
    ('CRU TS4 temperature',
     ['cru_ts4/cru_ts4.04.1901.2019.tmp.dat.nc',
      'cru_ts4/cru_ts4.08.1901.2023.tmp.dat.nc'], 'CRU_TS4', 100),
    ('CRU TS4 precipitation',
     ['cru_ts4/cru_ts4.04.1901.2019.pre.dat.nc',
      'cru_ts4/cru_ts4.08.1901.2023.pre.dat.nc'], 'CRU_TS4', 100),
    ('CHELSA temperature',
     ['chelsa_w5e5/chelsa-w5e5_obsclim_tas_monthlymean_1979-2016_HMA.nc',
      'chelsa_w5e5/chelsa_w5e5_tas_1979_2016.nc'], 'CHELSA', 1000),
    ('CHELSA precipitation',
     ['chelsa_w5e5/chelsa-w5e5_obsclim_pr_monthlytotal_HMA.nc',
      'chelsa_w5e5/chelsa-w5e5_obsclim_pr_monthlymean_1979-2016_HMA.nc'],
     'CHELSA', 500),
    ('HARv2 temperature',   ['HRAv2/HARv2_d10km_m_2d_t2_1980_2025.nc'], 'HARv2', 50),
    ('HARv2 precipitation', ['HRAv2/HARv2_d10km_m_2d_prcp_1980_2025.nc'], 'HARv2', 50),
]

RGI_FILES = {
    'RGI60-13': ['RGI/rgi60/13_rgi60_CentralAsia/13_rgi60_CentralAsia.shp',
                 'RGI/13_rgi60_CentralAsia/13_rgi60_CentralAsia.shp'],
    'RGI60-14': ['RGI/rgi60/14_rgi60_SouthAsiaWest/14_rgi60_SouthAsiaWest.shp',
                 'RGI/14_rgi60_SouthAsiaWest/14_rgi60_SouthAsiaWest.shp'],
    'RGI60-15': ['RGI/rgi60/15_rgi60_SouthAsiaEast/15_rgi60_SouthAsiaEast.shp',
                 'RGI/15_rgi60_SouthAsiaEast/15_rgi60_SouthAsiaEast.shp'],
}

REQUIRED_PYTHON_FILES = [
    'climate_loader_unified.py',
    'calibration_v3.py',
    'main_calibration_parallel_v2.py',
]
OPTIONAL_PYTHON_FILES = [
    'compile_v2.py', '06_ANALYSIS_v3.py', 'submit_calibration_v2.sh',
]


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────

def check_python_imports() -> int:
    hdr("1. Python environment")
    failures = 0
    # skimage / salem / rasterio are OPTIONAL imports inside OGGM: if missing,
    # OGGM imports fine but e.g. glacier_masks fails ("name 'skdraw' is not
    # defined") for every glacier; without tables (pytables) the Hugonnet
    # table cannot be read.
    for name in ['numpy', 'pandas', 'xarray', 'geopandas', 'scipy',
                 'matplotlib', 'oggm', 'netCDF4', 'shapely',
                 'skimage', 'salem', 'rasterio', 'tables']:
        try:
            m = importlib.import_module(name)
            ok(f"{name} {getattr(m, '__version__', '?')}")
        except ImportError:
            fail(f"{name} not installed")
            failures += 1
    try:
        import oggm
        major_minor = tuple(int(x) for x in oggm.__version__.split('.')[:2])
        if major_minor < (1, 6):
            fail(f"OGGM {oggm.__version__} < 1.6 — v3 needs 1.6 "
                 "(calendar-year MB, temp_bias/prcp_fac setters)")
            failures += 1
    except Exception:
        pass
    for name in ['pyarrow']:
        try:
            importlib.import_module(name)
            ok(f"{name} (parquet for compile/analysis)")
        except ImportError:
            warn(f"{name} missing → pip install pyarrow (needed by compile)")
    try:
        import pyproj
        ok(f"pyproj {pyproj.__version__} (PROJ: {pyproj.proj_version_str})")
    except Exception as e:
        warn(f"pyproj issue: {e}")
    return failures


def check_python_files() -> int:
    hdr("2. Code files")
    failures = 0
    for f in REQUIRED_PYTHON_FILES:
        if (SCRIPT_DIR / f).exists():
            ok(f)
        else:
            fail(f"{f} NOT FOUND in {SCRIPT_DIR}")
            failures += 1
    for f in OPTIONAL_PYTHON_FILES:
        (ok if (SCRIPT_DIR / f).exists() else warn)(f"{f} (optional)")
    return failures


def check_climate_files(data_root: Path) -> tuple:
    hdr("3. Climate data files")
    failures, warnings_n = 0, 0
    status = {f: True for f in FORCINGS}
    for label, cands, forcing, min_mb in REQUIRED_FILES:
        fp = next((data_root / r for r in cands if (data_root / r).exists()), None)
        if fp is None:
            fail(f"{label} [{forcing}]  expected e.g. {data_root / cands[0]}")
            status[forcing] = False
            failures += 1
        else:
            mb = fp.stat().st_size / 1e6
            if mb < min_mb * 0.1:
                warn(f"{label}: {fp.name} ({mb:.0f} MB — suspiciously small)")
                warnings_n += 1
            else:
                ok(f"{label}: {fp.name} ({mb:.0f} MB)")
    return failures, warnings_n, status


def check_orography(data_root: Path, status: dict) -> int:
    """NEW: grid-cell elevation for every forcing (required by v3)."""
    hdr("4. Grid-cell orography (ref_hgt)  — required by v3")
    sys.path.insert(0, str(SCRIPT_DIR))
    from climate_loader_unified import find_orography_file, OROG_PATHS
    failures = 0
    for f in FORCINGS:
        try:
            p = find_orography_file(f, data_root)
            ok(f"{f:10s}: {p}")
        except FileNotFoundError:
            fail(f"{f:10s}: no orography file. Add its path to OROG_PATHS in "
                 f"climate_loader_unified.py; looked for e.g. "
                 f"{data_root / OROG_PATHS[f][0]}")
            status[f] = False
            failures += 1
    if failures:
        info("Fix:  python 07_GET_OROGRAPHY.py inspect   then   "
             "python 07_GET_OROGRAPHY.py all")
        info("Needed (same grid as the climate file):")
        info("  ERA5 / ERA5-Land : invariant 'geopotential' (CDS single-level / "
             "ERA5-Land invariant)")
        info("  CRU TS4          : CRU CL2.0 10' elevation averaged to 0.5° "
             "(07_GET_OROGRAPHY.py cru does this)")
        info("  CHELSA-W5E5      : 'orog' field (ISIMIP3a CHELSA-W5E5 orog)")
        info("  HARv2            : static file, variable 'hgt'")
    return failures


def elevation_offset_report(data_root: Path, status: dict, n: int = 60) -> None:
    """Glacier Zmed − grid-cell elevation: the height error v2 ignored."""
    hdr("5. Elevation offset v2 ignored (glacier Zmed − grid cell)")
    try:
        import numpy as np
        import pandas as pd
        import geopandas as gpd
        from climate_loader_unified import grid_elevation
        parts = []
        for reg, cands in RGI_FILES.items():
            fp = next((data_root / r for r in cands if (data_root / r).exists()), None)
            if fp is not None:
                g = gpd.read_file(fp, ignore_geometry=True)
                parts.append(g.sample(min(n, len(g)), random_state=1))
        s = pd.concat(parts)
        for f in FORCINGS:
            if not status.get(f):
                continue
            offs = []
            for _, r in s.iterrows():
                try:
                    offs.append(r.Zmed - grid_elevation(f, r.CenLon, r.CenLat,
                                                        data_root)['grid_elev_m'])
                except Exception:
                    pass
            if offs:
                o = np.array(offs)
                info(f"{f:10s}: median {np.median(o):6.0f} m  "
                     f"(10–90 %: {np.percentile(o, 10):5.0f}…{np.percentile(o, 90):5.0f})"
                     f"  ≈ {np.median(o) * 0.0060:+.1f} °C absorbed by temp_bias in v2")
                if np.median(o) < -200 or np.median(o) > 4000:
                    warn(f"{f}: offset looks implausible — check orography units")
    except Exception as e:
        warn(f"offset report skipped: {e}")


def check_rgi_files(data_root: Path) -> int:
    hdr("6. RGI shapefiles")
    failures = 0
    for region, cands in RGI_FILES.items():
        fp = next((data_root / r for r in cands if (data_root / r).exists()), None)
        if fp:
            ok(f"{region}: {fp.name}")
        else:
            fail(f"{region}: shapefile not found")
            failures += 1
    return failures


def check_field_data(data_root: Path) -> None:
    hdr("7. Field mass balance data (validation only)")
    for p in [data_root / 'Field_MB/Mass_Balance.csv',
              data_root / 'field_data/Mass_Balance.csv',
              SCRIPT_DIR / 'Mass_Balance.csv']:
        if p.exists():
            try:
                import pandas as pd
                df = pd.read_csv(p)
                ok(f"{p.name}: {len(df)} rows, cols {list(df.columns[:6])}")
            except Exception as e:
                warn(f"Field data unreadable: {e}")
            return
    warn("No field MB data — validation section of 06_ANALYSIS will be skipped")


def test_climate_loading(data_root: Path, status: dict) -> int:
    hdr("8. Climate loader functional test (Chhota Shigri area)")
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from climate_loader_unified import load_climate, grid_elevation
    except ImportError as e:
        fail(f"Cannot import climate_loader_unified: {e}")
        return 1
    LON, LAT, ZMED = 77.5, 32.2, 4900.0
    failures = 0
    for f in FORCINGS:
        if not status.get(f):
            warn(f"{f}: skipped (files missing)")
            continue
        try:
            t0 = time.time()
            df, meta = load_climate(f, LON, LAT, cache_root=data_root)
            ze = grid_elevation(f, LON, LAT, data_root)['grid_elev_m']
            t = df['temp_C'].mean()
            ok(f"{f:10s}: {len(df):4d} months {df.time.dt.year.min()}-"
               f"{df.time.dt.year.max()}  filled={meta.get('n_filled_months', 0)}  "
               f"grid {ze:5.0f} m  T@grid {t:+5.1f}°C → T@{ZMED:.0f} m "
               f"{t - 0.006 * (ZMED - ze):+5.1f}°C  P {df.prcp_mm.mean() * 12:5.0f} mm/yr "
               f"({time.time() - t0:.1f}s)")
        except Exception as e:
            fail(f"{f}: {str(e)[:120]}")
            status[f] = False
            failures += 1
    return failures


def check_disk_space(output_root: Path) -> None:
    hdr("9. Disk space")
    try:
        import shutil
        base = output_root if output_root.exists() else output_root.parent
        total, used, free = shutil.disk_usage(base)
        est_gb = 20.0   # 95k glaciers × 5 forcings × ~40 kB
        info(f"Free {free / 1e9:.0f} GB of {total / 1e9:.0f} GB; "
             f"estimated v3 output ~{est_gb:.0f} GB")
        (ok if free / 1e9 > 2 * est_gb else warn)("disk space")
    except Exception as e:
        warn(f"Could not check disk space: {e}")


def check_oggm_config(data_root: Path) -> int:
    hdr("10. OGGM initialisation")
    try:
        from oggm import cfg
        cfg.initialize(logging_level='CRITICAL')
        cfg.PATHS['dl_cache_dir'] = str(data_root)
        ok(f"OGGM {__import__('oggm').__version__} initialised")
        return 0
    except Exception as e:
        fail(f"OGGM init failed: {e}")
        return 1


def main():
    print(f"\n{'=' * 65}\n{BOLD}  GLACIER PIPELINE PRECHECK  (v3){RESET}")
    print(f"  Script directory : {SCRIPT_DIR}\n{'=' * 65}")
    cfg = find_or_create_config()
    data_root, output_root = Path(cfg['data_root']), Path(cfg['output_root'])

    nf, nw = 0, 0
    nf += check_python_imports()
    nf += check_python_files()
    f, w, status = check_climate_files(data_root)
    nf += f
    nw += w
    nf += check_orography(data_root, status)
    elevation_offset_report(data_root, status)
    nf += check_rgi_files(data_root)
    check_field_data(data_root)
    nf += test_climate_loading(data_root, status)
    check_disk_space(output_root)
    nf += check_oggm_config(data_root)

    ready = [k for k, v in status.items() if v]
    print(f"\n{'=' * 65}\n{BOLD}  PRECHECK SUMMARY{RESET}\n{'=' * 65}")
    print(f"  Failures : {nf}\n  Warnings : {nw}\n  Forcings ready: {ready}")
    if nf == 0:
        print(f"\n  {GREEN}{BOLD}ALL CHECKS PASSED{RESET}")
        print("  Next:  OGGM_V3_TEST=200 python main_calibration_parallel_v2.py"
              "   (test subset first)")
        print("         python main_calibration_parallel_v2.py   (full run)")
    else:
        print(f"\n  {RED}{BOLD}FAILED — fix the issues above before running.{RESET}")
    print(f"\n  data_root  : {data_root}\n  output_root: {output_root}\n")
    return 0 if nf == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
