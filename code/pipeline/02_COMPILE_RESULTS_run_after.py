"""
02_COMPILE_RESULTS_run_after.py (== compile_v2.py) — v3 compile
Compiles per-glacier outputs of 01_MAIN_PIPELINE_run_this.py (v3)

    compiled_v3/
        per_forcing/<F>/yearly.parquet      all glacier-years, one forcing
        per_forcing/<F>/metadata.csv        one row per glacier (ALL json keys)
        metadata_all.csv                    all forcings stacked
        failures_all.csv                    glaciers that failed, with reason
        compile_report.txt

Parquet instead of a 28M-row CSV: the analysis script reads only the
columns it needs, so every number comes from the FULL dataset (the v2
notebook read nrows=500000 slices, i.e. the first ~1,400 glaciers).

Usage:  python 02_COMPILE_RESULTS_run_after.py
"""
import json
import os
import sys
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from climate_loader_unified import detect_cache_root   # noqa: E402

CACHE_ROOT   = detect_cache_root()


def _output_root():
    cf = SCRIPT_DIR / 'config.json'
    if cf.exists():
        p = json.loads(cf.read_text()).get('output_root', '')
        if p and 'v2' not in Path(p).name and Path(p).parent.exists():
            return Path(p)
    return CACHE_ROOT.parent / 'calibration_output_v3'


RUN_SUFFIX   = os.environ.get('OGGM_V3_RUN', 'run_03')
OUTPUT_ROOT  = _output_root()
COMPILE_ROOT = OUTPUT_ROOT.parent / ('compiled_v3' if RUN_SUFFIX == 'run_03'
                                     else f'compiled_v3_{RUN_SUFFIX}')
FORCINGS   = ['ERA5', 'ERA5-Land', 'CRU_TS4', 'CHELSA', 'HARv2']
REGIONS    = ['RGI60-13', 'RGI60-14', 'RGI60-15']
N_WORKERS  = max(1, min(32, int(cpu_count() * 0.3)))

YEARLY_COLS = ['rgi_id', 'year', 'mb_annual_mwe', 'dm_gt', 'area_start_km2',
               'volume_km3', 'volume_km3_low', 'volume_km3_high',
               'area_km2', 'area_km2_low', 'area_km2_high',
               'mean_thickness_m', 'length_km', 'is_valid', 'reconstructed',
               'extinct']


def _flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f'{prefix}{k}_'))
        elif isinstance(v, list):
            out[f'{prefix}{k}'] = json.dumps(v)
        else:
            out[f'{prefix}{k}'] = v
    return out


def _read(args):
    gl_dir, forcing, region = args
    gl_dir = Path(gl_dir)
    try:
        g = pd.read_csv(gl_dir / 'glacier_geometry.csv')
        g['rgi_id'] = gl_dir.name
        g = g[[c for c in YEARLY_COLS if c in g.columns]]
        m = json.loads((gl_dir / 'diagnostics' / 'calibration_metadata.json')
                       .read_text())
        if m.get('pipeline_version') != 'v3':
            return None, None
        return g, _flatten(m)
    except Exception:
        return None, None


def compile_forcing(forcing):
    args = []
    for reg in REGIONS:
        rd = OUTPUT_ROOT / forcing / f'{reg}_{RUN_SUFFIX}'
        if not rd.exists():
            continue
        args += [(str(d), forcing, reg) for d in rd.iterdir()
                 if d.is_dir() and d.name.startswith('RGI')
                 and (d / 'glacier_geometry.csv').exists()]
    print(f'  [{forcing}] {len(args):,} glacier folders')
    ys, ms = [], []
    with Pool(N_WORKERS) as pool:
        for g, m in pool.imap_unordered(_read, args, chunksize=100):
            if g is not None:
                ys.append(g)
                ms.append(m)
    if not ys:
        return None, None
    y = pd.concat(ys, ignore_index=True)
    for c in ['is_valid', 'reconstructed', 'extinct']:
        if c in y:
            y[c] = y[c].astype(bool)
    y['year'] = y['year'].astype('int16')
    y = y.sort_values(['rgi_id', 'year']).reset_index(drop=True)
    meta = pd.DataFrame(ms).sort_values('rgi_id').reset_index(drop=True)
    return y, meta


def main():
    t0 = datetime.now()
    COMPILE_ROOT.mkdir(parents=True, exist_ok=True)
    report, metas, fails = [], [], []
    for f in FORCINGS:
        y, m = compile_forcing(f)
        if y is None:
            report.append(f'{f}: no data')
            continue
        od = COMPILE_ROOT / 'per_forcing' / f
        od.mkdir(parents=True, exist_ok=True)
        y.to_parquet(od / 'yearly.parquet', index=False)
        m.to_csv(od / 'metadata.csv', index=False)
        metas.append(m)
        st = m['calib_status'].value_counts().to_dict()
        line = (f'{f}: {m.rgi_id.nunique():,} glaciers, {len(y):,} rows, '
                f'{int(y.year.min())}-{int(y.year.max())}, calib {st}, '
                f'median dT {m.temp_bias.median():+.2f}, median pf '
                f'{m.prcp_fac.median():.2f}, median elev offset '
                f'{m.elev_offset_m.median():.0f} m, flowlines '
                f'{m.get("flowline_type", pd.Series(dtype=str)).value_counts().to_dict()}')
        print('  ' + line)
        report.append(line)
        for reg in REGIONS:
            fp = OUTPUT_ROOT / f / f'{reg}_{RUN_SUFFIX}' / 'failures.csv'
            if fp.exists():
                fd = pd.read_csv(fp, header=None, names=['rgi_id', 'error'])
                fd['forcing'], fd['region'] = f, reg
                fails.append(fd)
    if metas:
        pd.concat(metas, ignore_index=True).to_csv(
            COMPILE_ROOT / 'metadata_all.csv', index=False)
    if fails:
        fa = pd.concat(fails, ignore_index=True)
        fa.to_csv(COMPILE_ROOT / 'failures_all.csv', index=False)
        report.append(f'failures (raw log lines, may include retried): {len(fa):,}')
    report.append(f'compiled {t0:%Y-%m-%d %H:%M} in '
                  f'{(datetime.now() - t0).total_seconds() / 60:.1f} min')
    (COMPILE_ROOT / 'compile_report.txt').write_text('\n'.join(report))
    print('\n'.join(report[-2:]))


if __name__ == '__main__':
    main()
