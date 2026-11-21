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
# Parallel readers: env OGGM_V3_COMPILE_WORKERS, default = all cores but 2
# (v2 default was 30 % of cores, meant for the 172-core server)
N_WORKERS  = int(os.environ.get('OGGM_V3_COMPILE_WORKERS',
                                max(2, min(32, cpu_count() - 2))))
BATCH      = 400            # glaciers per worker task (one DataFrame returned)
FORCE      = os.environ.get('OGGM_V3_COMPILE_FORCE', '0') == '1'

YEARLY_COLS = ['rgi_id', 'year', 'mb_annual_mwe', 'dm_gt', 'area_start_km2',
               'volume_km3', 'volume_km3_low', 'volume_km3_high',
               'area_km2', 'area_km2_low', 'area_km2_high',
               'mean_thickness_m', 'length_km', 'is_valid', 'reconstructed',
               'extinct']
_CSV_COLS = set(YEARLY_COLS) - {'rgi_id'}
_BOOL_COLS = ['is_valid', 'reconstructed', 'extinct']


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


def _read_batch(dirs):
    """Read a batch of glacier folders -> (one DataFrame, list of meta dicts, n_bad)."""
    parts, metas, bad = [], [], 0
    for gl_dir in dirs:
        gl_dir = Path(gl_dir)
        try:
            m = json.loads((gl_dir / 'diagnostics' / 'calibration_metadata.json')
                           .read_text())
            if m.get('pipeline_version') != 'v3':
                bad += 1
                continue
            g = pd.read_csv(gl_dir / 'glacier_geometry.csv',
                            usecols=lambda c: c in _CSV_COLS)
            g.insert(0, 'rgi_id', gl_dir.name)
            parts.append(g)
            metas.append(_flatten(m))
        except Exception:
            bad += 1
    if not parts:
        return None, metas, bad
    y = pd.concat(parts, ignore_index=True)
    for c in y.columns:
        if c in _BOOL_COLS:
            y[c] = y[c].astype(bool)
        elif c == 'year':
            y[c] = y[c].astype('int16')
        elif c != 'rgi_id':
            y[c] = y[c].astype('float32')     # halves memory vs float64
    return y, metas, bad


class _Progress:
    """tqdm if installed, else a plain line every 5 %."""
    def __init__(self, total, desc):
        self.total, self.desc, self.n, self.last = total, desc, 0, -1
        try:
            from tqdm import tqdm
            self.bar = tqdm(total=total, desc=desc, unit='gl', ncols=90,
                            smoothing=0.05, mininterval=2, file=sys.stdout)
        except ImportError:
            self.bar = None
        self.t0 = datetime.now()

    def update(self, k):
        self.n += k
        if self.bar is not None:
            self.bar.update(k)
            return
        pct = int(self.n / max(self.total, 1) * 20)
        if pct != self.last:
            self.last = pct
            el = (datetime.now() - self.t0).total_seconds()
            rate = self.n / el if el > 0 else 0
            eta = (self.total - self.n) / rate / 60 if rate > 0 else 0
            print(f'    {self.desc}: {self.n:,}/{self.total:,} '
                  f'({pct * 5}%)  {rate:.0f} gl/s  ETA {eta:.1f} min', flush=True)

    def close(self):
        if self.bar is not None:
            self.bar.close()


def _already_compiled(forcing) -> bool:
    """Parquet exists and is newer than every checkpoint of this forcing."""
    if FORCE:
        return False
    pq = COMPILE_ROOT / 'per_forcing' / forcing / 'yearly.parquet'
    md = COMPILE_ROOT / 'per_forcing' / forcing / 'metadata.csv'
    if not (pq.exists() and md.exists()):
        return False
    cks = [OUTPUT_ROOT / forcing / f'{r}_{RUN_SUFFIX}' / f'checkpoint_{r}.json'
           for r in REGIONS]
    cks = [c for c in cks if c.exists()]
    return bool(cks) and pq.stat().st_mtime > max(c.stat().st_mtime for c in cks)


def compile_forcing(forcing):
    dirs = []
    for reg in REGIONS:
        rd = OUTPUT_ROOT / forcing / f'{reg}_{RUN_SUFFIX}'
        if not rd.exists():
            continue
        with os.scandir(rd) as it:
            dirs += [e.path for e in it
                     if e.is_dir() and e.name.startswith('RGI')]
    print(f'\n  [{forcing}] {len(dirs):,} glacier folders, {N_WORKERS} workers',
          flush=True)
    if not dirs:
        return None, None
    batches = [dirs[i:i + BATCH] for i in range(0, len(dirs), BATCH)]
    ys, ms, n_bad = [], [], 0
    prog = _Progress(len(dirs), f'[{forcing}]')
    with Pool(N_WORKERS) as pool:
        for y, m, bad in pool.imap_unordered(_read_batch, batches):
            if y is not None:
                ys.append(y)
            ms.extend(m)
            n_bad += bad
            prog.update(len(m) + bad)
    prog.close()
    if n_bad:
        print(f'  [{forcing}] skipped {n_bad} folders (incomplete / not v3)',
              flush=True)
    if not ys:
        return None, None
    print(f'  [{forcing}] merging + sorting ...', flush=True)
    y = pd.concat(ys, ignore_index=True)
    del ys
    y = y.sort_values(['rgi_id', 'year']).reset_index(drop=True)
    meta = pd.DataFrame(ms).sort_values('rgi_id').reset_index(drop=True)
    return y, meta


def main():
    t0 = datetime.now()
    COMPILE_ROOT.mkdir(parents=True, exist_ok=True)
    report, metas, fails = [], [], []
    for f in FORCINGS:
        if _already_compiled(f):
            od = COMPILE_ROOT / 'per_forcing' / f
            print(f'\n  [{f}] already compiled (newer than checkpoints) — '
                  f'skipping. Set OGGM_V3_COMPILE_FORCE=1 to redo.', flush=True)
            m = pd.read_csv(od / 'metadata.csv', low_memory=False)
            metas.append(m)
            report.append(f'{f}: {m.rgi_id.nunique():,} glaciers (previously compiled)')
            continue
        y, m = compile_forcing(f)
        if y is None:
            report.append(f'{f}: no data')
            continue
        od = COMPILE_ROOT / 'per_forcing' / f
        od.mkdir(parents=True, exist_ok=True)
        print(f'  [{f}] writing parquet ...', flush=True)
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
        print('  ' + line, flush=True)
        del y
        report.append(line)
    for f in FORCINGS:
        for reg in REGIONS:
            fp = OUTPUT_ROOT / f / f'{reg}_{RUN_SUFFIX}' / 'failures.csv'
            if fp.exists() and fp.stat().st_size > 0:
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
