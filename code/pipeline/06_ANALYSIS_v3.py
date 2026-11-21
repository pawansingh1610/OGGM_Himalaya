"""
================================================================================
06_ANALYSIS_v3.py — core numbers for the paper, from the FULL v3 dataset
================================================================================
Replaces the exploratory notebook for every number that goes in the paper.
No nrows / sampling anywhere; no hard-coded conclusions.

Outputs (in --out, default compiled_v3/analysis/):
  A_calibration_diagnostics.csv      status / bound hits / dT / pf / elev offset
                                     per forcing × sub-region
  B_regional_series.csv              per forcing × unit × year: specific MB,
                                     volume (+low/high), area, n glaciers
  B_ensemble_series.csv              across forcings, with n_forcings per year
  C_volume_change_windows.csv        % volume change in COMMON windows
  D_validation_overall.csv           all-pairs and common-subset metrics
  D_validation_per_glacier.csv       per glacier: bias, anomaly r, Hugonnet
  D_validation_decadal.csv           per decade, WITH N
  D_geodetic_vs_glaciological.csv    field mean vs Hugonnet (bias diagnosis)
  E_split_sample.csv                 2010s-2000s change: model vs Hugonnet
  F_extinction_by_year.csv           per forcing + consensus, forward only
  F_extinct_glaciers.csv
  G_positive_mb_fraction.csv
  REPORT.txt                         human-readable summary of all of the above

Glacier sample used for regional series (stated in REPORT):
  calib_status == 'ok' and backward_vanished == False, per forcing.
  The excluded area fraction is reported so you can judge its effect.

Usage:
  python 06_ANALYSIS_v3.py --hma-shp /path/HMA_subregions.shp \
         [--compiled DIR] [--field Mass_Balance.csv] [--rgi-dir DIR] [--out DIR]
================================================================================
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

FORCINGS = ['ERA5', 'ERA5-Land', 'CRU_TS4', 'CHELSA', 'HARv2']
RHO = 0.900
REPORT = []


def say(s=''):
    print(s)
    REPORT.append(s)


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compiled', default=None)
    ap.add_argument('--hma-shp', required=True)
    ap.add_argument('--hma-name-col', default=None,
                    help='column with sub-region names (auto: full_name/name)')
    ap.add_argument('--field', default=None)
    ap.add_argument('--rgi-dir', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    cache = None
    if a.compiled is None or a.rgi_dir is None or a.field is None:
        from climate_loader_unified import detect_cache_root
        cache = detect_cache_root()
    a.compiled = Path(a.compiled or cache.parent / 'compiled_v3')
    a.rgi_dir = Path(a.rgi_dir or cache / 'RGI')
    if a.field is None:
        for p in [cache / 'Field_MB/Mass_Balance.csv',
                  cache / 'field_data/Mass_Balance.csv']:
            if p.exists():
                a.field = p
                break
    a.out = Path(a.out or a.compiled / 'analysis')
    a.out.mkdir(parents=True, exist_ok=True)
    return a


# ─────────────────────────────────────────────────────────────────────────────
# 0. Glacier lookup: RGI attributes + HMA sub-region (point in polygon)
# ─────────────────────────────────────────────────────────────────────────────

def build_lookup(a):
    import geopandas as gpd
    cache = a.out / 'glacier_lookup.csv'
    if cache.exists():
        return pd.read_csv(cache)
    parts = []
    for shp in sorted(a.rgi_dir.rglob('1[345]_rgi60_*.shp')):
        g = gpd.read_file(shp, ignore_geometry=True)
        if 'SurgeType' not in g and 'Surging' in g:      # RGI6 shapefile name
            g = g.rename(columns={'Surging': 'SurgeType'})
        for c in ['TermType', 'SurgeType', 'Name', 'Zmed']:
            if c not in g:
                g[c] = np.nan
        parts.append(g[['RGIId', 'CenLon', 'CenLat', 'Area', 'Zmed',
                        'TermType', 'SurgeType', 'Name']])
    if not parts:
        raise FileNotFoundError(f'No RGI 13/14/15 shapefiles under {a.rgi_dir}')
    lk = pd.concat(parts, ignore_index=True).drop_duplicates('RGIId')
    lk = lk.rename(columns={'RGIId': 'rgi_id', 'Area': 'area_rgi_km2'})
    lk['rgi_region'] = lk['rgi_id'].str[:8]

    hma = gpd.read_file(a.hma_shp)
    col = a.hma_name_col or next((c for c in ['full_name', 'name', 'Name',
                                              'NAME', 'region', 'long_code']
                                  if c in hma.columns), None)
    if col is None:
        raise ValueError(f'Pass --hma-name-col; columns: {list(hma.columns)}')
    pts = gpd.GeoDataFrame(lk[['rgi_id']],
                           geometry=gpd.points_from_xy(lk.CenLon, lk.CenLat),
                           crs='EPSG:4326')
    hma = hma.to_crs('EPSG:4326')[[col, 'geometry']].rename(
        columns={col: 'subregion'})
    j = gpd.sjoin(pts, hma, how='left', predicate='within')
    j = j.drop_duplicates('rgi_id')[['rgi_id', 'subregion']]
    # glaciers just outside polygons -> nearest polygon
    miss = j['subregion'].isna()
    if miss.any():
        pm = pts[pts.rgi_id.isin(j.loc[miss, 'rgi_id'])].to_crs('ESRI:102025')
        hn = hma.to_crs('ESRI:102025')
        nn = gpd.sjoin_nearest(pm, hn, how='left', max_distance=50_000)
        nn = nn.drop_duplicates('rgi_id').set_index('rgi_id')['subregion']
        j.loc[miss, 'subregion'] = j.loc[miss, 'rgi_id'].map(nn)
    lk = lk.merge(j, on='rgi_id', how='left')
    lk['subregion'] = lk['subregion'].fillna('Outside_HMA_polygons').astype(str)
    lk.to_csv(cache, index=False)
    return lk


def load_meta(a, lk):
    m = pd.read_csv(a.compiled / 'metadata_all.csv', low_memory=False)
    m = m.merge(lk[['rgi_id', 'subregion', 'rgi_region', 'CenLon', 'CenLat',
                    'area_rgi_km2']], on='rgi_id', how='left')
    for c in ['at_tb_bound', 'at_pf_bound', 'backward_vanished']:
        if c in m:
            m[c] = m[c].astype(str).str.lower().isin(['true', '1'])
    m['in_sample'] = (m['calib_status'] == 'ok') & ~m['backward_vanished']
    return m


# ─────────────────────────────────────────────────────────────────────────────
# A. Calibration diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def calib_diagnostics(a, m):
    def agg(g):
        w = g['A_ref_km2']
        return pd.Series({
            'n': len(g),
            'area_km2': w.sum(),
            'pct_ok': (g.calib_status == 'ok').mean() * 100,
            'pct_unreachable': (g.calib_status == 'unreachable').mean() * 100,
            'pct_area_unreachable': w[g.calib_status == 'unreachable'].sum() / w.sum() * 100,
            'pct_tb_bound': g.at_tb_bound.mean() * 100,
            'pct_pf_bound': g.at_pf_bound.mean() * 100,
            'pct_backward_vanished': g.backward_vanished.mean() * 100,
            'median_dT': g.temp_bias.median(),
            'iqr_dT': g.temp_bias.quantile(.75) - g.temp_bias.quantile(.25),
            'median_pf': g.prcp_fac.median(),
            'median_abs_resid': g.geodetic_residual.abs().median(),
            'median_elev_offset_m': g.elev_offset_m.median(),
            'pct_consensus_volume': (g.volume_source.astype(str)
                                     .str.startswith('Farinotti')).mean() * 100,
        })
    rows = []
    for (f, s), g in m.groupby(['forcing', 'subregion']):
        rows.append(agg(g).rename((f, s)))
    for f, g in m.groupby('forcing'):
        rows.append(agg(g).rename((f, 'ALL_HMA')))
    d = pd.DataFrame(rows)
    d.index = pd.MultiIndex.from_tuples(d.index, names=['forcing', 'subregion'])
    d = d.reset_index()
    d.to_csv(a.out / 'A_calibration_diagnostics.csv', index=False)

    say('=' * 78)
    say('A. CALIBRATION DIAGNOSTICS (all HMA)')
    say('=' * 78)
    t = d[d.subregion == 'ALL_HMA'].set_index('forcing')
    say(t[['n', 'pct_ok', 'pct_area_unreachable', 'pct_tb_bound',
           'pct_pf_bound', 'pct_backward_vanished', 'median_dT', 'median_pf',
           'median_elev_offset_m']].round(2).to_string())
    worst = d[d.subregion != 'ALL_HMA'].sort_values('pct_area_unreachable',
                                                     ascending=False).head(8)
    say('\n  Worst sub-regions by unreachable area (%):')
    say(worst[['forcing', 'subregion', 'n', 'pct_area_unreachable',
               'median_dT', 'median_pf']].round(2).to_string(index=False))
    return d


# ─────────────────────────────────────────────────────────────────────────────
# B. Regional series (full data, fixed sample per forcing)
# ─────────────────────────────────────────────────────────────────────────────

YCOLS = ['rgi_id', 'year', 'dm_gt', 'area_start_km2', 'mb_annual_mwe',
         'volume_km3', 'volume_km3_low', 'volume_km3_high', 'area_km2',
         'is_valid']


def read_yearly(a, f):
    p = a.compiled / 'per_forcing' / f / 'yearly.parquet'
    if not p.exists():
        return None
    return pd.read_parquet(p, columns=YCOLS)


def regional_series(a, m, lk):
    out = []
    excl = []
    posfrac = []
    for f in FORCINGS:
        y = read_yearly(a, f)
        if y is None:
            continue
        mf = m[m.forcing == f]
        keep = set(mf.loc[mf.in_sample, 'rgi_id'])
        excl.append({'forcing': f,
                     'n_total': len(mf), 'n_used': len(keep),
                     'area_excluded_pct': 100 * (1 - mf.loc[mf.in_sample,
                                                 'A_ref_km2'].sum()
                                                 / mf['A_ref_km2'].sum())})
        y = y[y.rgi_id.isin(keep)].merge(
            lk[['rgi_id', 'subregion', 'rgi_region']], on='rgi_id', how='left')

        # positive-MB fraction (count- and area-based), all valid glaciers
        v = y[np.isfinite(y.mb_annual_mwe)]
        pf_ = v.groupby('year').apply(lambda g: pd.Series({
            'n': len(g),
            'frac_count': (g.mb_annual_mwe > 0).mean(),
            'frac_area': g.loc[g.mb_annual_mwe > 0, 'area_start_km2'].sum()
                         / g['area_start_km2'].sum()})).reset_index()
        pf_['forcing'] = f
        posfrac.append(pf_)

        for unit_col in ['subregion', 'rgi_region', None]:
            gcols = ['year'] + ([unit_col] if unit_col else [])
            g = y.groupby(gcols).agg(
                dm_gt=('dm_gt', 'sum'), A_start=('area_start_km2', 'sum'),
                V=('volume_km3', 'sum'), V_low=('volume_km3_low', 'sum'),
                V_high=('volume_km3_high', 'sum'), A=('area_km2', 'sum'),
                n=('rgi_id', 'nunique')).reset_index()
            g['unit'] = g[unit_col] if unit_col else 'ALL_HMA'
            g['level'] = unit_col or 'hma'
            g['B_mwe'] = np.where(g.A_start > 0, g.dm_gt / g.A_start * 1e3, np.nan)
            g['forcing'] = f
            out.append(g[['forcing', 'level', 'unit', 'year', 'B_mwe', 'dm_gt',
                          'V', 'V_low', 'V_high', 'A', 'n']])
        del y
    s = pd.concat(out, ignore_index=True)
    # Reference-year row (1999) has no MB; B_mwe NaN there by construction
    s.to_csv(a.out / 'B_regional_series.csv', index=False)
    pd.concat(posfrac).to_csv(a.out / 'G_positive_mb_fraction.csv', index=False)

    # ensemble
    s_b = s.assign(forcing_B=s.forcing.where(s.B_mwe.notna()))
    e = s_b.groupby(['level', 'unit', 'year']).agg(
        n_forcings=('B_mwe', 'count'),          # forcings WITH an MB value
        forcings=('forcing_B', lambda x: '+'.join(sorted(x.dropna()))),
        B_mean=('B_mwe', 'mean'), B_min=('B_mwe', 'min'),
        B_max=('B_mwe', 'max'), B_std=('B_mwe', 'std'),
        V_mean=('V', 'mean'), V_min=('V', 'min'), V_max=('V', 'max'),
        V_low_min=('V_low', 'min'), V_high_max=('V_high', 'max'),
    ).reset_index()
    e.to_csv(a.out / 'B_ensemble_series.csv', index=False)

    say('\n' + '=' * 78)
    say('B. REGIONAL SERIES — glacier sample used')
    say('=' * 78)
    say(pd.DataFrame(excl).round(2).to_string(index=False))
    h = s[(s.level == 'hma')]
    cov = h.dropna(subset=['B_mwe']).groupby('forcing').year.agg(['min', 'max'])
    say('\n  Years with MB per forcing:')
    say(cov.to_string())
    say('\n  NOTE: ensemble statistics carry n_forcings; 5 forcings only in '
        'the years all of them cover (see coverage above).')
    return s, e, pd.DataFrame(excl)


# ─────────────────────────────────────────────────────────────────────────────
# C. Volume change in COMMON windows
# ─────────────────────────────────────────────────────────────────────────────

def volume_windows(a, s):
    """V change over [y0, y1] = V_end(y1) - V_end(y0-1)."""
    cov = s[s.level == 'hma'].dropna(subset=['B_mwe']).groupby('forcing').year
    ymin, ymax = cov.min(), cov.max()
    common_start = int(ymin.max())        # latest first MB year (all forcings)
    common_end = int(ymax.min())          # earliest last year (all forcings)
    last = int(ymax.max())
    first = int(ymin.min())               # earliest first MB year (1950)
    windows = {
        # common to ALL forcings (starts when the latest-starting one does)
        f'recon_{common_start}-1999': (common_start, 1999),
        f'fwd_2000-{common_end}': (2000, common_end),
        'fwd_2000-2019': (2000, 2019),
        f'fwd_2000-{last}': (2000, last),
        f'full_{common_start}-{common_end}': (common_start, common_end),
        # long record: only forcings that start by `first` (ERA5, ERA5-Land,
        # CRU from 1950) — each window lists only forcings covering it
        f'recon_{first}-1999': (first, 1999),
        f'recon_{first}-1979': (first, 1979),
        f'full_{first}-2019': (first, 2019),
        f'full_{first}-{last}': (first, last),
    }

    rows = []
    for (f, lev, u), g in s.groupby(['forcing', 'level', 'unit']):
        gi = g.set_index('year')
        for wname, (y0, y1) in windows.items():
            if (y0 - 1) not in gi.index or y1 not in gi.index:
                continue
            if not (ymin[f] <= y0 and ymax[f] >= y1):
                continue
            v0, v1 = gi.loc[y0 - 1, 'V'], gi.loc[y1, 'V']
            rows.append({'forcing': f, 'level': lev, 'unit': u, 'window': wname,
                         'V_start_km3': v0, 'V_end_km3': v1,
                         'dV_km3': v1 - v0,
                         'dV_pct': (v1 - v0) / v0 * 100 if v0 > 0 else np.nan,
                         'dM_Gt': (v1 - v0) * RHO,
                         'V_start_low': gi.loc[y0 - 1, 'V_low'],
                         'V_start_high': gi.loc[y0 - 1, 'V_high']})
    w = pd.DataFrame(rows)
    w.to_csv(a.out / 'C_volume_change_windows.csv', index=False)

    say('\n' + '=' * 78)
    say('C. VOLUME CHANGE (%) — HMA total, common windows, per forcing')
    say('=' * 78)
    t = w[w.level == 'hma'].pivot_table(index='forcing', columns='window',
                                        values='dV_pct')
    say(t.round(2).to_string())
    t2 = w[w.level == 'hma'].groupby('window').agg(
        n_forcings=('dV_pct', 'count'), mean=('dV_pct', 'mean'),
        min=('dV_pct', 'min'), max=('dV_pct', 'max'),
        forcings=('forcing', lambda x: '+'.join(sorted(x))))
    say('\n  Spread across forcings:')
    say(t2.round(2).to_string())
    v99 = s[(s.level == 'hma') & (s.year == 1999)].groupby('forcing').V.first()
    say(f'\n  V_ref (end 1999, consensus) by forcing [km3] — should be equal '
        f'up to sample differences:\n{v99.round(0).to_string()}')
    return w


# ─────────────────────────────────────────────────────────────────────────────
# D. Field validation
# ─────────────────────────────────────────────────────────────────────────────

def load_field(path):
    df = pd.read_csv(path)
    rc = next(c for c in df.columns if c.upper().replace('_', '') == 'RGIID')
    yc = next(c for c in df.columns if c.lower() in ('year', 'yr', 'hydro_year'))
    mc = next((c for c in df.columns
               if 'annual' in c.lower() and 'balance' in c.lower()), None)
    if mc is None:
        mc = next(c for c in df.columns if c.lower() in ('mb', 'ba', 'annual_mb'))
    nc = next((c for c in df.columns if 'name' in c.lower()), None)
    out = pd.DataFrame({
        'rgi_id': df[rc].astype(str).str.replace('RGI50-', 'RGI60-'),
        'year': pd.to_numeric(df[yc], errors='coerce'),
        'obs': pd.to_numeric(df[mc], errors='coerce'),
        'name': df[nc] if nc else ''}).dropna(subset=['year', 'obs'])
    out['year'] = out['year'].astype(int)
    unit = 'mm w.e.' if out['obs'].abs().median() > 20 else 'm w.e.'
    if unit == 'mm w.e.':
        out['obs'] = out['obs'] / 1000.0
    out = out.drop_duplicates(['rgi_id', 'year'])
    return out, unit, mc


def _metrics(d, col='mod'):
    r = d[col] - d['obs']
    n = len(d)
    if n < 3:
        return pd.Series({'N': n})
    an_o = d['obs'] - d.groupby('rgi_id')['obs'].transform('mean')
    an_m = d[col] - d.groupby('rgi_id')[col].transform('mean')
    return pd.Series({
        'N': n, 'n_glaciers': d.rgi_id.nunique(),
        'bias': r.mean(), 'MAE': r.abs().mean(),
        'RMSE': np.sqrt((r ** 2).mean()),
        'r': np.corrcoef(d['obs'], d[col])[0, 1],
        'r_anomaly': np.corrcoef(an_o, an_m)[0, 1] if an_o.std() > 0 else np.nan,
        'std_obs': d['obs'].std(), 'std_mod': d[col].std(),
    })


def validation(a, m, lk):
    if a.field is None or not Path(a.field).exists():
        say('\nD. VALIDATION skipped — no field file')
        return
    fld, unit, mcol = load_field(a.field)
    say('\n' + '=' * 78)
    say('D. FIELD VALIDATION')
    say('=' * 78)
    say(f'  Field file: {a.field}  column "{mcol}" detected as {unit}')
    say(f'  {len(fld)} glacier-years, {fld.rgi_id.nunique()} glaciers, '
        f'{fld.year.min()}-{fld.year.max()}')
    unknown = set(fld.rgi_id) - set(lk.rgi_id)
    if unknown:
        say(f'  WARNING: {len(unknown)} field RGI ids not in RGI6 13-15: '
            f'{sorted(unknown)[:10]}  (check RGI50->RGI60 mapping)')
    chk = fld.groupby('rgi_id').agg(name=('name', 'first'), n=('year', 'size'),
                                    y0=('year', 'min'), y1=('year', 'max'))
    chk = chk.join(lk.set_index('rgi_id')[['Name', 'area_rgi_km2',
                                           'subregion']], how='left')
    chk.to_csv(a.out / 'D_field_glacier_check.csv')
    say('  -> D_field_glacier_check.csv: compare field name vs RGI6 Name/area '
        'to verify the id mapping')

    pairs = []
    for f in FORCINGS:
        y = read_yearly(a, f)
        if y is None:
            continue
        y = y[y.rgi_id.isin(fld.rgi_id)][['rgi_id', 'year', 'mb_annual_mwe']]
        y = y.rename(columns={'mb_annual_mwe': f})
        pairs.append(y.set_index(['rgi_id', 'year']))
    if not pairs:
        return
    P = fld.set_index(['rgi_id', 'year']).join(pairs, how='left').reset_index()
    P = P.merge(lk[['rgi_id', 'subregion']], on='rgi_id', how='left')
    P.to_csv(a.out / 'D_validation_pairs.csv', index=False)
    fs = [f for f in FORCINGS if f in P]

    rows = []
    subsets = {
        'all_pairs': None,
        'common_all5': fs,
        'common_ERA5_ERA5L_HARv2': [f for f in ['ERA5', 'ERA5-Land', 'HARv2'] if f in fs],
    }
    for sname, req in subsets.items():
        for f in fs:
            if req is not None and f not in req:
                continue
            need = [f] if req is None else req
            d = P.dropna(subset=need).rename(columns={f: 'mod'})
            rows.append(_metrics(d).rename((sname, f)))
    ov = pd.DataFrame(rows).reindex(columns=['N', 'n_glaciers', 'bias', 'RMSE', 'r', 'r_anomaly', 'std_obs', 'std_mod'])
    if ov.empty or ov['N'].fillna(0).max() < 3:
        say('  Not enough field glacier-years modelled yet (partial run?) — '
            'validation skipped.')
        return
    ov.index = pd.MultiIndex.from_tuples(ov.index, names=['subset', 'forcing'])
    ov.to_csv(a.out / 'D_validation_overall.csv')
    say('\n  Overall (subset = which glacier-years are compared):')
    say(ov[['N', 'n_glaciers', 'bias', 'RMSE', 'r', 'r_anomaly',
            'std_obs', 'std_mod']].round(3).to_string())

    # decadal WITH N, common-3 subset and all pairs
    P['decade'] = (P.year // 10) * 10
    dec = []
    for f in fs:
        for dcd, g in P.dropna(subset=[f]).groupby('decade'):
            dec.append(_metrics(g.rename(columns={f: 'mod'})).rename((f, dcd)))
    dd = pd.DataFrame(dec).reindex(columns=['N', 'n_glaciers', 'bias', 'RMSE', 'r', 'r_anomaly', 'std_obs', 'std_mod'])
    dd.index = pd.MultiIndex.from_tuples(dd.index, names=['forcing', 'decade'])
    dd.to_csv(a.out / 'D_validation_decadal.csv')
    say('\n  Decadal (all pairs) — read N before interpreting RMSE:')
    say(dd[['N', 'n_glaciers', 'bias', 'RMSE']].round(3).unstack(0)
        .to_string())

    # per glacier + Hugonnet: where does the bias come from?
    mm = m.set_index(['forcing', 'rgi_id'])
    pg = []
    for f in fs:
        for gid, g in P.dropna(subset=[f]).groupby('rgi_id'):
            row = {'forcing': f, 'rgi_id': gid, 'name': g['name'].iloc[0],
                   'subregion': g['subregion'].iloc[0], 'N': len(g),
                   'y0': g.year.min(), 'y1': g.year.max(),
                   'obs_mean': g.obs.mean(), 'mod_mean': g[f].mean(),
                   'bias': (g[f] - g.obs).mean(),
                   'r_anomaly': (np.corrcoef(g.obs, g[f])[0, 1]
                                 if len(g) > 3 and g.obs.std() > 0 else np.nan)}
            if (f, gid) in mm.index:
                r = mm.loc[(f, gid)]
                row.update({'hug_2000_2020': r.get('hug_obs_2000_2020'),
                            'mod_2000_2020': r.get('hug_mod_2000_2020'),
                            'calib_status': r.get('calib_status'),
                            'temp_bias': r.get('temp_bias'),
                            'prcp_fac': r.get('prcp_fac')})
            g2 = g[(g.year >= 2000) & (g.year <= 2019)]
            row['obs_mean_2000_2019'] = g2.obs.mean() if len(g2) >= 3 else np.nan
            row['n_2000_2019'] = len(g2)
            pg.append(row)
    pg = pd.DataFrame(pg)
    pg.to_csv(a.out / 'D_validation_per_glacier.csv', index=False)

    # Geodetic vs glaciological on the same glaciers, same period
    gg = pg.drop_duplicates('rgi_id')
    gg = gg[gg.n_2000_2019 >= 3][['rgi_id', 'name', 'n_2000_2019',
                                  'obs_mean_2000_2019', 'hug_2000_2020']]
    gg['glac_minus_geod'] = gg.obs_mean_2000_2019 - gg.hug_2000_2020
    gg.to_csv(a.out / 'D_geodetic_vs_glaciological.csv', index=False)
    say('\n  Bias diagnosis — field (glaciological) vs Hugonnet (geodetic) on '
        'the SAME glaciers, 2000-2019:')
    if len(gg):
        say(gg.round(3).to_string(index=False))
        say(f'  mean (glaciological − geodetic) = {gg.glac_minus_geod.mean():+.3f}'
            f' m w.e./yr over {len(gg)} glaciers.')
        say('  If this is strongly positive, the model (which matches Hugonnet '
            'by construction) MUST look negatively biased against field data: '
            'the "bias" is an observation-type discrepancy, not debris/'
            'avalanches.')
    else:
        say('  (no field glacier has >= 3 years in 2000-2019)')
    say('\n  Share of bias explained by period: compare D_validation_decadal '
        '(pre-2000 = extrapolated, not calibrated).')


# ─────────────────────────────────────────────────────────────────────────────
# E. Split-sample geodetic test (2010s minus 2000s)
# ─────────────────────────────────────────────────────────────────────────────

def split_sample(a, m):
    need = ['hug_obs_2000_2010', 'hug_obs_2010_2020',
            'hug_mod_2000_2010', 'hug_mod_2010_2020']
    if not all(c in m for c in need):
        say('\nE. SPLIT-SAMPLE skipped (columns missing)')
        return
    d = m[(m.calib_period == '2000-01-01_2020-01-01') & m.in_sample].dropna(
        subset=need).copy()
    d['d_obs'] = d.hug_obs_2010_2020 - d.hug_obs_2000_2010
    d['d_mod'] = d.hug_mod_2010_2020 - d.hug_mod_2000_2010
    d['w'] = d.A_ref_km2
    rows = []
    for (f, s), g in list(d.groupby(['forcing', 'subregion'])) + \
            [((f, 'ALL_HMA'), g) for f, g in d.groupby('forcing')]:
        w = g.w
        do = np.average(g.d_obs, weights=w)
        dm = np.average(g.d_mod, weights=w)
        big = g[g.A_ref_km2 >= 1.0]      # single-glacier Hugonnet noise
        rows.append({'forcing': f, 'subregion': s, 'n': len(g),
                     'dObs_areaw': do, 'dMod_areaw': dm,
                     'dMod_minus_dObs': dm - do,
                     'r_glacier_ge1km2': (np.corrcoef(big.d_obs, big.d_mod)[0, 1]
                                          if len(big) > 10 else np.nan),
                     'n_ge1km2': len(big)})
    e = pd.DataFrame(rows)
    e.to_csv(a.out / 'E_split_sample.csv', index=False)
    say('\n' + '=' * 78)
    say('E. SPLIT-SAMPLE GEODETIC TEST — change 2010-19 minus 2000-09 '
        '[m w.e./yr], area-weighted')
    say('   (calibrated on the 20-yr mean only: does each forcing reproduce '
        'the observed acceleration?)')
    say('=' * 78)
    say(e[e.subregion == 'ALL_HMA'].round(3).to_string(index=False))
    piv = e[e.subregion != 'ALL_HMA'].pivot_table(index='subregion',
                                                  columns='forcing',
                                                  values='dMod_minus_dObs')
    say('\n  dMod − dObs by sub-region (0 = perfect):')
    say(piv.round(3).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# F. Extinction (forward period only)
# ─────────────────────────────────────────────────────────────────────────────

def extinction(a, m, s):
    cov = s[s.level == 'hma'].dropna(subset=['B_mwe']).groupby('forcing').year
    ymax = cov.max().to_dict()
    ok = m[m.calib_status == 'ok'].copy()
    ok['extinct_year'] = pd.to_numeric(ok['extinct_year'], errors='coerce')
    years = range(2000, int(max(ymax.values())) + 1)

    per = []
    for f, g in ok.groupby('forcing'):
        c = g.extinct_year.value_counts().sort_index()
        for y in years:
            if y <= ymax.get(f, 0):
                per.append({'forcing': f, 'year': y, 'n_extinct': int(c.get(y, 0))})
    per = pd.DataFrame(per)

    # consensus among forcings AVAILABLE in that year
    wide = ok.drop_duplicates(['rgi_id', 'forcing']).set_index(
        ['rgi_id', 'forcing'])
    pv = wide['extinct_year'].unstack('forcing')            # NaN = survives
    present = wide['temp_bias'].unstack('forcing').notna()  # same index
    pv = pv.reindex(index=present.index, columns=present.columns)
    cons = {}
    for y in years:
        avail = [f for f in pv.columns if ymax.get(f, 0) >= y]
        if len(avail) < 3:
            continue
        n_av = present[avail].sum(axis=1)
        n_ex = (pv[avail] <= y).sum(axis=1)
        flag = (n_av >= 3) & (n_ex >= np.ceil(0.5 * n_av)) & (n_ex >= 2)
        for gid in flag[flag].index:
            cons.setdefault(gid, (y, len(avail)))
    cdf = pd.DataFrame([{'rgi_id': k, 'consensus_year': v[0],
                         'n_forcings_available': v[1]} for k, v in cons.items()])
    if len(cdf):
        cdf = cdf.merge(ok.drop_duplicates('rgi_id')[['rgi_id', 'subregion',
                                                      'A_ref_km2']],
                        on='rgi_id', how='left')
    cdf.to_csv(a.out / 'F_extinct_glaciers.csv', index=False)
    cy = (cdf.groupby('consensus_year').size() if len(cdf)
          else pd.Series(dtype=int))
    byyear = per.pivot_table(index='year', columns='forcing',
                             values='n_extinct').reindex(years)
    byyear['consensus'] = cy.reindex(years).fillna(0).astype(int)
    byyear['n_forcings_available'] = [sum(ymax.get(f, 0) >= y for f in ymax)
                                      for y in years]
    byyear.to_csv(a.out / 'F_extinction_by_year.csv')

    say('\n' + '=' * 78)
    say('F. EXTINCTION (V reaches 0 after 2000; consensus = >=50 % of the '
        'forcings available that year, min 2, needs >=3 available)')
    say('=' * 78)
    dec = byyear.copy()
    dec['decade'] = (dec.index // 5) * 5
    say(dec.groupby('decade').agg('sum').drop(columns='n_forcings_available')
        .round(0).to_string())
    say('  (5-yr bins; consensus definition changes when forcings drop out — '
        'see n_forcings_available in the csv)')
    if len(cdf):
        say(f'  Consensus extinct: {len(cdf)} glaciers; area at RGI epoch: '
            f'median {cdf.A_ref_km2.median():.3f} km2, '
            f'{(cdf.A_ref_km2 < 0.05).mean() * 100:.0f} % < 0.05 km2')


# ─────────────────────────────────────────────────────────────────────────────

def main():
    a = parse_args()
    say(f'v3 analysis — compiled data: {a.compiled}')
    lk = build_lookup(a)
    m = load_meta(a, lk)
    say(f'Metadata rows: {len(m):,}  glaciers: {m.rgi_id.nunique():,}  '
        f'forcings: {sorted(m.forcing.unique())}')
    calib_diagnostics(a, m)
    s, e, _ = regional_series(a, m, lk)
    volume_windows(a, s)
    validation(a, m, lk)
    split_sample(a, m)
    extinction(a, m, s)
    (a.out / 'REPORT.txt').write_text('\n'.join(REPORT))
    print(f'\nAll outputs in {a.out}')


if __name__ == '__main__':
    main()
