#!/usr/bin/env python
"""Check 2025 in the ERA5, ERA5-Land and HARv2 inputs: months present, NaN, and 2025 monthly means over the
HMA box (67-105E, 26.7-46N) against 2015-2024 (mean, min, max, z-score). Output: tables/climate_2025_check.txt
Gap-filling per glacier is recorded as n_filled_months in per_forcing/<F>/metadata.csv (all 0 in run_04_full)."""
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from config import DOWNLOAD_CACHE  # noqa: E402
import xarray as xr, numpy as np, pandas as pd
C=str(DOWNLOAD_CACHE) + "/"
files={'ERA5':(C+'ERA5/era5_t2m.nc',C+'ERA5/era5_tp.nc'),
       'ERA5-Land':(C+'era5_land_data/raw/era5_land_t2m_1950_2026.nc',C+'era5_land_data/raw/era5_land_tp_1950_2026.nc'),
       'HARv2':(C+'HRAv2/HARv2_d10km_m_2d_t2_1980_2025.nc',C+'HRAv2/HARv2_d10km_m_2d_prcp_1980_2025.nc')}
for f,(ft,fp) in files.items():
    for path in (ft,fp):
        ds=xr.open_dataset(path)
        v=[k for k in ds.data_vars if ds[k].ndim>=3][0]
        tn=[d for d in ds[v].dims if 'time' in d.lower()][0]
        t=pd.to_datetime(ds[tn].values)
        print(f'== {f} {path.split("/")[-1]} var={v} dims={ds[v].dims} shape={ds[v].shape} time {t.min():%Y-%m} .. {t.max():%Y-%m} n={len(t)} units={ds[v].attrs.get("units")}')
        print('   attrs:', {k:ds[v].attrs[k] for k in list(ds[v].attrs)[:6]})
        sp=[d for d in ds[v].dims if d!=tn]
        # HMA box subset if lat/lon dims
        da=ds[v]
        if 'latitude' in da.dims or 'lat' in da.dims:
            la='latitude' if 'latitude' in da.dims else 'lat'; lo='longitude' if 'longitude' in da.dims else 'lon'
            lat=da[la].values; sl=slice(46,26.7) if lat[0]>lat[-1] else slice(26.7,46)
            da=da.sel({la:sl, lo:slice(67,105)})
        sub=da.sel({tn:slice('2015-01-01','2025-12-31')})
        tt=pd.to_datetime(sub[tn].values)
        m=sub.mean(dim=[d for d in sub.dims if d!=tn]).values
        nanfrac=sub.isnull().mean(dim=[d for d in sub.dims if d!=tn]).values
        s=pd.DataFrame({'t':tt,'mean':m,'nan':nanfrac})
        s['yr']=s.t.dt.year; s['mo']=s.t.dt.month
        n25=(s.yr==2025).sum(); print('   2025 months present:',sorted(s[s.yr==2025].mo.tolist()), ' max NaN frac 2025:',round(s[s.yr==2025].nan.max(),4),' (2015-24 max', round(s[s.yr<2025].nan.max(),4),')')
        clim=s[s.yr<2025].groupby('mo')['mean'].agg(['mean','std','min','max'])
        y25=s[s.yr==2025].set_index('mo')['mean']
        z=((y25-clim['mean'])/clim['std']).round(2)
        out=pd.DataFrame({'2025':y25.round(4),'clim15_24':clim['mean'].round(4),'min':clim['min'].round(4),'max':clim['max'].round(4),'z':z})
        print(out.to_string())
        if (s.yr==2026).any(): print('   2026 months:',sorted(s[s.yr==2026].mo.tolist()))

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PER_FORCING
M = str(PER_FORCING / '{}' / 'metadata.csv')
for f in ['ERA5', 'ERA5-Land', 'HARv2']:
    m = pd.read_csv(M.format(f), usecols=['n_filled_months'])
    print(f'{f}: glaciers with gap-filled months = {(m.n_filled_months > 0).sum()} (max {m.n_filled_months.max()})')
