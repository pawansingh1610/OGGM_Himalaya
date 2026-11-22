"""
================================================================================
UNIFIED CLIMATE LOADER FOR 5 FORCINGS  — v1.2
================================================================================
Supports: ERA5, ERA5-Land, CRU_TS4, CHELSA-W5E5, HARv2

v1.2 (analysis fixes):
  - NEW grid_elevation(): elevation of the climate grid cell actually used.
    This MUST be the OGGM ref_hgt. v1.1 callers used glacier mid-elevation,
    so temp_bias had to absorb a 1–2 km height error.
  - FIX month-shift bug: v1.1 did df.dropna() and the OGGM writer only
    checked len % 12. One missing month in the middle silently shifted
    every earlier month's calendar month (OGGM counts years from the end).
    Now the series is re-indexed to a complete monthly axis; gaps of
    <= MAX_GAP_MONTHS are filled with that calendar month's climatology
    (count reported in meta['n_filled_months']); longer gaps raise.
  - write_oggm_climate_file(): writes only complete calendar years and
    asserts a gap-free monthly axis.

v1.1 fixes (kept):
  - ERA5 / ERA5-Land precipitation m/day -> mm/month (*1000*30.4375)
  - ERA5-Land OGGM flat format ('points' dim)
  - HARv2 2D lat/lon cached at module level

All outputs: df with columns ['time', 'temp_C', 'prcp_mm']
  temp_C  : degrees Celsius   (valid at meta['grid_elev_m'] if requested)
  prcp_mm : mm / month

Usage:
    from climate_loader_unified import detect_cache_root, load_climate, \
        grid_elevation
    cache    = detect_cache_root()
    df, meta = load_climate('ERA5', lon, lat, cache)
    zref     = grid_elevation('ERA5', lon, lat, cache)['grid_elev_m']
================================================================================
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings('ignore')

MAX_GAP_MONTHS = 3      # longer gaps inside the record -> error
G = 9.80665


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_HMA_ROOT = Path(os.environ.get("HMA_OGGM_ROOT", Path.home() / "OGGM"))
_CANDIDATE_ROOTS = [
    Path(os.environ["HMA_DOWNLOAD_CACHE"]) if os.environ.get("HMA_DOWNLOAD_CACHE") else _HMA_ROOT / "download_cache",
    Path("/scratch/OGGM/download_cache"),
]
_CACHE_MARKER = "chelsa_w5e5"


def detect_cache_root(hint: str = None) -> Path:
    """
    Find the OGGM download cache root automatically.
      1. explicit hint  2. config.json next to this script
      3. known candidate paths  4. walk upward from this script
    """
    if hint is not None:
        p = Path(hint)
        if p.exists():
            return p
        raise FileNotFoundError(f"Provided cache hint does not exist: {p}")

    config_file = Path(__file__).resolve().parent / 'config.json'
    if config_file.exists():
        try:
            cfg = json.loads(config_file.read_text())
            p = Path(cfg['data_root'])
            if p.exists():
                return p
        except Exception:
            pass

    for c in _CANDIDATE_ROOTS:
        if c.exists() and (c / _CACHE_MARKER).exists():
            return c

    here = Path(__file__).resolve().parent
    for parent in [here] + list(here.parents):
        # <parent>/download_cache  or  <parent>/OGGM/download_cache
        for marker in [parent / "download_cache",
                       parent / "OGGM" / "download_cache"]:
            if marker.exists() and (marker / _CACHE_MARKER).exists():
                return marker

    raise FileNotFoundError(
        "Cannot locate OGGM download cache.\n"
        "Run precheck.py first to create config.json, or pass "
        "cache_root explicitly to load_climate().")


# ─────────────────────────────────────────────────────────────────────────────
# PATH REGISTRY — climate
# ─────────────────────────────────────────────────────────────────────────────

_REL_PATHS = {
    'era5_temp':   ["ERA5/era5_t2m.nc", "ERA5/era5_t2m_HMA.nc", "era5/era5_t2m.nc"],
    'era5_precip': ["ERA5/era5_tp.nc", "ERA5/era5_tp_HMA.nc", "era5/era5_tp.nc"],
    'era5l_temp': [
        "era5_land_data/raw/era5_land_t2m_1950_2026.nc",
        "era5_land_data/raw/era5_land_t2m_1950_2025.nc",
        "cluster.klima.uni-bremen.de/~oggm/climate/era5-land/monthly/v1.0/"
        "era5_land_monthly_t2m_1981-2018_flat.nc",
    ],
    'era5l_precip': [
        # run_07: ERA5-Land monthly tp for Sep 2022-Feb 2024 replaced (ECMWF known issue, accumulated variables);
        # file made by scripts/era5land_tp_fix.py. If absent the loader falls through to the original file.
        "era5_land_data/raw/era5_land_tp_1950_2026_patched.nc",
        "era5_land_data/raw/era5_land_tp_1950_2026.nc",
        "era5_land_data/raw/era5_land_tp_1950_2025.nc",
        "cluster.klima.uni-bremen.de/~oggm/climate/era5-land/monthly/v1.0/"
        "era5_land_monthly_prcp_1981-2018_flat.nc",
    ],
    'cru_temp': [
        "cru_ts4/cru_ts4.04.1901.2019.tmp.dat.nc",
        "cru_ts4/cru_ts4.08.1901.2023.tmp.dat.nc",
        "cru_ts4/cru_ts4.07.1901.2022.tmp.dat.nc",
    ],
    'cru_precip': [
        "cru_ts4/cru_ts4.04.1901.2019.pre.dat.nc",
        "cru_ts4/cru_ts4.08.1901.2023.pre.dat.nc",
        "cru_ts4/cru_ts4.07.1901.2022.pre.dat.nc",
    ],
    'chelsa_temp': [
        "chelsa_w5e5/chelsa-w5e5_obsclim_tas_monthlymean_1979-2016_HMA.nc",
        "chelsa_w5e5/chelsa_w5e5_tas_1979_2016.nc",
    ],
    'chelsa_precip': [
        "chelsa_w5e5/chelsa-w5e5_obsclim_pr_monthlytotal_HMA.nc",
        "chelsa_w5e5/chelsa-w5e5_obsclim_pr_monthlymean_1979-2016_HMA.nc",
        "chelsa_w5e5/chelsa_w5e5_pr_1979_2016.nc",
    ],
    'harv2_temp': [
        "HRAv2/HARv2_d10km_m_2d_t2_1980_2025.nc",
        "HARv2/HARv2_d10km_m_2d_t2_1980_2025.nc",
    ],
    'harv2_precip': [
        "HRAv2/HARv2_d10km_m_2d_prcp_1980_2025.nc",
        "HARv2/HARv2_d10km_m_2d_prcp_1980_2025.nc",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# PATH REGISTRY — grid-cell orography (NEW v1.2)
# Same grid as the climate files. Add your own paths at the top of a list.
#   ERA5 / ERA5-Land : invariant 'geopotential' (z, m2 s-2)  -> / g
#   CRU TS4          : CRU CL2.0 elevation (elv/elev, km)    -> * 1000
#   CHELSA-W5E5      : 'orog' (m)
#   HARv2            : static 'hgt' (m)
# ─────────────────────────────────────────────────────────────────────────────

OROG_PATHS = {
    'ERA5': [
        "ERA5/era5_invariant.nc",
        "ERA5/era5_geopotential.nc",
        "ERA5/era5_orography.nc",
        "cluster.klima.uni-bremen.de/~oggm/climate/era5/monthly/v1.1/era5_invariant.nc",
        "cluster.klima.uni-bremen.de/~oggm/climate/era5/monthly/v1.0/era5_invariant.nc",
    ],
    'ERA5-Land': [
        # OGGM-published invariants first (a CDS 'era5_land_invariant.nc'
        # download can contain z = 100 % NaN)
        "cluster.klima.uni-bremen.de/~oggm/climate/era5-land/monthly/vhma/"
        "era5_land_invariant_flat_HMA.nc",
        "cluster.klima.uni-bremen.de/~oggm/climate/era5-land/monthly/v1.0/"
        "era5_land_invariant_flat.nc",
        "era5_land_data/raw/era5_land_geopotential.nc",
        "era5_land_data/raw/era5_land_invariant.nc",
        "era5_land_data/era5_land_invariant.nc",
    ],
    'CRU_TS4': [
        "cru_ts4/cru_cl2.0_elv.nc",
        "cru_ts4/cru_elev.nc",
        "cru_ts4/cru_cl2.nc",
        "cluster.klima.uni-bremen.de/~oggm/climate/cru/cru_cl2.nc",
    ],
    'CHELSA': [
        "chelsa_w5e5/chelsa-w5e5_orog_HMA.nc",
        "chelsa_w5e5/chelsa-w5e5_obsclim_orog_30arcsec_HMA.nc",
        "chelsa_w5e5/chelsa-w5e5_orog_30arcsec_global.nc",
        "chelsa_w5e5/orog.nc",
    ],
    'HARv2': [
        "HRAv2/HARv2_d10km_static_hgt.nc",
        "HARv2/HARv2_d10km_static_hgt.nc",
        "HRAv2/HARv2_d10km_static.nc",
        "HARv2/HARv2_d10km_static.nc",
    ],
}


def _resolve(key: str, cache_root: Path) -> Path:
    for rel in _REL_PATHS.get(key, []):
        p = cache_root / rel
        if p.exists():
            return p
    tried = [str(cache_root / r) for r in _REL_PATHS.get(key, [])]
    raise FileNotFoundError(
        f"Cannot find dataset '{key}'.\nTried:\n" +
        "\n".join(f"  {t}" for t in tried))


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_1d(ds, lon, lat, lat_dim, lon_dim):
    lon_vals = ds[lon_dim].values
    lon_q = lon + 360.0 if (lon_vals.max() > 180 and lon < 0) else lon
    lat_idx = int(np.argmin(np.abs(ds[lat_dim].values - lat)))
    lon_idx = int(np.argmin(np.abs(lon_vals - lon_q)))
    return ds.isel({lat_dim: lat_idx, lon_dim: lon_idx})


def _convert_precip(prcp: np.ndarray, units: str) -> np.ndarray:
    """mm/month from m/day (ERA5), kg m-2 s-1, or mm/month."""
    prcp = np.array(prcp, dtype=float)
    u = (units or '').lower().strip()
    if u in ('m', 'metres', 'meter', 'meters'):
        positive = prcp[prcp > 0]
        mean_pos = float(np.nanmean(positive)) if len(positive) > 0 else 0.0
        prcp = prcp * 1000.0 * (30.4375 if mean_pos < 0.1 else 1.0)
    elif 'kg' in u or 's-1' in u:
        prcp = prcp * 30.4375 * 86400.0
    return np.clip(prcp, 0.0, None)


def _to_monthly_df(time_vals, temp_vals, prcp_vals,
                   temp_in_kelvin: bool, prcp_unit: str,
                   meta: dict = None) -> pd.DataFrame:
    """
    Clean monthly DataFrame on a COMPLETE, gap-free monthly axis.
    Missing months inside the record are filled with that calendar
    month's mean if the gap is <= MAX_GAP_MONTHS, otherwise ValueError.
    Leading/trailing NaN months are trimmed (not filled).
    """
    temp = np.array(temp_vals, dtype=float)
    prcp = np.array(prcp_vals, dtype=float)
    if temp_in_kelvin or np.nanmean(temp) > 100:
        temp = temp - 273.15
    prcp = _convert_precip(prcp, prcp_unit)

    times = pd.to_datetime(time_vals).to_period('M').to_timestamp()
    n = min(len(times), len(temp), len(prcp))
    df = pd.DataFrame({'time': times[:n], 'temp_C': temp[:n],
                       'prcp_mm': prcp[:n]})
    df = df.drop_duplicates('time').sort_values('time').set_index('time')

    valid = df.dropna()
    if valid.empty:
        raise ValueError("Climate series is all-NaN at this grid cell "
                         "(e.g. ERA5-Land sea/lake cell)")
    full = pd.date_range(valid.index.min(), valid.index.max(), freq='MS')
    df = df.reindex(full)

    n_filled = 0
    bad = df['temp_C'].isna() | df['prcp_mm'].isna()
    if bad.any():
        runs = (bad != bad.shift()).cumsum()[bad]
        longest = runs.value_counts().max()
        if longest > MAX_GAP_MONTHS:
            raise ValueError(f"Climate gap of {longest} consecutive months "
                             f"inside the record — refusing to fill")
        clim = df.groupby(df.index.month).mean()
        for c in ['temp_C', 'prcp_mm']:
            m = df[c].isna()
            df.loc[m, c] = clim.loc[df.index[m].month, c].values
        n_filled = int(bad.sum())
    if meta is not None:
        meta['n_filled_months'] = n_filled

    df.index.name = 'time'
    return df.reset_index()


def _trim_years(df, start_year=None, end_year=None):
    if start_year:
        df = df[df['time'].dt.year >= start_year]
    if end_year:
        df = df[df['time'].dt.year <= end_year]
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# PER-FORCING LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_era5(lon, lat, cache_root):
    """ERA5 0.25 deg monthly: t2m [K], tp [m/day]."""
    temp_path = _resolve('era5_temp', cache_root)
    precip_path = _resolve('era5_precip', cache_root)
    with xr.open_dataset(temp_path, engine='netcdf4') as ds:
        time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
        t_var = next((v for v in ['t2m', 'T2M', 'temperature']
                      if v in ds.data_vars), list(ds.data_vars)[0])
        pt = _nearest_1d(ds, lon, lat, 'latitude', 'longitude')
        temp_vals = pt[t_var].values
        time_vals = ds[time_dim].values
    with xr.open_dataset(precip_path, engine='netcdf4') as ds:
        p_var = next((v for v in ['tp', 'TP', 'precipitation']
                      if v in ds.data_vars), list(ds.data_vars)[0])
        p_unit = ds[p_var].attrs.get('units', 'm')
        pp = _nearest_1d(ds, lon, lat, 'latitude', 'longitude')
        prcp_vals = pp[p_var].values
    meta = {'forcing': 'ERA5', 'resolution': '0.25deg (~31km)',
            'temp_var': t_var, 'prcp_var': p_var,
            'files': [str(temp_path), str(precip_path)]}
    df = _to_monthly_df(time_vals, temp_vals, prcp_vals, True, p_unit, meta)
    meta['period'] = f"{df['time'].dt.year.min()}-{df['time'].dt.year.max()}"
    return df, meta


def _load_era5land(lon, lat, cache_root):
    """ERA5-Land ~9 km; regular grid or OGGM flat 'points' format."""
    temp_path = _resolve('era5l_temp', cache_root)
    precip_path = _resolve('era5l_precip', cache_root)

    def _extract(path, var_candidates):
        with xr.open_dataset(path, engine='netcdf4') as ds:
            var = next((v for v in var_candidates if v in ds.data_vars),
                       list(ds.data_vars)[0])
            units = ds[var].attrs.get('units', 'm')
            time_dim = next((d for d in ['time', 'valid_time']
                             if d in ds.dims), None)
            if time_dim is None:
                raise ValueError(f"No time dimension in {path.name}")
            times = ds[time_dim].values
            if 'points' in ds.dims:
                lat_arr = ds['latitude'].values
                lon_arr = ds['longitude'].values
                lon_q = lon + 360 if (lon_arr.max() > 180 and lon < 0) else lon
                idx = int(np.argmin((lat_arr - lat) ** 2 + (lon_arr - lon_q) ** 2))
                vals = ds[var].isel(points=idx).values
            else:
                lat_dim = next((d for d in ['latitude', 'lat'] if d in ds.dims), None)
                lon_dim = next((d for d in ['longitude', 'lon'] if d in ds.dims), None)
                if lat_dim is None or lon_dim is None:
                    raise ValueError(f"Cannot find lat/lon dims in {path.name}: "
                                     f"{list(ds.dims)}")
                vals = _nearest_1d(ds, lon, lat, lat_dim, lon_dim)[var].values
        n = min(len(times), len(vals))
        return vals[:n], times[:n], units, var

    temp_vals, time_vals, _, t_var = _extract(
        temp_path, ['t2m', 'temperature', 'T2M', 'tmp'])
    prcp_vals, time_p, p_unit, p_var = _extract(
        precip_path, ['tp', 'precipitation', 'prcp', 'pr'])

    # Align temp/precip on TIME (v1.1 aligned by position)
    t = pd.Series(temp_vals, index=pd.to_datetime(time_vals).to_period('M'))
    p = pd.Series(prcp_vals, index=pd.to_datetime(time_p).to_period('M'))
    t = t[~t.index.duplicated()]
    p = p[~p.index.duplicated()]
    j = pd.concat([t, p], axis=1, join='inner')
    meta = {'forcing': 'ERA5-Land', 'resolution': '0.1deg (~9km)',
            'temp_var': t_var, 'prcp_var': p_var,
            'files': [str(temp_path), str(precip_path)]}
    df = _to_monthly_df(j.index.to_timestamp(), j.iloc[:, 0].values,
                        j.iloc[:, 1].values, True, p_unit, meta)
    meta['period'] = f"{df['time'].dt.year.min()}-{df['time'].dt.year.max()}"
    return df, meta


def _load_cru(lon, lat, cache_root):
    """CRU TS4 0.5 deg: tmp [degC], pre [mm/month]."""
    temp_path = _resolve('cru_temp', cache_root)
    precip_path = _resolve('cru_precip', cache_root)
    with xr.open_dataset(temp_path, engine='netcdf4') as ds:
        t_var = next((v for v in ['tmp', 'temp', 'temperature']
                      if v in ds.data_vars), list(ds.data_vars)[0])
        temp_vals = _nearest_1d(ds, lon, lat, 'lat', 'lon')[t_var].values
        time_vals = ds['time'].values
    with xr.open_dataset(precip_path, engine='netcdf4') as ds:
        p_var = next((v for v in ['pre', 'prcp', 'precipitation']
                      if v in ds.data_vars), list(ds.data_vars)[0])
        prcp_vals = _nearest_1d(ds, lon, lat, 'lat', 'lon')[p_var].values
    meta = {'forcing': 'CRU_TS4', 'resolution': '0.5deg (~50km)',
            'temp_var': t_var, 'prcp_var': p_var,
            'files': [str(temp_path), str(precip_path)]}
    df = _to_monthly_df(time_vals, temp_vals, prcp_vals, False, 'mm', meta)
    meta['period'] = f"{df['time'].dt.year.min()}-{df['time'].dt.year.max()}"
    return df, meta


def _load_chelsa(lon, lat, cache_root):
    """CHELSA-W5E5 1 km HMA 1979-2016: tas [K], pr [mm/month or kg m-2 s-1]."""
    temp_path = _resolve('chelsa_temp', cache_root)
    precip_path = _resolve('chelsa_precip', cache_root)
    with xr.open_dataset(temp_path, engine='netcdf4') as ds:
        t_var = next((v for v in ['tas', 'temp', 't2m']
                      if v in ds.data_vars), list(ds.data_vars)[0])
        lat_d = next(d for d in ['lat', 'latitude'] if d in ds.dims)
        lon_d = next(d for d in ['lon', 'longitude'] if d in ds.dims)
        temp_vals = _nearest_1d(ds, lon, lat, lat_d, lon_d)[t_var].values
        time_vals = ds['time'].values
    with xr.open_dataset(precip_path, engine='netcdf4') as ds:
        p_var = next((v for v in ['pr', 'prcp', 'precipitation']
                      if v in ds.data_vars), list(ds.data_vars)[0])
        p_unit = ds[p_var].attrs.get('units', 'mm')
        lat_d = next(d for d in ['lat', 'latitude'] if d in ds.dims)
        lon_d = next(d for d in ['lon', 'longitude'] if d in ds.dims)
        prcp_vals = _nearest_1d(ds, lon, lat, lat_d, lon_d)[p_var].values
    meta = {'forcing': 'CHELSA', 'resolution': '~1km (0.008deg)',
            'temp_var': t_var, 'prcp_var': p_var,
            'files': [str(temp_path), str(precip_path)]}
    df = _to_monthly_df(time_vals, temp_vals, prcp_vals, True, p_unit, meta)
    meta['period'] = f"{df['time'].dt.year.min()}-{df['time'].dt.year.max()}"
    return df, meta


_HAR_LAT2D = None
_HAR_LON2D = None


def _harv2_index(lon, lat, cache_root):
    global _HAR_LAT2D, _HAR_LON2D
    if _HAR_LAT2D is None:
        with xr.open_dataset(_resolve('harv2_temp', cache_root),
                             engine='netcdf4') as ds:
            _HAR_LAT2D = ds['lat'].values.astype(np.float64)
            _HAR_LON2D = ds['lon'].values.astype(np.float64)
    dist = (_HAR_LAT2D - lat) ** 2 + (_HAR_LON2D - lon) ** 2
    yi, xi = np.unravel_index(np.argmin(dist), dist.shape)
    return int(yi), int(xi)


def _load_harv2(lon, lat, cache_root):
    """HARv2 10 km WRF Lambert grid 1980-2025: t2 [K/C], prcp [mm/month]."""
    temp_path = _resolve('harv2_temp', cache_root)
    precip_path = _resolve('harv2_precip', cache_root)
    yi, xi = _harv2_index(lon, lat, cache_root)
    with xr.open_dataset(temp_path, engine='netcdf4') as ds:
        t_var = next((v for v in ['t2', 'T2', 'temp', 't2m']
                      if v in ds.data_vars), list(ds.data_vars)[0])
        t_units = ds[t_var].attrs.get('units', '')
        temp_vals = ds[t_var].isel(south_north=yi, west_east=xi).values
        time_vals = ds['time'].values
    with xr.open_dataset(precip_path, engine='netcdf4') as ds:
        p_var = next((v for v in ['prcp', 'PRCP', 'precipitation', 'pr']
                      if v in ds.data_vars), list(ds.data_vars)[0])
        p_unit = ds[p_var].attrs.get('units', 'mm')
        prcp_vals = ds[p_var].isel(south_north=yi, west_east=xi).values
    temp_in_kelvin = (np.nanmean(temp_vals) > 100) or ('K' in t_units)
    meta = {'forcing': 'HARv2', 'resolution': '10km (Lambert WRF)',
            'grid_indices': (yi, xi),
            'nearest_lat': float(_HAR_LAT2D[yi, xi]),
            'nearest_lon': float(_HAR_LON2D[yi, xi]),
            'temp_var': t_var, 'prcp_var': p_var,
            'files': [str(temp_path), str(precip_path)]}
    df = _to_monthly_df(time_vals, temp_vals, prcp_vals, temp_in_kelvin,
                        p_unit, meta)
    meta['period'] = f"{df['time'].dt.year.min()}-{df['time'].dt.year.max()}"
    return df, meta


# ─────────────────────────────────────────────────────────────────────────────
# GRID-CELL OROGRAPHY  (NEW v1.2)
# ─────────────────────────────────────────────────────────────────────────────

_OROG_VARS = ['z', 'orog', 'elev', 'elv', 'hgt', 'HGT', 'height',
              'altitude', 'surface_geopotential', 'oro', 'topo']
_ELEV_MIN, _ELEV_MAX = -100.0, 8900.0
_OROG_CACHE = {}


def find_orography_file(forcing: str, cache_root: Path) -> Path:
    for rel in OROG_PATHS.get(forcing, []):
        p = Path(rel)
        p = p if p.is_absolute() else Path(cache_root) / rel
        if p.exists():
            return p
    tried = "\n".join(f"  {Path(cache_root) / r}" for r in OROG_PATHS.get(forcing, []))
    raise FileNotFoundError(
        f"No orography file for {forcing}. Tried:\n{tried}\n"
        "Download the grid-cell elevation for this product (same grid as the "
        "climate data) and add its path to OROG_PATHS. Do NOT fall back to "
        "glacier elevation.")


def _squeeze_non_spatial(da, keep: set):
    for d in list(da.dims):
        if d not in keep:
            da = da.isel({d: 0})
    return da


def _unit_scale(da, var, units) -> float:
    """Conversion to metres, decided once per file."""
    u = (units or '').lower().replace(' ', '')
    if var in ('z', 'surface_geopotential') or 'm**2' in u or 'm2s-2' in u \
            or 'm^2' in u:
        return 1.0 / G
    if u == 'km':
        return 1000.0
    sub = da
    for d in da.dims:
        n = da.sizes[d]
        if n > 200:
            sub = sub.isel({d: slice(None, None, max(1, n // 200))})
    vmax = float(np.nanmax(np.abs(sub.values)))
    if 0 < vmax < 10.0:
        return 1000.0                 # km without units attribute
    if vmax > 20000.0:
        return 1.0 / G                # geopotential without units attribute
    return 1.0


def grid_elevation(forcing: str, lon: float, lat: float,
                   cache_root: Path = None) -> dict:
    """
    Elevation (m a.s.l.) of the nearest grid cell of `forcing` — the height
    at which load_climate()'s temperature is valid. Use it as OGGM ref_hgt.

    Returns dict(grid_elev_m, orog_file, orog_lat, orog_lon)
    """
    if cache_root is None:
        cache_root = detect_cache_root()
    path = find_orography_file(forcing, cache_root)
    key = str(path)

    if key not in _OROG_CACHE:
        ds = xr.open_dataset(path)
        var = next((v for v in _OROG_VARS if v in ds.data_vars),
                   list(ds.data_vars)[0])
        units = ds[var].attrs.get('units', '')
        info = {'ds': ds, 'var': var, 'units': units,
                'scale': _unit_scale(ds[var], var, units)}
        lat_d = next((d for d in ['latitude', 'lat'] if d in ds.dims), None)
        lon_d = next((d for d in ['longitude', 'lon'] if d in ds.dims), None)
        if lat_d and lon_d:
            info.update(kind='1d', lat=ds[lat_d].values, lon=ds[lon_d].values,
                        lat_d=lat_d, lon_d=lon_d)
        elif 'points' in ds.dims:
            info.update(kind='points', lat=ds['latitude'].values,
                        lon=ds['longitude'].values)
        else:
            lat_v = next(v for v in ['lat', 'XLAT', 'latitude'] if v in ds)
            lon_v = next(v for v in ['lon', 'XLONG', 'longitude'] if v in ds)
            ydim, xdim = [d for d in ds[var].dims if d in ds[lat_v].dims][-2:]
            la = _squeeze_non_spatial(ds[lat_v], {ydim, xdim})
            lo = _squeeze_non_spatial(ds[lon_v], {ydim, xdim})
            info.update(kind='2d', lat=la.values.astype(np.float64),
                        lon=lo.values.astype(np.float64), ydim=ydim, xdim=xdim)
        _OROG_CACHE[key] = info

    info = _OROG_CACHE[key]
    ds, var = info['ds'], info['var']
    W = 2      # search window (cells) if the nearest cell is NaN (masked)

    def _collapse(da, spatial):
        """Constant field: collapse time/expver/number dims with nanmax
        (v1.2.0 took index 0, which is NaN in some CDS files)."""
        other = [d for d in da.dims if d not in spatial]
        v = da.values.astype(float)
        if other:
            axes = tuple(da.dims.index(d) for d in other)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                v = np.nanmax(v, axis=axes)
        return v

    note = ''
    if info['kind'] == '1d':
        lats, lons = info['lat'], info['lon']
        lon_q = lon + 360.0 if (lons.max() > 180 and lon < 0) else lon
        yi = int(np.argmin(np.abs(lats - lat)))
        xi = int(np.argmin(np.abs(lons - lon_q)))
        ys = slice(max(0, yi - W), yi + W + 1)
        xs = slice(max(0, xi - W), xi + W + 1)
        sub = _collapse(ds[var].isel({info['lat_d']: ys, info['lon_d']: xs}),
                        {info['lat_d'], info['lon_d']})
        if ds[var].dims.index(info['lat_d']) > ds[var].dims.index(info['lon_d']):
            sub = sub.T
        cy, cx = yi - ys.start, xi - xs.start
        glat, glon = np.meshgrid(lats[ys], lons[xs], indexing='ij')
    elif info['kind'] == 'points':
        lats, lons = info['lat'], info['lon']
        lon_q = lon + 360.0 if (lons.max() > 180 and lon < 0) else lon
        d2 = (lats - lat) ** 2 + (lons - lon_q) ** 2
        order = np.argsort(d2)[:25]
        vals = np.array([float(np.nanmax(np.atleast_1d(
            _collapse(ds[var].isel(points=int(i)), set())))) for i in order])
        k = 0 if np.isfinite(vals[0]) else int(np.argmax(np.isfinite(vals)))
        if k:
            note = f'nearest point NaN, used point {k} of 25 nearest'
        elev = vals[k] * info['scale']
        olat, olon = float(lats[order[k]]), float(lons[order[k]])
        sub = None
    else:
        dist = (info['lat'] - lat) ** 2 + (info['lon'] - lon) ** 2
        yi, xi = np.unravel_index(np.argmin(dist), dist.shape)
        ys = slice(max(0, yi - W), yi + W + 1)
        xs = slice(max(0, xi - W), xi + W + 1)
        sub = _collapse(ds[var].isel({info['ydim']: ys, info['xdim']: xs}),
                        {info['ydim'], info['xdim']})
        if ds[var].dims.index(info['ydim']) > ds[var].dims.index(info['xdim']):
            sub = sub.T
        cy, cx = yi - ys.start, xi - xs.start
        glat, glon = info['lat'][ys, xs], info['lon'][ys, xs]

    if sub is not None:
        if np.isfinite(sub[cy, cx]):
            ry, rx = cy, cx
        else:
            fin = np.isfinite(sub)
            if not fin.any():
                raise ValueError(f"Orography NaN in a {2 * W + 1}x{2 * W + 1} "
                                 f"window for {forcing} at ({lon:.3f},{lat:.3f})"
                                 f" in {path.name}")
            iy, ix = np.nonzero(fin)
            k = int(np.argmin((iy - cy) ** 2 + (ix - cx) ** 2))
            ry, rx = int(iy[k]), int(ix[k])
            note = f'nearest cell NaN, used neighbour offset ({ry - cy},{rx - cx})'
        elev = float(sub[ry, rx]) * info['scale']
        olat, olon = float(glat[ry, rx]), float(glon[ry, rx])

    if not np.isfinite(elev) or not (_ELEV_MIN <= elev <= _ELEV_MAX):
        raise ValueError(
            f"Implausible grid elevation {elev:.1f} m for {forcing} at "
            f"({lon:.3f},{lat:.3f}) from {path.name} [var={var}, "
            f"units='{info['units']}']")
    return {'grid_elev_m': elev, 'orog_file': str(path),
            'orog_lat': olat, 'orog_lon': olon, 'orog_note': note}


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

FORCING_NAMES = ['ERA5', 'ERA5-Land', 'CRU_TS4', 'CHELSA', 'HARv2']

_LOADERS = {
    'ERA5': _load_era5, 'ERA5-Land': _load_era5land, 'CRU_TS4': _load_cru,
    'CHELSA': _load_chelsa, 'HARv2': _load_harv2,
}


def load_climate(forcing: str, lon: float, lat: float,
                 cache_root: Path = None, start_year: int = None,
                 end_year: int = None) -> tuple:
    """
    Monthly climate at the nearest grid cell.
    Returns (df[time, temp_C, prcp_mm], meta). The series is gap-free.
    """
    forcing = forcing.strip()
    if forcing not in _LOADERS:
        raise ValueError(f"Unknown forcing '{forcing}'. Choose from: {FORCING_NAMES}")
    if cache_root is None:
        cache_root = detect_cache_root()
    df, meta = _LOADERS[forcing](lon, lat, cache_root)
    if start_year or end_year:
        df = _trim_years(df, start_year, end_year)
    meta['n_months'] = len(df)
    return df, meta


def write_oggm_climate_file(df: pd.DataFrame, gdir, meta: dict,
                            ref_hgt: float) -> None:
    """
    Write climate_historical.nc for OGGM: complete CALENDAR years only
    (OGGM 1.6 requirement), gap-free monthly axis asserted.

    ref_hgt : elevation of the CLIMATE GRID CELL [m] — use grid_elevation().
              (Not the glacier elevation.)
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    first_jan = df.loc[df['time'].dt.month == 1, 'time'].min()
    last_dec = df.loc[df['time'].dt.month == 12, 'time'].max()
    if pd.isna(first_jan) or pd.isna(last_dec) or last_dec < first_jan:
        raise ValueError(f"No complete calendar year for "
                         f"{meta.get('forcing', '?')}")
    df = df[(df['time'] >= first_jan) & (df['time'] <= last_dec)]
    df = df.reset_index(drop=True)

    expected = pd.date_range(first_jan, last_dec, freq='MS')
    if len(df) != len(expected) or not (df['time'].values == expected.values).all():
        raise ValueError("Monthly axis has gaps/duplicates — would shift months")
    if df[['temp_C', 'prcp_mm']].isna().any().any():
        raise ValueError("NaN in climate series")

    time_index = pd.DatetimeIndex(df['time'])
    ds = xr.Dataset(
        {'temp': (['time'], df['temp_C'].values.astype(np.float32),
                  {'units': 'degC', 'long_name': '2m temperature'}),
         'prcp': (['time'], df['prcp_mm'].values.astype(np.float32),
                  {'units': 'mm month-1',
                   'long_name': 'Monthly precipitation total'})},
        coords={'time': time_index},
        attrs={'ref_hgt': float(ref_hgt),
               'climate_source': meta.get('forcing', 'unknown'),
               'resolution': meta.get('resolution', ''),
               'period': meta.get('period', ''),
               'hydro_yr_0': int(time_index.year.min()),
               'hydro_yr_1': int(time_index.year.max())})
    climate_file = gdir.get_filepath('climate_historical')
    if os.path.exists(climate_file):
        os.remove(climate_file)
    ds.to_netcdf(climate_file)
