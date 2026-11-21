#!/usr/bin/env python
"""Modelled 1975-1999 specific mass balance per sub-region and forcing (area-weighted, RGI area), for the comparison with
published KH-9/Corona geodetic balances. Glaciers with a value in all years 1975-1999 (ERA5 family and CRU_TS4 have 1975-1999;
CHELSA has 1980 on, shown for 1980-1999 only). Output tables/kh9_model_side.csv (m w.e. yr-1).  Env oggm."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from paths import PER_FORCING
LK = pd.read_csv(ROOT / "analysis/glacier_lookup.csv").set_index("rgi_id")
rows = []
for f, (y0, y1) in {"ERA5": (1975, 1999), "ERA5-Land": (1975, 1999), "CRU_TS4": (1975, 1999), "HARv2": (1980, 1999), "CHELSA": (1980, 1999)}.items():
    d = pd.read_parquet(PER_FORCING / f / "yearly.parquet", columns=["rgi_id", "year", "mb_annual_mwe"])
    d = d[d.year.between(y0, y1)]
    g = d.groupby("rgi_id").mb_annual_mwe.agg(["mean", "count"]); g = g[g["count"] == y1 - y0 + 1]
    g["area"] = LK.area_rgi_km2.reindex(g.index); g["sub"] = LK.subregion.reindex(g.index); g["reg"] = LK.rgi_region.reindex(g.index)
    for key, grp in list(g.groupby("sub")) + [("ALL_HMA", g)]:
        rows.append(dict(forcing=f, window=f"{y0}-{y1}", subregion=key, n=len(grp), area_km2=grp.area.sum(), B=np.average(grp["mean"], weights=grp.area)))
T = pd.DataFrame(rows); T.to_csv(ROOT / "tables/kh9_model_side.csv", index=False)
print(T.pivot_table(index="subregion", columns="forcing", values="B").round(3).to_string())
print(T[T.forcing == "ERA5"][["subregion", "n", "area_km2"]].round(0).to_string(index=False))

# ---- areas matching published values (boxes defined by us from the site descriptions; degrees lon_min, lon_max, lat_min, lat_max)
BOX = {"Himalaya 75-93E (Maurer 2019 transect)": ("sub", ["West Himalaya", "Central Himalaya", "East Himalaya"], (75, 93, 26, 36)),
       "Karakoram sub-region (Zhou 2017)": ("sub", ["Karakoram"], None),
       "Poiqu (Bhattacharya 2021)": ("box", None, (85.3, 86.2, 27.8, 28.5)),
       "Purogangri (Bhattacharya 2021)": ("box", None, (88.5, 89.7, 33.5, 34.4)),
       "Ak-Shirak (Bhattacharya 2021)": ("box", None, (77.8, 78.8, 41.5, 42.2)),
       "W Nyainqentanglha (Bhattacharya 2021)": ("box", None, (90.0, 91.2, 30.0, 30.8))}
out = []
for f, (y0, y1) in {"ERA5": (1975, 1999), "ERA5-Land": (1975, 1999), "CRU_TS4": (1975, 1999), "HARv2": (1980, 1999), "CHELSA": (1980, 1999)}.items():
    d = pd.read_parquet(PER_FORCING / f / "yearly.parquet", columns=["rgi_id", "year", "mb_annual_mwe"]); d = d[d.year.between(y0, y1)]
    g = d.groupby("rgi_id").mb_annual_mwe.agg(["mean", "count"]); g = g[g["count"] == y1 - y0 + 1]
    L = LK.reindex(g.index); g["area"] = L.area_rgi_km2; g["lon"] = L.CenLon; g["lat"] = L.CenLat; g["sub"] = L.subregion
    for name, (kind, subs, b) in BOX.items():
        m = g["sub"].isin(subs) if subs else pd.Series(True, index=g.index)
        if b: m &= g.lon.between(b[0], b[1]) & g.lat.between(b[2], b[3])
        h = g[m]
        out.append(dict(area=name, forcing=f, window=f"{y0}-{y1}", n=len(h), area_km2=h.area.sum(), B=np.average(h["mean"], weights=h.area) if len(h) else np.nan))
O = pd.DataFrame(out); O.to_csv(ROOT / "tables/kh9_model_vs_published.csv", index=False)
print(O.pivot_table(index="area", columns="forcing", values="B").round(3).to_string()); print(O[O.forcing == "ERA5"][["area", "n", "area_km2"]].round(0).to_string(index=False))
