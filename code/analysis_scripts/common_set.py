#!/usr/bin/env python
"""Forcing spread on a glacier set common to all forcings that cover a window.

For each window, the glaciers kept are those with a value (dm_gt and mb_annual_mwe) in every year of the window for EVERY
forcing that covers the window, and that are not flagged backward_vanished in any of them. For these glaciers each forcing's
volume change (sum dm_gt / 0.9, km3 ice) and area-weighted mass balance is computed, with the % of the common set's V_1999.
Output: tables/common_set_spread.csv (the Supplement table tables/S14_common_set.tex is made from it by common_set_table.py); the same numbers on each forcing's own glacier set come from analysis/C_volume_change_windows.csv.
Env `oggm`.  Usage: python scripts/common_set.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paths import PER_FORCING  # noqa: E402

LK = pd.read_csv(ROOT / "analysis" / "glacier_lookup.csv").set_index("rgi_id")
ids = LK.index
WIN = {"1950-1979": ["ERA5", "ERA5-Land", "CRU_TS4"],
       "1980-1999": ["ERA5", "ERA5-Land", "HARv2", "CRU_TS4", "CHELSA"],
       "2000-2019": ["ERA5", "ERA5-Land", "HARv2", "CRU_TS4"],
       "2000-2016": ["ERA5", "ERA5-Land", "HARv2", "CRU_TS4", "CHELSA"]}
area = LK.area_rgi_km2.values


def load(f, y0, y1):
    d = pd.read_parquet(PER_FORCING / f / "yearly.parquet", columns=["rgi_id", "year", "mb_annual_mwe", "dm_gt", "volume_km3"])
    d = d[(d.year >= y0) & (d.year <= y1)]
    r = ids.get_indexer(d.rgi_id)
    k = d.year.values - y0
    ok = r >= 0
    out = {}
    for c in ("mb_annual_mwe", "dm_gt"):
        A = np.full((len(ids), y1 - y0 + 1), np.nan)
        A[r[ok], k[ok]] = d[c].values[ok]
        out[c] = A
    return out


v99 = pd.read_parquet(PER_FORCING / "ERA5" / "yearly.parquet", columns=["rgi_id", "year", "volume_km3"])
v99 = v99[v99.year == 1999].set_index("rgi_id").volume_km3.reindex(ids).values
rows = []
for w, forcs in WIN.items():
    y0, y1 = map(int, w.split("-"))
    D = {f: load(f, y0, y1) for f in forcs}
    valid = np.ones(len(ids), bool)
    for f in forcs:
        valid &= ~np.isnan(D[f]["dm_gt"]).any(1) & ~np.isnan(D[f]["mb_annual_mwe"]).any(1)
        m = pd.read_csv(PER_FORCING / f / "metadata.csv", usecols=["rgi_id", "backward_vanished"]).set_index("rgi_id")
        valid &= ~m.backward_vanished.reindex(ids).fillna(False).astype(bool).values
    valid &= ~np.isnan(v99)
    V = v99[valid].sum()
    print(f"{w}: {valid.sum()} common glaciers, {area[valid].sum():.0f} of {area.sum():.0f} km2, V1999 {V:.0f} km3", flush=True)
    for f in forcs:
        dV = (D[f]["dm_gt"][valid].sum(1) / 0.9).sum()
        B = np.average(D[f]["mb_annual_mwe"][valid].mean(1), weights=area[valid])
        rows.append(dict(window=w, forcing=f, n_glaciers=int(valid.sum()), area_km2=area[valid].sum(), V1999_km3=V,
                         dV_km3=dV, dV_pct_V1999=100 * dV / V, B_mwe=B))
T = pd.DataFrame(rows)
T.to_csv(ROOT / "tables" / "common_set_spread.csv", index=False)
print(T.round(3).to_string(index=False))
for w, g in T.groupby("window"):
    print(f"{w}: range of dV % {g.dV_pct_V1999.min():+.2f} .. {g.dV_pct_V1999.max():+.2f} (spread {g.dV_pct_V1999.max() - g.dV_pct_V1999.min():.2f} pp); B {g.B_mwe.min():+.3f} .. {g.B_mwe.max():+.3f}")
