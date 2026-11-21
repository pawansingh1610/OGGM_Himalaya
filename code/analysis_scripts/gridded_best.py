#!/usr/bin/env python
"""1 deg x 1 deg gridded mass balance and volume change of the HMA glaciers from the best-estimate ensemble.

Env `oggm`. Reads the run_06 per-glacier files (scripts/paths.py) and analysis/glacier_lookup.csv, writes
tables/gridded_{mb,vol}_<window>.csv (read by scripts/figures/fig16_gridded.py).

Ensemble (members covering every year of the window only, as in Table 7): the member weights of scripts/best_mb_ensemble.py (scheme and scores from tables/best_mb_weights.csv,
`member_weights` of that script), applied glacier by glacier and year by year to the members that have a valid
value for that glacier-year (renormalised, family-corrected). No single-best-forcing composite.
  mb   best-estimate B per glacier-year; cell value = RGI-area-weighted mean over the glaciers of the cell of the
       window mean; plus the per-forcing cell means (for the sign test and the per-forcing supplement map).
  vol  dV = sum_f w_f dm_gt_f / 0.9 per glacier-year (km3 ice), summed over the window; cell value in km3 and in
       % of the cell's consensus volume at the end of 1999 (V_1999).
A glacier enters a cell value only if every year of the window has a value from at least one member.
Windows: mb 1980-1999 and 2000-2019 (all five forcings), vol 2000-2019. The 1950-1979 volume maps use only
ERA5, ERA5-Land, CRU_TS4 (no weights: one map per forcing); glaciers with backward_vanished == True in any of the
three are excluded and counted.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paths import PER_FORCING  # noqa: E402
from best_mb_ensemble import member_weights, FORC  # noqa: E402

TAB = ROOT / "tables"
q = pd.read_csv(TAB / "best_mb_weights.csv", index_col=0).score
SCHEME = pd.read_csv(TAB / "best_mb_weights.csv", index_col=0).chosen_scheme.iloc[0]
LK = pd.read_csv(ROOT / "analysis" / "glacier_lookup.csv").set_index("rgi_id")
print("scheme", SCHEME, "scores", q.round(3).to_dict())


def load(f, y0, y1):
    d = pd.read_parquet(PER_FORCING / f / "yearly.parquet", columns=["rgi_id", "year", "mb_annual_mwe", "dm_gt", "volume_km3"])
    return d[(d.year >= y0) & (d.year <= y1)]


def stack(y0, y1, col, forcs):
    ids = LK.index
    yrs = np.arange(y0, y1 + 1)
    A = np.full((len(forcs), len(ids), len(yrs)), np.nan, dtype="float32")
    for i, f in enumerate(forcs):
        d = load(f, y0, y1)
        r = ids.get_indexer(d.rgi_id)
        k = d.year.values - y0
        m = r >= 0
        A[i, r[m], k[m]] = d[col].values[m]
        print("  loaded", f, col, y0, y1, flush=True)
    return A, ids, yrs


def best(A, forcs):
    """weighted combination over axis 0 with weights depending on which members are present (per glacier-year)."""
    ok = ~np.isnan(A)
    code = (ok * (1 << np.arange(len(forcs)))[:, None, None]).sum(0)
    out = np.full(A.shape[1:], np.nan, dtype="float64")
    Z = np.nan_to_num(A, nan=0.0).astype("float64")
    for c in np.unique(code):
        if c == 0:
            continue
        pres = [forcs[i] for i in range(len(forcs)) if c >> i & 1]
        w = member_weights(q, pres, SCHEME)
        sel = code == c
        out[sel] = sum(w[f] * Z[forcs.index(f)][sel] for f in w.index)
    return out


FIRST = {"ERA5": 1950, "ERA5-Land": 1950, "CRU_TS4": 1950, "CHELSA": 1979, "HARv2": 1980}
LAST = {"ERA5": 2025, "ERA5-Land": 2025, "HARv2": 2025, "CRU_TS4": 2019, "CHELSA": 2016}


def covering(y0, y1, forcs):
    """members that cover every year of the window (the rule of Table 7: a member is used for a period only if it covers all of it)."""
    return [f for f in forcs if FIRST[f] <= y0 and LAST[f] >= y1]


def cell_ids(ids):
    lon, lat = LK.loc[ids, "CenLon"].values, LK.loc[ids, "CenLat"].values
    return np.floor(lon).astype(int), np.floor(lat).astype(int)


def window_mean(X):
    """mean over years; NaN if any year missing."""
    return np.where(np.isnan(X).any(-1), np.nan, X.mean(-1))


def write(df, name):
    p = TAB / name
    df.to_csv(p, index=False)
    print("  ->", p.relative_to(ROOT), len(df), "cells")


def mb_window(y0, y1, tag):
    A, ids, _ = stack(y0, y1, "mb_annual_mwe", FORC)
    use = covering(y0, y1, FORC)
    print("  ensemble members for", tag, use)
    B = best(A[[FORC.index(f) for f in use]], use)
    mb_best = window_mean(B)
    per_f = {f: window_mean(A[i]) for i, f in enumerate(FORC)}
    area = LK.loc[ids, "area_rgi_km2"].values
    lon, lat = cell_ids(ids)
    df = pd.DataFrame({"lon": lon, "lat": lat, "area": area, "best": mb_best, **{f: per_f[f] for f in FORC}})
    rows = []
    for (lo, la), g in df.groupby(["lon", "lat"]):
        r = {"lon": lo, "lat": la, "n_glaciers": len(g), "area_km2": g.area.sum()}
        ok = g.best.notna()
        w = g.area[ok]
        r["n_used"], r["area_used_km2"] = int(ok.sum()), w.sum()
        r["B_best"] = np.average(g.best[ok], weights=w) if ok.any() else np.nan
        allv = g[FORC].notna().all(axis=1)       # glaciers with a value from every member, for the sign test
        for f in FORC:
            r["B_" + f] = np.average(g[f][allv], weights=g.area[allv]) if allv.any() else np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    out["n_pos"] = (out[["B_" + f for f in FORC]] > 0).sum(1)
    out["sign_disagree"] = (out.n_pos > 0) & (out.n_pos < out[["B_" + f for f in FORC]].notna().sum(1))
    write(out, f"gridded_mb_{tag}.csv")
    ok = ~np.isnan(mb_best)
    print(f"  HMA best B {tag}: {np.average(mb_best[ok], weights=area[ok]):+.3f} m w.e. yr-1 on {ok.sum()} glaciers "
          f"({area[ok].sum():.0f} of {area.sum():.0f} km2)")


def vol_window(y0, y1, tag, forcs, ensemble=True):
    A, ids, _ = stack(y0, y1, "dm_gt", forcs)
    dV = A / 0.9                                    # km3 ice per glacier-year (dm_gt in Gt, 0.9 Gt km-3)
    v99 = load("ERA5", 1999, 1999).set_index("rgi_id").volume_km3.reindex(ids).values
    lon, lat = cell_ids(ids)
    area = LK.loc[ids, "area_rgi_km2"].values
    cols = {}
    if ensemble:
        use = covering(y0, y1, forcs)
        print("  ensemble members for", tag, use)
        D = best(dV[[forcs.index(f) for f in use]], use)
        cols["best"] = np.where(np.isnan(D).any(-1), np.nan, D.sum(-1))
    for i, f in enumerate(forcs):
        cols[f] = np.where(np.isnan(dV[i]).any(-1), np.nan, dV[i].sum(-1))
    df = pd.DataFrame({"lon": lon, "lat": lat, "area": area, "v99": v99, **cols})
    if not ensemble:                                 # backward-vanished glaciers (V undefined before 2000) are excluded
        bv = np.zeros(len(ids), bool)
        for f in forcs:
            m = pd.read_csv(PER_FORCING / f / "metadata.csv", usecols=["rgi_id", "backward_vanished"]).set_index("rgi_id")
            bv |= m.backward_vanished.reindex(ids).fillna(False).astype(bool).values
        print(f"  backward_vanished in any of {forcs}: {bv.sum()} glaciers, {area[bv].sum():.1f} km2 (excluded)")
        df.loc[bv, forcs] = np.nan
    rows = []
    for (lo, la), g in df.groupby(["lon", "lat"]):
        r = {"lon": lo, "lat": la, "n_glaciers": len(g), "area_km2": g.area.sum()}
        for c in cols:
            ok = g[c].notna() & g.v99.notna()
            r["dV_" + c] = g[c][ok].sum()
            r["V1999_" + c] = g.v99[ok].sum()
            r["n_" + c] = int(ok.sum())
        rows.append(r)
    out = pd.DataFrame(rows)
    write(out, f"gridded_vol_{tag}.csv")
    for c in cols:
        print(f"  HMA dV {tag} {c}: {out['dV_' + c].sum():+.1f} km3 = {100 * out['dV_' + c].sum() / out['V1999_' + c].sum():+.2f} % of V1999 "
              f"({out['n_' + c].sum()} glaciers)")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "mb":
        mb_window(2000, 2019, "2000_2019")
        mb_window(1980, 1999, "1980_1999")
    elif what == "vol":
        vol_window(2000, 2019, "2000_2019", FORC)
    elif what == "vol2025":      # ERA5 family only (the only members after 2019); best = weighted over these three
        vol_window(2000, 2025, "2000_2025", ["ERA5", "ERA5-Land", "HARv2"])
        vol_window(2020, 2025, "2020_2025", ["ERA5", "ERA5-Land", "HARv2"])
    elif what == "vol1950":
        vol_window(1950, 1979, "1950_1979", ["ERA5", "ERA5-Land", "CRU_TS4"], ensemble=False)
