#!/usr/bin/env python
"""Sensitivity of the main results to RGI surge-type glaciers (RGI 6.0 'Surging' >= 1: possible, probable, observed).

Rebuilds the regional series exactly as 06_ANALYSIS_v3.regional_series (in_sample = calib ok and not backward-vanished;
sums of dm_gt, area_start, volume per year) for the full sample and for the sample without surge-type glaciers, then
  - volume change per window and forcing (km3 ice, % of the sample's own V_1999 and of the full-sample V_1999),
  - period-mean B and the equal-family best estimate (members covering every year; as best_mb_ensemble.aggregate),
  - KH-9 rows (as kh9_compare.py) with surge-type glaciers removed from each sample.
The full-sample numbers are checked against analysis/C_volume_change_windows.csv and tables/best_mb_periods.csv.
Output: context/reports/external2_round3/surge_*.csv. Env oggm. Read-only on External2."""
import sys, io, contextlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from paths import PER_FORCING
OUT = ROOT / "context/reports/external2_round3"; OUT.mkdir(parents=True, exist_ok=True)
FORC = ["ERA5", "ERA5-Land", "HARv2", "CRU_TS4", "CHELSA"]
FAM = {"ERA5": "E", "ERA5-Land": "E", "HARv2": "E", "CRU_TS4": "C", "CHELSA": "C"}
LK = pd.read_csv(ROOT / "analysis/glacier_lookup.csv")
SURGE = set(LK.loc[LK.SurgeType.isin([1, 2, 3]), "rgi_id"])
WIN = {"1980-1999": (1980, 1999), "2000-2016": (2000, 2016), "2000-2019": (2000, 2019), "1950-1979": (1950, 1979),
       "2000-2025": (2000, 2025)}
PER = {"1980-1999": (1980, 1999), "2000-2019": (2000, 2019), "1950-1979": (1950, 1979), "2020-2023": (2020, 2023)}


def series(f, drop):
    m = pd.read_csv(PER_FORCING / f / "metadata.csv", usecols=["rgi_id", "calib_status", "backward_vanished"])
    bv = m.backward_vanished.astype(str).str.lower().isin(["true", "1"])
    keep = set(m.loc[(m.calib_status == "ok") & ~bv, "rgi_id"]) - drop
    y = pd.read_parquet(PER_FORCING / f / "yearly.parquet", columns=["rgi_id", "year", "dm_gt", "area_start_km2", "volume_km3"])
    y = y[y.rgi_id.isin(keep)].merge(LK[["rgi_id", "rgi_region"]], on="rgi_id")
    out = []
    for unit, g in [("ALL_HMA", y)] + list(y.groupby("rgi_region")):
        s = g.groupby("year").agg(dm=("dm_gt", "sum"), A=("area_start_km2", "sum"), V=("volume_km3", "sum"))
        s["B"] = np.where(s.A > 0, s.dm / s.A * 1e3, np.nan); s["unit"] = unit; s["forcing"] = f
        out.append(s.reset_index())
    return pd.concat(out), len(keep)


def best_eq_family(means):
    """equal family weights over the members present (dict forcing -> value)."""
    fams = {}
    for f in means: fams.setdefault(FAM[f], []).append(f)
    w = {f: 1 / len(fams) / len(fams[FAM[f]]) for f in means}
    return sum(w[f] * means[f] for f in means)


rows, prow, nrow = [], [], []
S = {}
for case, drop in [("all", set()), ("no_surge", SURGE)]:
    for f in FORC:
        s, n = series(f, drop); S[(case, f)] = s; nrow.append(dict(case=case, forcing=f, n_glaciers=n))
for case in ["all", "no_surge"]:
    for f in FORC:
        for unit, g in S[(case, f)].groupby("unit"):
            gi = g.set_index("year"); cov = gi.B.dropna().index
            v99 = gi.loc[1999, "V"]; v99_all = S[("all", f)].query("unit == @unit").set_index("year").loc[1999, "V"]
            for w, (a, b) in WIN.items():
                if a < cov.min() or b > cov.max(): continue
                dv = gi.loc[b, "V"] - gi.loc[a - 1, "V"]
                rows.append(dict(case=case, forcing=f, unit=unit, window=w, dV_km3=dv, V1999=v99, pct_own_V1999=100 * dv / v99,
                                 pct_full_V1999=100 * dv / v99_all))
            for p, (a, b) in PER.items():
                if a < cov.min() or b > cov.max(): continue
                prow.append(dict(case=case, forcing=f, unit=unit, period=p, B=gi.loc[a:b, "B"].mean()))
W = pd.DataFrame(rows); P = pd.DataFrame(prow)
best = []
for (case, unit, p), g in P.groupby(["case", "unit", "period"]):
    best.append(dict(case=case, unit=unit, period=p, forcing="BEST_eq_family", B=best_eq_family(dict(zip(g.forcing, g.B)))))
P = pd.concat([P, pd.DataFrame(best)])
W.to_csv(OUT / "surge_volume_windows.csv", index=False); P.to_csv(OUT / "surge_period_B.csv", index=False)
pd.DataFrame(nrow).to_csv(OUT / "surge_sample_sizes.csv", index=False)

# ---- checks against the paper's tables
C = pd.read_csv(ROOT / "analysis/C_volume_change_windows.csv"); C = C[C.level == "hma"]
cw = {"1980-1999": "recon_1980-1999", "2000-2019": "fwd_2000-2019", "2000-2016": "fwd_2000-2016", "1950-1979": "recon_1950-1979", "2000-2025": "fwd_2000-2025"}
chk = W[(W.case == "all") & (W.unit == "ALL_HMA")].copy()
chk["paper_dV"] = [C[(C.forcing == r.forcing) & (C.window == cw[r.window])].dV_km3.squeeze() if len(C[(C.forcing == r.forcing) & (C.window == cw[r.window])]) else np.nan for r in chk.itertuples()]
print("check vs C_volume_change_windows (max |diff| km3):", float((chk.dV_km3 - chk.paper_dV).abs().max()))
BP = pd.read_csv(ROOT / "tables/best_mb_periods.csv"); BP = BP[BP.level == "hma"].set_index("period").B_best
for p in ["1980-1999", "2000-2019", "1950-1979"]:
    v = P[(P.case == "all") & (P.unit == "ALL_HMA") & (P.period == p) & (P.forcing == "BEST_eq_family")].B.squeeze()
    print(f"check best {p}: rebuilt {v:.4f} paper {BP[p]:.4f}")

pd.set_option("display.width", 250)
for unit in ["ALL_HMA", "RGI60-13", "RGI60-14", "RGI60-15"]:
    t = W[W.unit == unit].pivot_table(index=["window", "forcing"], columns="case", values=["dV_km3", "pct_own_V1999"])
    print(f"\n== {unit} volume change\n", t.round(2).to_string())
    q = P[P.unit == unit].pivot_table(index=["period", "forcing"], columns="case", values="B")
    q["diff"] = q.no_surge - q["all"]
    print(f"\n== {unit} period-mean B (m w.e./yr)\n", q.round(3).to_string())

# ---- KH-9 rows without surge-type glaciers (same construction as kh9_compare.py)
with contextlib.redirect_stdout(io.StringIO()):
    from gridded_best import stack, best as gbest, LK as GLK, FORC as GF
A, ids, yrs = stack(1968, 2003, "mb_annual_mwe", GF)
area = GLK.loc[ids, "area_rgi_km2"].values; lon = GLK.loc[ids, "CenLon"].values; lat = GLK.loc[ids, "CenLat"].values; sub = GLK.loc[ids, "subregion"].values
Bb = gbest(A, GF); sg = np.isin(ids, list(SURGE))
him = np.isin(sub, ["West Himalaya", "Central Himalaya", "East Himalaya"]) & (lon >= 75) & (lon <= 93)
top = np.zeros(len(ids), bool); top[np.argsort(-np.where(him, area, -1))[:650]] = True
box = lambda a: (lon >= a[0]) & (lon <= a[1]) & (lat >= a[2]) & (lat <= a[3])
SITES = [("Himalaya 650 largest", -0.22, 0.13, 1975, 2000, top), ("Himalaya all", -0.22, 0.13, 1975, 2000, him),
         ("Karakoram (Zhou)", -0.09, 0.03, 1973, 2000, box((74.33, 78.25, 33.75, 37.5))),
         ("Purogangri", -0.22, 0.07, 1975, 2000, box((88.5, 89.7, 33.5, 34.4))), ("Poiqu", -0.30, 0.10, 1974, 2004, box((85.3, 86.2, 27.8, 28.5))),
         ("W Nyainqentanglha", -0.26, 0.13, 1968, 2001, box((90.0, 91.2, 30.0, 30.8))), ("Ak-Shirak", -0.54, 0.10, 1980, 2002, box((77.8, 78.8, 41.5, 42.2)))]
K = []
for name, pub, sig, y0, y1, m in SITES:
    k = (yrs >= y0) & (yrs <= y1 - 1)
    for case, mm in [("all", m), ("no_surge", m & ~sg)]:
        r = dict(area=name, case=case, n=int(mm.sum()), n_surge_removed=int((m & sg).sum()), area_surge_pct=100 * area[m & sg].sum() / area[m].sum())
        for i, f in enumerate(GF):
            h = A[i][:, k][mm].astype("float64"); ok = ~np.isnan(h).all(1)
            r[f] = np.average(np.nanmean(h[ok], axis=1), weights=area[mm][ok]) if ok.any() else np.nan
        bb = Bb[:, k][mm]; ok = ~np.isnan(bb).any(1); r["best"] = np.average(bb[ok].mean(1), weights=area[mm][ok])
        K.append(r)
K = pd.DataFrame(K); K.to_csv(OUT / "surge_kh9.csv", index=False)
print("\n== KH-9 rows\n", K.round(3).to_string(index=False))
