#!/usr/bin/env python
"""Representativeness of sigma_prior (sigma_T = 1.5 vs 4 on the 681-glacier test subset).

sigma_prior(period) = sqrt( sum_f w_f dB_f^2 / sum_f w_f ), dB_f = period mean of B(1.5) - B(4) for member f, w_f = equal family
weights of the members covering every year (as best_mb_ensemble.aggregate). Variants of the regional B:
  V0 current      : sum(dm_gt)/sum(area_start) over in-sample subset glaciers (= B_regional_series of the test runs)
  V1 rgi_area     : per-glacier dB weighted by RGI area, subset glaciers in sample in both runs
  V2 strat_all    : V1 with post-stratification weights A_i * A_HMA(s)/A_subset(s), strata = RGI region x size class
  V3 strat_random : V2 on the 600 random glaciers only (field glaciers dropped)
  V4 count        : per-glacier dB, equal weight per glacier
Each variant for HMA and per RGI region. Full-scale check (ERA5, RGI 13): run_04_full (sigma_T 1.5) vs run_05 (sigma_T 4),
both specific-rate targets like the test runs: all 54 429 glaciers, RGI-area weighted and as sum(dm)/sum(A_start); and the
same per-glacier differences on the subset glaciers only (V1/V2/V3 construction).
Output: context/reports/external2_round3/sigma_prior_*.csv. Env oggm."""
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from config import OGGM_ROOT, CODE_DIR  # noqa: E402
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
E2 = OGGM_ROOT
OUT = ROOT / "context/reports/external2_round3"; OUT.mkdir(parents=True, exist_ok=True)
R15, R4 = E2 / "compiled_v3/per_forcing", E2 / "compiled_v3_run_03_sT4/per_forcing"
FORC = ["ERA5", "ERA5-Land", "HARv2", "CRU_TS4", "CHELSA"]
FAM = {"ERA5": "E", "ERA5-Land": "E", "HARv2": "E", "CRU_TS4": "C", "CHELSA": "C"}
PER = {"1950-1979": (1950, 1979), "1980-1999": (1980, 1999), "1980s": (1980, 1989), "1990s": (1990, 1999),
       "2000-2019": (2000, 2019), "2020-2025": (2020, 2025), "2020-2023": (2020, 2023)}
LK = pd.read_csv(ROOT / "analysis/glacier_lookup.csv").set_index("rgi_id")
FIELD = set(pd.read_csv(ROOT / "analysis/D_validation_pairs.csv").rgi_id.dropna())
BINS = [0, 1, 5, 20, 100, 1e9]; BLAB = ["<1", "1-5", "5-20", "20-100", ">100"]
LK["sz"] = pd.cut(LK.area_rgi_km2, BINS, labels=BLAB, right=False)
HMA_STRAT = LK.groupby(["rgi_region", "sz"], observed=False).area_rgi_km2.sum()


def load(root, f, ids=None):
    m = pd.read_csv(root / f / "metadata.csv", usecols=["rgi_id", "calib_status", "backward_vanished"])
    ok = m[(m.calib_status == "ok") & ~m.backward_vanished.astype(str).str.lower().isin(["true", "1"])].rgi_id
    y = pd.read_parquet(root / f / "yearly.parquet", columns=["rgi_id", "year", "mb_annual_mwe", "dm_gt", "area_start_km2"])
    y = y[y.rgi_id.isin(ok) & (y.year >= 1950)]
    return y[y.rgi_id.isin(ids)] if ids is not None else y


def per_glacier_diff(a, b):
    """period-mean dB per glacier (a - b), only glaciers in both and with all years of the period."""
    out = {}
    for p, (y0, y1) in PER.items():
        n = y1 - y0 + 1
        ga = a[a.year.between(y0, y1)].groupby("rgi_id").mb_annual_mwe.agg(["mean", "count"])
        gb = b[b.year.between(y0, y1)].groupby("rgi_id").mb_annual_mwe.agg(["mean", "count"])
        j = ga[ga["count"] == n].join(gb[gb["count"] == n], lsuffix="_a", rsuffix="_b", how="inner")
        if len(j): out[p] = (j.mean_a - j.mean_b)
    return out


def ratio_series_diff(a, b):
    out = {}
    for p, (y0, y1) in PER.items():
        sa = a[a.year.between(y0, y1)].groupby("year").agg(dm=("dm_gt", "sum"), A=("area_start_km2", "sum"))
        sb = b[b.year.between(y0, y1)].groupby("year").agg(dm=("dm_gt", "sum"), A=("area_start_km2", "sum"))
        if len(sa) == y1 - y0 + 1 and len(sb) == y1 - y0 + 1:
            out[p] = (sa.dm / sa.A * 1e3).mean() - (sb.dm / sb.A * 1e3).mean()
    return out


def weights(ids, kind):
    L = LK.loc[ids]
    if kind == "count": return pd.Series(1.0, index=ids)
    if kind == "rgi_area": return L.area_rgi_km2
    sub = L.groupby(["rgi_region", "sz"], observed=False).area_rgi_km2.sum()
    fac = (HMA_STRAT / sub).replace([np.inf], np.nan)
    w = L.area_rgi_km2 * [fac.get((r, s), np.nan) for r, s in zip(L.rgi_region, L.sz)]
    # strata of HMA without subset glaciers: their area is not represented -> report the share
    return w.fillna(0)


def unrepresented(ids):
    L = LK.loc[ids]; have = set(zip(L.rgi_region, L.sz))
    return {r: float(sum(v for (rr, s), v in HMA_STRAT.items() if rr == r and (rr, s) not in have) / HMA_STRAT[r].sum() * 100)
            for r in ["RGI60-13", "RGI60-14", "RGI60-15"]}


rows = []
D = {}   # forcing -> {period: per-glacier dB}
for f in FORC:
    a, b = load(R15, f), load(R4, f)
    D[f] = per_glacier_diff(a, b)
    for unit in ["ALL_HMA", "RGI60-13", "RGI60-14", "RGI60-15"]:
        sel = (lambda d: d) if unit == "ALL_HMA" else (lambda d, u=unit: d[d.rgi_id.str[:8] == u])
        for p, v in ratio_series_diff(sel(a), sel(b)).items():
            rows.append(dict(forcing=f, unit=unit, period=p, variant="V0_current", dB=v))
    for p, d in D[f].items():
        for unit in ["ALL_HMA", "RGI60-13", "RGI60-14", "RGI60-15"]:
            ids = d.index if unit == "ALL_HMA" else d.index[d.index.str[:8] == unit]
            rnd = ids[~ids.isin(FIELD)]
            for var, ii, kind in [("V1_rgi_area", ids, "rgi_area"), ("V2_strat_all", ids, "strat"),
                                  ("V3_strat_random", rnd, "strat"), ("V4_count", ids, "count")]:
                w = weights(ii, kind)
                rows.append(dict(forcing=f, unit=unit, period=p, variant=var, dB=float(np.average(d[ii], weights=w)), n=len(ii)))
M = pd.DataFrame(rows); M.to_csv(OUT / "sigma_prior_members.csv", index=False)


def sig(g):
    w = {f: 1.0 for f in g.forcing}
    fams = {}
    for f in w: fams.setdefault(FAM[f], []).append(f)
    w = pd.Series({f: 1 / len(fams) / len(fams[FAM[f]]) for f in w})
    d = g.set_index("forcing").dB
    return float(np.sqrt((w * d ** 2).sum() / w.sum()))


S = M.groupby(["unit", "period", "variant"]).apply(sig, include_groups=False).rename("sigma_prior").reset_index()
S.to_csv(OUT / "sigma_prior_variants.csv", index=False)
pd.set_option("display.width", 250)
print(S.pivot_table(index=["unit", "period"], columns="variant", values="sigma_prior").round(3).to_string())
BP = pd.read_csv(ROOT / "tables/best_mb_periods.csv")
print("\npaper sigma_prior (tables/best_mb_periods.csv):",
      BP[BP.level.isin(["hma", "rgi_region"])].set_index(["unit", "period"]).sigma_prior.round(3).loc[lambda s: s.index.get_level_values(1).isin(list(PER))].to_dict())
print("\nstrata of HMA area with no subset glacier (% of region area):", unrepresented(list(D["ERA5"]["1980-1999"].index)))
print("subset composition (ERA5, 1980-1999 glaciers):", LK.loc[D["ERA5"]["1980-1999"].index].groupby(["rgi_region", "sz"], observed=False).size().unstack().to_string())

# ---- full-scale ERA5 x RGI 13
fa = load(E2 / "compiled_v3_run_04_full/per_forcing", "ERA5"); fa = fa[fa.rgi_id.str.startswith("RGI60-13")]
fb = load(E2 / "compiled_v3_run_05_era5_r13_sT4/per_forcing", "ERA5")
Dfull = per_glacier_diff(fa, fb); Rfull = ratio_series_diff(fa, fb)
sub13 = D["ERA5"]["1980-1999"].index; sub13 = sub13[sub13.str[:8] == "RGI60-13"]
F = []
for p, d in Dfull.items():
    r = dict(period=p, full_ratio=Rfull.get(p), full_rgi_area=float(np.average(d, weights=LK.loc[d.index].area_rgi_km2)),
             full_count=float(d.mean()), n_full=len(d))
    ii = d.index.intersection(sub13); rnd = ii[~ii.isin(FIELD)]
    r.update(fullrun_on_subset_rgi_area=float(np.average(d[ii], weights=weights(ii, "rgi_area"))),
             fullrun_on_subset_strat=float(np.average(d[ii], weights=weights(ii, "strat"))),
             fullrun_on_random_strat=float(np.average(d[rnd], weights=weights(rnd, "strat"))), n_subset=len(ii))
    t = M[(M.forcing == "ERA5") & (M.unit == "RGI60-13") & (M.period == p)].set_index("variant").dB
    r.update(testrun_V0=t.get("V0_current"), testrun_V1=t.get("V1_rgi_area"), testrun_V2=t.get("V2_strat_all"), testrun_V3=t.get("V3_strat_random"))
    if p in D["ERA5"]:
        dd = D["ERA5"][p]; jj = dd.index.intersection(d.index)
        r["max_abs_glacier_diff_test_vs_full"] = float((dd[jj] - d[jj]).abs().max())
    F.append(r)
F = pd.DataFrame(F); F.to_csv(OUT / "sigma_prior_fullscale_era5_r13.csv", index=False)
print("\nFull-scale ERA5 RGI-13 dB = B(1.5) - B(4), m w.e./yr (single member, so sigma_prior contribution = |dB|):")
print(F.round(3).to_string(index=False))
