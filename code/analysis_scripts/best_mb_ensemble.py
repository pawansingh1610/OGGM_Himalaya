#!/usr/bin/env python
"""Best-estimate HMA mass balance from the five forcings

Design:
  - member weights from the split-sample geodetic test and the field anomaly correlation;
  - family correction: CRU family (CRU TS4, CHELSA-W5E5) vs ERA5 family (ERA5, ERA5-Land, HARv2);
  - leave-one-out validation; fall back to equal family weights if weighting does not win;
  - annual / pentad / decadal outputs with sigma_forcing, sigma_calib, sigma_prior, sigma_trend listed separately;
  - n_members / n_families flags; 1950-1979 "not constrained"; 2020-2025 "unconstrained extrapolation".

Definitions (all inputs are files read below; nothing is typed in by hand):
  split skill   s_f = max(0, 1 - |dMod_f - dObs| / |dObs|), HMA, area-weighted (analysis/E_split_sample.csv).
                CHELSA ends in 2016 and has no split test: it gets the lowest split skill of the other members.
  field skill   r_f = anomaly correlation on the common five-forcing subset (analysis/D_validation_pairs.csv).
  member score  q_f = s_f * r_f.
  weighted      family weight proportional to the mean score of its members present that year; within a family,
                proportional to q_f.   equal-family: 1/2 per family present, equal within the family.
  LOO tests     (1) leave-one-field-glacier-out: r_f recomputed without the glacier, ensemble predicted for that
                glacier; pooled RMSE and anomaly r of the held-out predictions.  (2) leave-one-sub-region-out:
                s_f recomputed from the other 14 sub-regions (area-weighted), ensemble change 2000s->2010s
                predicted for the held-out sub-region; area-weighted mean absolute error.
                Weighting "wins" only if it beats equal-family weights in BOTH tests; otherwise equal-family.
  sigma_forcing weighted standard deviation of the members around the best estimate (NaN with one member).
  sigma_calib   Hugonnet et al. (2021) regional 2000-2019 uncertainty (Extended Data Table 1, 1 sigma = 95 % CI / 1.96;
                HMA: the three regions summed linearly). Systematic: same in every year, does not shrink when averaging.
                Bounds from the per-glacier errors (fully correlated / independent) are kept as extra columns.
  sigma_trend     observed sub-period misfit: half the absolute split-test misfit of the chosen ensemble (a model
                calibrated to the 20-yr mean misplaces each 10-yr half by this much), scaled with the distance of the year or
                period midpoint from the centre of the calibration period (2009.5): factor max(1, |mid - 2009.5| / 5), i.e. a
                trend error that persists linearly outside 2000-2019.
                The unscaled value is kept as sigma_trend_base. 1980-1999: factor 4; 1950-1979: 9; 2020-2025: 2.6.
  sigma_prior   prior sensitivity: weighted RMS over members of B(sigma_T = 1.5) - B(sigma_T = 4) on the
                681-glacier test subset (compiled_v3 vs compiled_v3_run_03_sT4), computed for pentads, decades and
                periods only; an annual value carries the sigma_prior of its pentad (no single-year values).
  sigma_family  only where one family is available (2020-2025: ERA5 family only; years >= 2020 and periods starting in 2020):
                RMS over 2000-2019 of the difference between the two family means (CRU family minus ERA5 family, members present in each
                year): annual values use annual means, period means (2020-2025, 2020-2023, ...) use running means of the same length.
  sigma_total   quadrature sum of the four (five with sigma_family; assumes independence).
Outputs: tables/best_mb_{annual,pentad,decadal,weights,loo}.csv, tables/T7_best_mb.tex, tables/S8_ensemble_weights.tex,
figures/F13_best_mb.png.
"""
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from config import OGGM_ROOT  # noqa: E402
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_figures import (FORC, LABEL, COL, LS, INK, INK2, MUTED, AXIS, DOUBLE, write_table, fmt, save,  # noqa: E402
                          TAB, shade_periods, legend_handles)
from paths import PER_FORCING  # noqa: E402
from hugonnet_regional import regional  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

AN = ROOT / "analysis"
TEST_15 = OGGM_ROOT / "compiled_v3" / "analysis"            # sigma_T = 1.5, test subset
TEST_4 = OGGM_ROOT / "compiled_v3_run_03_sT4" / "analysis"  # sigma_T = 4,   test subset
FAMILY = {"ERA5": "ERA5 family", "ERA5-Land": "ERA5 family", "HARv2": "ERA5 family",
          "CRU_TS4": "CRU family", "CHELSA": "CRU family"}


def period_flag(y0, y1=None):
    y1 = y0 if y1 is None else y1
    if y1 <= 1979: return "not constrained"
    if y0 >= 2020: return "unconstrained extrapolation"
    if y0 >= 2000 and y1 <= 2019: return "geodetic calibration period"
    if y0 >= 1980 and y1 <= 1999: return "reconstruction"
    return "mixed"


# ---------------------------------------------------------------- skills
def split_skill(E, exclude=None, area=None):
    """s_f from E_split_sample; exclude = sub-region left out (area-weighted over the others)."""
    s = {}
    if exclude is None:
        h = E[E.subregion == "ALL_HMA"].set_index("forcing")
        for f, r in h.iterrows():
            s[f] = max(0.0, 1 - abs(r.dMod_areaw - r.dObs_areaw) / abs(r.dObs_areaw))
    else:
        d = E[(E.subregion != "ALL_HMA") & (E.subregion != exclude)].copy()
        d["a"] = d.subregion.map(area)
        for f, g in d.groupby("forcing"):
            mo = np.average(g.dMod_areaw, weights=g.a); ob = np.average(g.dObs_areaw, weights=g.a)
            s[f] = max(0.0, 1 - abs(mo - ob) / abs(ob))
    s = pd.Series(s)
    for f in FORC:            # CHELSA (no split test): lowest split skill of the others
        if f not in s: s[f] = s.min()
    return s[FORC]


def anomaly_r(P, f, drop=None):
    d = P.dropna(subset=["obs"] + FORC)          # common five-forcing subset (as in D_validation_overall)
    if drop is not None: d = d[d.rgi_id != drop]
    ao = d.obs - d.groupby("rgi_id").obs.transform("mean")
    am = d[f] - d.groupby("rgi_id")[f].transform("mean")
    return float(np.corrcoef(ao, am)[0, 1])


def field_skill(P, drop=None):
    return pd.Series({f: max(0.0, anomaly_r(P, f, drop)) for f in FORC})[FORC]


def member_weights(q, present, scheme):
    """Normalised weights for the members in `present` (list), family-corrected."""
    present = [f for f in FORC if f in present]
    if not present: return pd.Series(dtype=float)
    fams = {}
    for f in present: fams.setdefault(FAMILY[f], []).append(f)
    w = {}
    if scheme == "equal_member":
        return pd.Series({f: 1 / len(present) for f in present})
    if scheme == "equal_family":
        for fam, ms in fams.items():
            for f in ms: w[f] = 1 / len(fams) / len(ms)
        return pd.Series(w)
    # weighted: family weight ~ mean member score, within family ~ member score
    fw = {fam: np.mean([q[f] for f in ms]) for fam, ms in fams.items()}
    tot = sum(fw.values())
    for fam, ms in fams.items():
        qs = sum(q[f] for f in ms)
        for f in ms:
            w[f] = (fw[fam] / tot) * (q[f] / qs if qs > 0 else 1 / len(ms))
    return pd.Series(w)


def combine(row, q, scheme):
    present = [f for f in FORC if pd.notna(row.get(f))]
    w = member_weights(q, present, scheme)
    if w.empty: return np.nan, w
    return float(sum(w[f] * row[f] for f in w.index)), w


# ---------------------------------------------------------------- leave-one-out
def loo_field(P, s_split):
    P = P[P.rgi_id.notna()]
    out = {k: [] for k in ["weighted", "equal_family", "equal_member"]}
    for g, d in P.groupby("rgi_id"):
        q = s_split * field_skill(P, drop=g)
        for sch in out:
            for _, r in d.iterrows():
                b, _ = combine(r, q, sch)
                if pd.notna(b): out[sch].append((g, r.obs, b))
    res = {}
    for sch, L in out.items():
        d = pd.DataFrame(L, columns=["rgi_id", "obs", "mod"])
        ao = d.obs - d.groupby("rgi_id").obs.transform("mean")
        am = d["mod"] - d.groupby("rgi_id")["mod"].transform("mean")
        res[sch] = dict(N=len(d), n_glaciers=d.rgi_id.nunique(), bias=(d["mod"] - d.obs).mean(),
                        RMSE=np.sqrt(((d["mod"] - d.obs) ** 2).mean()), r_anomaly=np.corrcoef(ao, am)[0, 1])
    return pd.DataFrame(res).T


def loo_split(E, r_field, area):
    subs = [s for s in E.subregion.unique() if s != "ALL_HMA"]
    res = {k: [] for k in ["weighted", "equal_family", "equal_member"]}
    for k in subs:
        q = split_skill(E, exclude=k, area=area) * r_field
        d = E[E.subregion == k].set_index("forcing")
        row = d.dMod_areaw.to_dict()
        obs = d.dObs_areaw.mean()
        for sch in res:
            b, _ = combine(row, q, sch)
            res[sch].append((k, area[k], obs, b))
    out = {}
    for sch, L in res.items():
        d = pd.DataFrame(L, columns=["subregion", "area", "obs", "ens"])
        out[sch] = dict(n_subregions=len(d), MAE_areaw=np.average((d.ens - d.obs).abs(), weights=d.area),
                        RMSE_areaw=np.sqrt(np.average((d.ens - d.obs) ** 2, weights=d.area)))
    return pd.DataFrame(out).T


# ---------------------------------------------------------------- series
def wstats(vals, w):
    """weighted mean and weighted SD (vals, w: Series on the same index)."""
    w = w / w.sum()
    m = float((w * vals).sum())
    sd = float(np.sqrt((w * (vals - m) ** 2).sum())) if len(vals) > 1 else np.nan
    return m, sd


FAM_E = ["ERA5", "ERA5-Land", "HARv2"]
FAM_C = ["CRU_TS4", "CHELSA"]


CAL_MID, HALF_DECADE = 2009.5, 5.0   # centre of the 2000-2019 calibration period; distance of a decade centre from it


def trend_factor(mid):
    """Scale of sigma_trend: 1 inside the calibration decades, growing linearly with the distance of the midpoint from 2009.5."""
    return np.maximum(1.0, np.abs(np.asarray(mid, dtype=float) - CAL_MID) / HALF_DECADE)


def scale_trend(d, annual):
    """sigma_trend = sigma_trend_base * trend_factor(midpoint); recompute sigma_total (all five terms, in quadrature)."""
    if "sigma_trend_base" not in d: d["sigma_trend_base"] = d["sigma_trend"]
    mid = d["year"] if annual else (d["y0"] + d["y1"]) / 2
    d["sigma_trend"] = d["sigma_trend_base"] * trend_factor(mid)
    cols = [c for c in ["sigma_forcing", "sigma_calib", "sigma_prior", "sigma_trend", "sigma_family"] if c in d]
    d["sigma_total"] = np.sqrt(d[cols].fillna(0).pow(2).sum(axis=1))
    return d


def family_rms(Bu, L=1):
    """RMS over 2000-2019 of the difference between the CRU-family mean and the ERA5-family mean (members present), for annual
    values (L = 1) or for running means of L years (windows entirely inside 2000-2019): the scale of the between-family difference of
    an L-year mean in the calibration period, where both families are constrained to the same 20-year mean."""
    piv = Bu.pivot(index="year", columns="forcing", values="B_mwe").reindex(columns=FORC).loc[2000:2019]
    d = (piv[FAM_C].mean(axis=1, skipna=True) - piv[FAM_E].mean(axis=1, skipna=True)).dropna()
    d = d.rolling(L).mean().dropna() if L > 1 else d
    return float(np.sqrt((d ** 2).mean())) if len(d) >= 10 else np.nan


def add_family(d, Bu, annual):
    """sigma_family where only the ERA5 family is available and the year / period starts in 2020: annual values use the annual RMS,
    period means the RMS of running means of the same length; update sigma_total."""
    start = d["year"] if annual else d["y0"]
    cache = {}
    def val(L):
        if L not in cache: cache[L] = family_rms(Bu, L)
        return cache[L]
    L = pd.Series(1, index=d.index) if annual else d["n_years"]
    sf = np.array([val(int(l)) for l in L])
    d["sigma_family"] = np.where((d.n_families == 1) & (start >= 2020) & np.isfinite(sf), sf, np.nan)
    d["sigma_total"] = np.sqrt(d[["sigma_forcing", "sigma_calib", "sigma_prior", "sigma_trend", "sigma_family"]].fillna(0).pow(2).sum(axis=1))
    return d


def build_series(B, q, scheme, prior_diff, s_calib, s_obs, unit_col="unit"):
    """B: forcing x year B_mwe / dm_gt for one unit. prior_diff: forcing x year dB(1.5-4)."""
    piv = B.pivot(index="year", columns="forcing", values="B_mwe").reindex(columns=FORC)
    dm = B.pivot(index="year", columns="forcing", values="dm_gt").reindex(columns=FORC)
    rows = []
    for y, r in piv.iterrows():
        present = [f for f in FORC if pd.notna(r[f])]
        if not present: continue
        w = member_weights(q, present, scheme)
        b, sdf = wstats(r[w.index], w)
        pdv = prior_diff.reindex(index=[y], columns=w.index).iloc[0] if y in prior_diff.index else pd.Series(dtype=float)
        pdv = pdv.dropna()
        sp = float(np.sqrt((w[pdv.index] / w[pdv.index].sum() * pdv ** 2).sum())) if len(pdv) else np.nan
        rows.append(dict(year=y, B_best=b, dM_Gt_best=float((w * dm.loc[y, w.index]).sum()),
                         sigma_forcing=sdf, sigma_calib=s_calib, sigma_prior=sp, sigma_trend=s_obs,
                         n_members=len(present), n_families=len({FAMILY[f] for f in present}),
                         members="+".join(present), weights=";".join(f"{f}:{w[f]:.3f}" for f in w.index),
                         flag=period_flag(y)))
    d = pd.DataFrame(rows)
    d["sigma_total"] = np.sqrt(d[["sigma_forcing", "sigma_calib", "sigma_prior", "sigma_trend"]].fillna(0).pow(2).sum(axis=1))
    return d


def aggregate(B, q, scheme, prior_B15, prior_B4, s_calib, s_obs, bins):
    """period means: members averaged first, then combined (so sigma_forcing is the spread of period means).
    A member counts for a period only if it covers every year of it."""
    piv = B.pivot(index="year", columns="forcing", values="B_mwe").reindex(columns=FORC)
    p15 = prior_B15.pivot(index="year", columns="forcing", values="B_mwe").reindex(columns=FORC)
    p4 = prior_B4.pivot(index="year", columns="forcing", values="B_mwe").reindex(columns=FORC)
    rows = []
    for lab, (a, b) in bins.items():
        yrs = list(range(a, b + 1))
        sub = piv.reindex(yrs)
        present = [f for f in FORC if sub[f].notna().all()]
        if not present: continue
        means = sub[present].mean()
        w = member_weights(q, present, scheme)
        m, sdf = wstats(means[w.index], w)
        dpr = (p15.reindex(yrs)[present].mean() - p4.reindex(yrs)[present].mean()).dropna()
        sp = float(np.sqrt((w[dpr.index] / w[dpr.index].sum() * dpr ** 2).sum())) if len(dpr) else np.nan
        rows.append(dict(period=lab, y0=a, y1=b, n_years=len(yrs), B_best=m, sigma_forcing=sdf, sigma_calib=s_calib,
                         sigma_prior=sp, sigma_trend=s_obs, n_members=len(present),
                         n_families=len({FAMILY[f] for f in present}), members="+".join(present), flag=period_flag(a, b)))
    d = pd.DataFrame(rows)
    d["sigma_total"] = np.sqrt(d[["sigma_forcing", "sigma_calib", "sigma_prior", "sigma_trend"]].fillna(0).pow(2).sum(axis=1))
    return d


def main():
    E = pd.read_csv(AN / "E_split_sample.csv")
    P = pd.read_csv(AN / "D_validation_pairs.csv")
    A = pd.read_csv(AN / "A_calibration_diagnostics.csv")
    area = A[A.forcing == "HARv2"].set_index("subregion").area_km2.to_dict()
    B = pd.read_csv(AN / "B_regional_series.csv")

    s_split = split_skill(E)
    r_field = field_skill(P)
    q = s_split * r_field
    print("skills:\n", pd.DataFrame({"split": s_split, "field_r_anom": r_field, "score": q}).round(3).to_string())

    # ---- leave-one-out
    lf = loo_field(P, s_split)
    ls = loo_split(E, r_field, area)
    print("LOO field (held-out glaciers):\n", lf.round(3).to_string())
    print("LOO split (held-out sub-regions):\n", ls.round(4).to_string())
    win_field = lf.loc["weighted", "RMSE"] < lf.loc["equal_family", "RMSE"]
    win_split = ls.loc["weighted", "MAE_areaw"] < ls.loc["equal_family", "MAE_areaw"]
    scheme = "weighted" if (win_field and win_split) else "equal_family"
    # The main best estimate uses equal family weights; the skill-weighted estimate is the sensitivity result
    # (best_mb_scheme_sensitivity.csv).
    scheme = "equal_family"
    print(f"weighted beats equal-family: field {win_field}, split {win_split} -> scheme = {scheme}")
    loo = pd.concat({"field_leave_one_glacier_out": lf, "split_leave_one_subregion_out": ls}, axis=0)
    loo.to_csv(TAB / "best_mb_loo.csv")

    # ---- weights table (all five present)
    W = pd.DataFrame({"family": pd.Series(FAMILY)[FORC], "split_skill": s_split, "field_anomaly_r": r_field,
                      "score": q, "w_weighted": member_weights(q, FORC, "weighted"),
                      "w_equal_family": member_weights(q, FORC, "equal_family"),
                      "w_equal_member": member_weights(q, FORC, "equal_member")})
    W["chosen_scheme"] = scheme
    W.to_csv(TAB / "best_mb_weights.csv")
    print(W.round(3).to_string())

    # ---- sigma_trend: observed sub-period misfit = half the split-test misfit of the ensemble (HMA / sub-region)
    def ens_split(unit):
        d = E[E.subregion == unit].set_index("forcing")
        b, _ = combine(d.dMod_areaw.to_dict(), q, scheme)
        return 0.5 * abs(b - d.dObs_areaw.mean())
    # ---- sigma_calib: Hugonnet et al. (2021) regional 2000-2019 uncertainty (1 sigma), systematic
    H = regional()
    lk = pd.read_csv(AN / "glacier_lookup.csv", usecols=["rgi_id", "subregion", "rgi_region", "area_rgi_km2"])
    sub_reg = (lk.groupby(["subregion", "rgi_region"]).area_rgi_km2.sum().reset_index()
               .sort_values("area_rgi_km2").drop_duplicates("subregion", keep="last").set_index("subregion").rgi_region)
    s_cal = {"ALL_HMA": float(H.loc["ALL_HMA", "sigma1_mwe_corr"])}      # regions summed linearly (conservative)
    s_cal.update({r: float(H.loc[r, "sigma1_mwe"]) for r in ["RGI60-13", "RGI60-14", "RGI60-15"]})
    s_cal.update({s: s_cal[r] for s, r in sub_reg.items()})            # sub-region: its majority RGI region
    # supplementary bounds: per-glacier Hugonnet errors fully correlated / independent
    m = pd.read_csv(PER_FORCING / "ERA5" / "metadata.csv", usecols=["rgi_id", "A_ref_km2", "hug_err_2000_2020",
                                                                    "calib_status", "backward_vanished"])
    m = m[(m.calib_status == "ok") & ~m.backward_vanished.astype(str).str.lower().isin(["true", "1"])]
    m = m.merge(lk[["rgi_id", "subregion", "rgi_region"]], on="rgi_id", how="left")
    cor = lambda g: float(np.average(g.hug_err_2000_2020, weights=g.A_ref_km2))
    unc = lambda g: float(np.sqrt(((g.A_ref_km2 * g.hug_err_2000_2020) ** 2).sum()) / g.A_ref_km2.sum())
    bnd = {"ALL_HMA": (cor(m), unc(m))}
    for col in ["subregion", "rgi_region"]:
        for u, g in m.groupby(col): bnd[u] = (cor(g), unc(g))
    print(f"sigma_calib HMA (Hugonnet regional, 1 sigma) = {s_cal['ALL_HMA']:.4f}; bounds from per-glacier errors: "
          f"correlated {bnd['ALL_HMA'][0]:.4f}, independent {bnd['ALL_HMA'][1]:.4f}")
    # ---- prior sensitivity (test subset)
    b15 = pd.read_csv(TEST_15 / "B_regional_series.csv"); b4 = pd.read_csv(TEST_4 / "B_regional_series.csv")

    bins_dec = {f"{d}s" if d < 2020 else "2020-2025": (d, min(d + 9, 2025)) for d in range(1950, 2030, 10)}
    bins_dec = {k: v for k, v in bins_dec.items() if v[0] <= 2025}
    bins_pent = {f"{p}-{p + 4}": (p, p + 4) for p in range(1950, 2025, 5)}
    bins_pent["2025"] = (2025, 2025)
    bins_key = {"1950-1979": (1950, 1979), "1980-1999": (1980, 1999), "2000-2019": (2000, 2019),
                "2020-2025": (2020, 2025), "2000-2023": (2000, 2023), "2020-2023": (2020, 2023),
                "1980-2019": (1980, 2019)}

    fam_rms = {}

    def run(sch):
        ann, pen, dec, key = [], [], [], []
        for (lev, unit), Bu in B.groupby(["level", "unit"]):
            Bu = Bu[(Bu.year >= 1950)].dropna(subset=["B_mwe"])
            unit_e = "ALL_HMA" if lev in ("hma", "rgi_region") else unit
            if unit_e not in set(E.subregion) or unit not in s_cal and unit_e not in s_cal: continue
            so = ens_split(unit_e)
            sc = s_cal[unit] if unit in s_cal else s_cal[unit_e]
            lv = lev if lev != "rgi_region" else "rgi_region"
            t15 = b15[(b15.level == lv) & (b15.unit == unit)]; t4 = b4[(b4.level == lv) & (b4.unit == unit)]
            if not (len(t15) and len(t4)):
                t15, t4 = b15[b15.level == "hma"], b4[b4.level == "hma"]
            pdu = (t15.pivot(index="year", columns="forcing", values="B_mwe")
                   - t4.pivot(index="year", columns="forcing", values="B_mwe")).reindex(columns=FORC)
            a = build_series(Bu, q, sch, pdu, sc, so)
            out = [a]
            for store, bins in [(pen, bins_pent), (dec, bins_dec), (key, bins_key)]:
                g = aggregate(Bu, q, sch, t15, t4, sc, so, bins)
                out.append(g); store.append(g)
                if bins is bins_pent:   # sigma_prior only at pentad resolution: each year gets its pentad's value
                    yp = {y: r.sigma_prior for _, r in g.iterrows() for y in range(int(r.y0), int(r.y1) + 1)}
                    a["sigma_prior"] = a.year.map(yp)
                    a["sigma_total"] = np.sqrt(a[["sigma_forcing", "sigma_calib", "sigma_prior", "sigma_trend"]]
                                               .fillna(0).pow(2).sum(axis=1))
            fam_rms[(lev, unit)] = (family_rms(Bu, 1), family_rms(Bu, 4), family_rms(Bu, 6))
            add_family(a, Bu, True); scale_trend(a, True)
            for d in out[1:]:
                add_family(d, Bu, False); scale_trend(d, False)
            for d in out:
                d["sigma_calib_bound_correlated"], d["sigma_calib_bound_independent"] = bnd.get(unit, bnd["ALL_HMA"])
                d.insert(0, "unit", unit); d.insert(0, "level", lev)
            ann.append(a)
        return [pd.concat(x, ignore_index=True) for x in (ann, pen, dec, key)]

    ann, pen, dec, key = run(scheme)
    print("sigma_family (RMS 2000-2019 of CRU-family minus ERA5-family means; annual / 4-yr / 6-yr running means):",
          {f"{k[0]}:{k[1]}": tuple(round(x, 3) for x in v) for k, v in fam_rms.items() if k[0] in ("hma", "rgi_region")})
    for name, d in [("annual", ann), ("pentad", pen), ("decadal", dec), ("periods", key)]:
        d["scheme"] = scheme
        d.to_csv(TAB / f"best_mb_{name}.csv", index=False)
    h = lambda d: d[d.level == "hma"]
    pd.set_option("display.width", 250)
    cols = ["period", "B_best", "sigma_forcing", "sigma_calib", "sigma_prior", "sigma_trend", "sigma_total",
            "n_members", "n_families", "flag"]
    print("HMA decadal:\n", h(dec)[cols].round(3).to_string(index=False))
    print("HMA periods:\n", h(key)[cols].round(3).to_string(index=False))

    # ---- check against Hugonnet 2000-2019 (HMA and RGI regions)
    rows = []
    for u in ["ALL_HMA", "RGI60-13", "RGI60-14", "RGI60-15"]:
        k = key[(key.unit == u) & (key.period == "2000-2019")].iloc[0]
        rows.append(dict(unit=u, B_best_2000_2019=k.B_best, hugonnet_B=H.loc[u, "B_mwe"],
                         hugonnet_sigma1=H.loc[u, "sigma1_mwe_corr" if u == "ALL_HMA" else "sigma1_mwe"],
                         diff=k.B_best - H.loc[u, "B_mwe"], hugonnet_area_km2=H.loc[u, "area_km2"]))
    hc = pd.DataFrame(rows); hc.to_csv(TAB / "best_mb_vs_hugonnet.csv", index=False)
    print("best estimate vs Hugonnet (2000-2019):\n", hc.round(4).to_string(index=False))

    # ---- sensitivity: the other scheme (equal family weights if the weighted one is chosen)
    alt = "equal_family" if scheme == "weighted" else "weighted"
    _, pen2, dec2, key2 = run(alt)
    sens = []
    for lab, d1, d2 in [("pentad", pen, pen2), ("decadal", dec, dec2), ("periods", key, key2)]:
        m_ = d1.merge(d2[["level", "unit", "period", "B_best"]], on=["level", "unit", "period"], suffixes=("", "_alt"))
        m_["diff"] = m_.B_best - m_.B_best_alt
        m_["abs_diff_gt_sigma_total"] = m_["diff"].abs() > m_.sigma_total
        m_["resolution"] = lab
        sens.append(m_[["resolution", "level", "unit", "period", "B_best", "B_best_alt", "diff", "sigma_total",
                        "abs_diff_gt_sigma_total"]])
    sens = pd.concat(sens, ignore_index=True); sens["alt_scheme"] = alt
    sens.to_csv(TAB / "best_mb_scheme_sensitivity.csv", index=False)
    hs = sens[sens.level == "hma"]
    print(f"scheme sensitivity ({scheme} minus {alt}), HMA:\n", hs.round(3).to_string(index=False))
    print("cases with |diff| > sigma_total (any unit):\n",
          sens[sens.abs_diff_gt_sigma_total].round(3).to_string(index=False) if sens.abs_diff_gt_sigma_total.any() else "none")
    alt_b = hs[hs.resolution != "pentad"].drop_duplicates("period").set_index("period").B_best_alt
    t7(h(dec), h(key), scheme, alt_b, alt)
    s8(W, lf, ls, scheme)
    # figure: scripts/figures/fig13_best_mb.py (reads the tables written here)


def t7(dec, key, scheme, alt_b, alt):
    rows = []
    for d, mid in [(dec, False), (key[key.period.isin(["1950-1979", "1980-1999", "2000-2019", "2020-2023"])], True)]:
        if mid: rows.append("MID")
        for _, r in d.iterrows():
            lab = r.period.replace("-", "--")
            rows.append(f"{lab} & {int(r.n_members)} & {int(r.n_families)} & {fmt(r.B_best, 3)} & "
                        f"{fmt(r.sigma_forcing, 3, sign=False)} & {fmt(r.sigma_calib, 3, sign=False)} & "
                        f"{fmt(r.sigma_prior, 3, sign=False)} & {fmt(r.sigma_trend, 3, sign=False)} & "
                        f"{fmt(r.sigma_family, 3, sign=False) if pd.notna(r.sigma_family) else '--'} & "
                        f"{fmt(r.sigma_total, 3, sign=False)} & {fmt(alt_b.get(r.period, np.nan), 3)} & {r.flag}")
    sch = {"weighted": "skill-weighted with family correction",
           "equal_family": "equal family weights"}[scheme]
    write_table("T7_best_mb", "Best-estimate HMA specific mass balance (m w.e. yr$^{-1}$) by decade and period, "
                f"{sch}. A member is used for a period only if it covers every year of it. "
                "$\\sigma_\\mathrm{forcing}$: weighted spread of the members; $\\sigma_\\mathrm{calib}$: regional 2000--2019 "
                "uncertainty of Hugonnet et al. (2021); $\\sigma_\\mathrm{prior}$: full difference to the tighter, rejected prior ($\\sigma_T$ = 1.5 "
                "against 4\\,$^\\circ$C, 681-glacier subset), taken as 1$\\sigma$; $\\sigma_\\mathrm{trend}$: half the split-test misfit, scaled with the distance from the calibration period (Sect.~\\ref{sec:ensemble}); "
                "$\\sigma_\\mathrm{family}$: difference between the two families, added from 2020, when only the ERA5 family is "
                "available (Sect.~\\ref{sec:ensemble}); total in quadrature, all 1$\\sigma$. "
                "$B_\\mathrm{alt}$: the same estimate with "
                + ("equal family weights" if alt == "equal_family" else "skill weights") + ". The flag gives the level of constraint; rows flagged not constrained (1950--1979) give the weighted mean and spread of three members, not a best estimate.",
                "tab:best_mb", "lrrrrrrrrrrl",
                ["Period & $n_\\mathrm{mem}$ & $n_\\mathrm{fam}$ & $B$ & $\\sigma_\\mathrm{forcing}$ & $\\sigma_\\mathrm{calib}$ & "
                 "$\\sigma_\\mathrm{prior}$ & $\\sigma_\\mathrm{trend}$ & $\\sigma_\\mathrm{family}$ & $\\sigma_\\mathrm{total}$ & $B_\\mathrm{alt}$ & Flag"],
                rows, star=True, size="\\scriptsize")


def s8(W, lf, ls, scheme):
    rows = [f"{LABEL[f]} & {r.family} & {r.split_skill:.2f} & {r.field_anomaly_r:.3f} & {r.score:.3f} & "
            f"{r.w_weighted:.3f} & {r.w_equal_family:.3f}" for f, r in W.iterrows()]
    rows.append("MID")
    rows.append("\\multicolumn{7}{l}{\\textit{Leave-one-out validation}} ")
    for sch, lab in [("weighted", "Skill-weighted"), ("equal_family", "Equal family"), ("equal_member", "Equal member")]:
        rows.append(f"{lab} & \\multicolumn{{2}}{{l}}{{field: RMSE {lf.loc[sch, 'RMSE']:.3f}, anom. $r$ {lf.loc[sch, 'r_anomaly']:.3f}}} & "
                    f"\\multicolumn{{4}}{{l}}{{split: area-weighted MAE {ls.loc[sch, 'MAE_areaw']:.4f} m w.e. yr$^{{-1}}$}}")
    write_table("S8_ensemble_weights", "Ensemble member skills, weights (all five members present) and leave-one-out "
                "validation. Split skill = $1 - |\\Delta B_\\mathrm{mod} - \\Delta B_\\mathrm{obs}| / |\\Delta B_\\mathrm{obs}|$ "
                "(HMA, 2010--2019 minus 2000--2009); CHELSA-W5E5 has no split test (ends 2016) and is given the lowest split skill "
                "of the other members. Field $r$ = anomaly correlation on the common five-forcing subset. Score = product. "
                "Field LOO: one field glacier left out at a time (field $r$ recomputed), "
                f"{int(lf.loc['weighted', 'N'])} held-out glacier-years. Split LOO: one sub-region left out at a time. "
                f"Chosen scheme: {scheme.replace('_', ' ')}.",
                "tab:S8", "llrrrrr", ["Forcing & Family & Split skill & Field $r$ & Score & $w$ weighted & $w$ equal family"],
                rows, size="\\scriptsize")


def rescale_from_csv():
    """Apply the sigma_trend scaling to the existing best_mb_*.csv without recomputing the members or sigma_prior (which
    needs the sigma_T = 1.5 / 4 test-subset outputs). Idempotent: uses sigma_trend_base if present. Rewrites the
    four series, the scheme-sensitivity table and T7. A full run of main() gives the same result."""
    out = {}
    for name in ["annual", "pentad", "decadal", "periods"]:
        d = pd.read_csv(TAB / f"best_mb_{name}.csv")
        out[name] = scale_trend(d, name == "annual")
        out[name].to_csv(TAB / f"best_mb_{name}.csv", index=False)
    sens = pd.read_csv(TAB / "best_mb_scheme_sensitivity.csv")
    tot = pd.concat([out[k].assign(resolution=r) for k, r in [("pentad", "pentad"), ("decadal", "decadal"), ("periods", "periods")]])
    sens = sens.drop(columns="sigma_total").merge(tot[["resolution", "level", "unit", "period", "sigma_total"]],
                                                  on=["resolution", "level", "unit", "period"], how="left")
    sens["abs_diff_gt_sigma_total"] = sens["diff"].abs() > sens.sigma_total
    sens = sens[["resolution", "level", "unit", "period", "B_best", "B_best_alt", "diff", "sigma_total",
                 "abs_diff_gt_sigma_total", "alt_scheme"]]
    sens.to_csv(TAB / "best_mb_scheme_sensitivity.csv", index=False)
    scheme = out["periods"].scheme.iloc[0]; alt = sens.alt_scheme.iloc[0]
    hs = sens[sens.level == "hma"]
    alt_b = hs[hs.resolution != "pentad"].drop_duplicates("period").set_index("period").B_best_alt
    h = lambda d: d[d.level == "hma"]
    t7(h(out["decadal"]), h(out["periods"]), scheme, alt_b, alt)
    print("cases with |diff| > sigma_total:", int(sens.abs_diff_gt_sigma_total.sum()))
    print(h(out["periods"])[["period", "B_best", "sigma_forcing", "sigma_calib", "sigma_prior", "sigma_trend_base",
                             "sigma_trend", "sigma_family", "sigma_total"]].round(3).to_string(index=False))


if __name__ == "__main__":
    if "--rescale-trend-only" in sys.argv:
        rescale_from_csv()
    else:
        main()
