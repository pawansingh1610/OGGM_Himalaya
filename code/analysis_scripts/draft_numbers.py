#!/usr/bin/env python
"""Every model-derived number quoted in the paper, computed from analysis/ and tables/.

Usage:  python draft_numbers.py [--an DIR] [--tab DIR] [--out FILE]
Default dirs are analysis/ and tables/ of the working folder.
Output: one line per key, "key<TAB>value" (raw, unrounded). Keys are grouped by the paper section that quotes them.
Observation-only numbers (field data, Hugonnet, GlaMBIE, population) are not listed.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
F5 = ["ERA5", "ERA5-Land", "HARv2", "CRU_TS4", "CHELSA"]
EF = ["ERA5", "ERA5-Land", "HARv2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--an", default=str(ROOT / "analysis"))
    ap.add_argument("--tab", default=str(ROOT / "tables"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--meta", action="store_true", help="also area-factor stats from per-glacier metadata")
    a = ap.parse_args()
    AN, TAB = Path(a.an), Path(a.tab)
    out = {}

    def put(k, v):
        out[k] = v

    def rng(prefix, s):
        put(prefix + ".min", float(np.nanmin(s))); put(prefix + ".max", float(np.nanmax(s)))

    # --- 4.1 calibration (T2, A) ---
    T2 = pd.read_csv(TAB / "T2_calibration.csv").set_index("forcing")
    for f in F5:
        for c in ["pct_ok", "pct_area_unreachable", "median_dT", "median_pf", "pct_tb_bound", "pct_pf_bound",
                  "pct_backward_vanished", "median_elev_offset_m"]:
            put(f"calib.{c}.{f}", T2.loc[f, c])
        put(f"calib.max_bound.{f}", max(T2.loc[f, "pct_tb_bound"], T2.loc[f, "pct_pf_bound"]))
    A = pd.read_csv(AN / "A_calibration_diagnostics.csv")
    ch = A[A.forcing == "CHELSA"].set_index("subregion")
    for s in ["West Kun Lun", "East Kun Lun (Altyn Tagh)", "East Tien Shan (Dzhungaria)", "Karakoram"]:
        put(f"calib.chelsa_unreach.{s}", ch.loc[s, "pct_area_unreachable"])
    rep = (AN / "REPORT.txt").read_text().splitlines()
    i = next(j for j, l in enumerate(rep) if "area_excluded_pct" in l)
    for l in rep[i + 1:i + 6]:
        p = l.split()
        put(f"calib.area_excluded_pct.{p[0]}", float(p[-1]))
    S6 = pd.read_csv(TAB / "S6_backward_vanished.csv").set_index("forcing")
    for c in ["n", "pct_glaciers", "area_km2", "pct_area", "median_area_km2", "pct_lt_0p5km2", "median_geod_target"]:
        put(f"bv.ERA5.{c}", S6.loc["ERA5", c])

    # --- 4.2 volume windows (% of V_1999) ---
    W = pd.read_csv(AN / "C_volume_change_windows.csv")
    S = pd.read_csv(AN / "B_regional_series.csv")
    v99 = S[S.year == 1999].set_index(["forcing", "level", "unit"]).V.rename("V99")
    W = W.join(v99, on=["forcing", "level", "unit"])
    W["pct"] = W.dV_km3 / W.V99 * 100
    h = W[W.level == "hma"].set_index(["window", "forcing"])
    for (w, f), r in h.iterrows():
        put(f"vol.{w}.{f}.km3", r.dV_km3); put(f"vol.{w}.{f}.Gt", r.dM_Gt); put(f"vol.{w}.{f}.pct", r.pct)
    for w in h.index.get_level_values(0).unique():
        g = h.loc[w]
        rng(f"vol.{w}.km3", g.dV_km3); rng(f"vol.{w}.Gt", g.dM_Gt); rng(f"vol.{w}.pct", g.pct)
        put(f"vol.{w}.spread_pp", g.pct.max() - g.pct.min())
    g = h.loc["fwd_2000-2019"]
    rng("vol.fwd_2000-2019.Gt_per_yr", g.dM_Gt / 20)
    g = h.loc["fwd_2000-2025"].loc[EF]
    rng("vol.fwd_2000-2025.EF.km3", g.dV_km3); rng("vol.fwd_2000-2025.EF.pct", g.pct)
    for lev in ["rgi_region", "subregion"]:
        x = W[W.level == lev]
        for (w, u), g in x.groupby(["window", "unit"]):
            rng(f"{lev}.{w}.{u}.pct", g.pct)
            for _, r in g.iterrows():
                put(f"{lev}.{w}.{u}.{r.forcing}.pct", r.pct)

    # --- 4.3 split test ---
    E = pd.read_csv(AN / "E_split_sample.csv")
    eh = E[E.subregion.isin(["ALL_HMA", "HMA", "ALL"])].set_index("forcing")
    put("split.dObs_HMA", eh.dObs_areaw.iloc[0])
    for f in eh.index:
        put(f"split.share.{f}", 100 * eh.loc[f, "dMod_areaw"] / eh.loc[f, "dObs_areaw"])
    rng("split.r_glacier", eh.r_glacier_ge1km2)
    es = E.set_index(["subregion", "forcing"])
    for s in ["West Himalaya", "Hindu Kush", "East Himalaya"]:
        for f in ["ERA5", "ERA5-Land", "HARv2", "CRU_TS4"]:
            put(f"split.misfit.{s}.{f}", es.loc[(s, f), "dMod_minus_dObs"])
    T4 = pd.read_csv(TAB / "T4_validation.csv")
    t4 = T4[T4.subset == "common_all5"].set_index("forcing")
    for f in F5:
        for c in ["bias", "RMSE", "r_anomaly"]:
            put(f"field.{c}.{f}", t4.loc[f, c])
    put("field.common.N", t4.N.iloc[0]); put("field.common.n_glaciers", t4.n_glaciers.iloc[0])
    rng("field.common.bias", t4.bias); rng("field.common.RMSE", t4.RMSE); rng("field.common.r_anomaly", t4.r_anomaly)
    D = pd.read_csv(AN / "D_validation_decadal.csv")
    for _, r in D[D.decade == 1970].iterrows():
        put(f"field.1970s.bias.{r.forcing}", r.bias); put(f"field.1970s.N.{r.forcing}", r.N)

    # --- 4.4 best estimate ---
    Wt = pd.read_csv(TAB / "best_mb_weights.csv", index_col=0)
    for f in Wt.index:
        put(f"best.w.{f}", Wt.loc[f, "w_weighted"]); put(f"best.split_skill.{f}", Wt.loc[f, "split_skill"])
    for fam, g in Wt.groupby("family"):
        put(f"best.wfam.{fam}", g.w_weighted.sum())
    put("best.chosen", Wt.chosen_scheme.iloc[0])
    L = pd.read_csv(TAB / "best_mb_loo.csv", index_col=[0, 1])
    for (t, sch), r in L.iterrows():
        for c in ["RMSE", "r_anomaly", "MAE_areaw"]:
            if pd.notna(r.get(c)):
                put(f"best.loo.{t}.{sch}.{c}", r[c])
    for fn, key in [("best_mb_decadal.csv", "dec"), ("best_mb_periods.csv", "per"), ("best_mb_pentad.csv", "pent")]:
        P = pd.read_csv(TAB / fn)
        P = P[P.scheme == "equal_family"] if "scheme" in P else P
        for _, r in P[P.level.isin(["hma", "rgi_region"])].iterrows():
            for c in ["B_best", "sigma_forcing", "sigma_calib", "sigma_prior", "sigma_trend", "sigma_total"]:
                put(f"best.{key}.{r.unit}.{r.period}.{c}", r[c])
    P = pd.read_csv(TAB / "best_mb_decadal.csv"); P = P[(P.level == "hma") & (P.scheme == "equal_family")]
    pre80 = P[P.y1 < 1980]
    put("best.dec.pre1980.max_sigma_forcing", pre80.sigma_forcing.max())
    pre00 = P[P.y1 < 2000]
    rng("best.dec.pre2000.sigma_prior", pre00.sigma_prior)
    Ss = pd.read_csv(TAB / "best_mb_scheme_sensitivity.csv")
    d = Ss[(Ss.resolution == "decadal") & (Ss.level == "hma")]
    if not d.empty:
        d = d.assign(y0=d.period.str[:4].astype(int))
        put("best.scheme.dec.maxdiff", d["diff"].abs().max())
        put("best.scheme.dec.maxdiff_period", d.loc[d["diff"].abs().idxmax(), "period"])
        put("best.scheme.dec.maxdiff_post1980", d[d.y0 >= 1980]["diff"].abs().max())
    put("best.scheme.any_gt_sigma_total", bool(Ss.abs_diff_gt_sigma_total.any()))
    H = pd.read_csv(TAB / "best_mb_vs_hugonnet.csv").set_index("unit")
    for u in H.index:
        put(f"best.vs_hug.{u}.B_best", H.loc[u, "B_best_2000_2019"]); put(f"best.vs_hug.{u}.diff", H.loc[u, "diff"])

    # --- 4.5 GlaMBIE ---
    G = pd.read_csv(TAB / "glambie_periods.csv").set_index(["unit", "period"])
    for (u, p), r in G.iterrows():
        put(f"glambie.{u}.{p}.B_best", r.B_best_mean); put(f"glambie.{u}.{p}.cum_best", r.dm_gt_best_cum)
        put(f"glambie.{u}.{p}.best_minus_glambie", r.best_minus_glambie_mean)
        for f in F5:
            put(f"glambie.{u}.{p}.B_{f}", r[f"B_{f}_mean"])
    Ga = pd.read_csv(TAB / "glambie_annual.csv")
    ga = Ga[Ga.unit == "ALL_HMA"].set_index("year")
    for y in [2020, 2022, 2024, 2025]:
        if y in ga.index:
            put(f"glambie.annual.HMA.{y}.B_best", ga.loc[y, "B_best"])
            for f in EF:
                put(f"glambie.annual.HMA.{y}.B_{f}", ga.loc[y, f"B_{f}"])
    Gf = pd.read_csv(TAB / "glambie_field_2020s.csv", index_col=0)
    for f in EF:
        put(f"glambie.field.change.{f}", Gf.loc[f, "change"])
    # best estimate 2020-2025 and 2020-2023 from periods table (if present) or annual
    BA = pd.read_csv(TAB / "best_mb_annual.csv")
    ba = BA[(BA.level == "hma") & (BA.scheme == "equal_family")].set_index("year")
    put("best.ann.2020-2025.mean", ba.loc[2020:2025, "B_best"].mean())
    put("best.ann.2020-2023.mean", ba.loc[2020:2023, "B_best"].mean())
    for y in [2024, 2025]:
        put(f"best.ann.{y}", ba.loc[y, "B_best"])
    P = pd.read_csv(TAB / "best_mb_periods.csv"); P = P[(P.level == "hma") & (P.scheme == "equal_family")]
    for _, r in P.iterrows():
        put(f"best.per.{r.period}.sigma_forcing", r.sigma_forcing)

    # --- 4.6 extinction, positive MB area ---
    S5 = pd.read_csv(TAB / "S5_extinction.csv").set_index("set")
    for s in S5.index:
        for c in ["to_2019", "to_2024", "to_2025"]:
            put(f"ext.{s}.{c}", S5.loc[s, c])
    rng("ext.forcings.to_2024", S5.loc[EF, "to_2024"]); rng("ext.forcings.to_2025", S5.loc[EF, "to_2025"])
    Fy = pd.read_csv(AN / "F_extinction_by_year.csv").set_index("year")
    put("ext.cons.2010-2014", Fy.loc[2010:2014, "consensus"].sum())
    put("ext.cons.2015-2019", Fy.loc[2015:2019, "consensus"].sum())
    put("ext.cons.2020-2024", Fy.loc[2020:2024, "consensus"].sum())
    put("ext.cons.2025", Fy.loc[2025, "consensus"])
    put("ext.cons.pre2010", Fy.loc[:2009, "consensus"].sum())
    Fx = pd.read_csv(AN / "F_extinct_glaciers.csv")
    fx = Fx[Fx.consensus_year <= 2025]
    put("ext.median_area", fx.A_ref_km2.median()); put("ext.pct_lt_0p05", 100 * (fx.A_ref_km2 < 0.05).mean())
    put("ext.n_listed", len(fx))
    Gp = pd.read_csv(AN / "G_positive_mb_fraction.csv")
    for (p0, p1) in [(1950, 1979), (1980, 1999), (2000, 2019), (2020, 2025)]:
        for f, g in Gp.groupby("forcing"):
            s = g[g.year.between(p0, p1)]
            if len(s) == p1 - p0 + 1:
                put(f"posfrac.{p0}-{p1}.{f}", 100 * s.frac_area.mean())
    s = Gp[(Gp.forcing == "CHELSA") & Gp.year.between(2000, 2016)]
    put("posfrac.2000-2016.CHELSA", 100 * s.frac_area.mean())

    # --- 4.2 spatial pattern: cells where the forcings disagree on the sign of the 1980-1999 mass balance ---
    Gm = pd.read_csv(TAB / "gridded_mb_1980_1999.csv")
    put("grid.8099.n_cells", len(Gm)); put("grid.8099.n_sign_disagree", int(Gm.sign_disagree.sum()))
    put("grid.8099.median_absB.sign_disagree", Gm.B_best[Gm.sign_disagree].abs().median())
    put("grid.8099.median_absB.sign_agree", Gm.B_best[~Gm.sign_disagree].abs().median())

    # --- 4.7 basins ---
    B = pd.read_csv(TAB / "T6_basin_volume_by_forcing.csv")
    for (b, w), g in B.groupby(["basin", "window"]):
        rng(f"basin.{b}.{w}.pct", g.dV_pct_V1999); rng(f"basin.{b}.{w}.km3", g.dV_km3)
        put(f"basin.{b}.{w}.n", len(g))

    # --- area-consistent factor (per-glacier metadata of the run in paths.py; only with --meta) ---
    if a.meta:
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from paths import PER_FORCING
        for f in F5:
            m = pd.read_csv(PER_FORCING / f / "metadata.csv", usecols=["area_factor", "area_factor_flag", "A_ref_km2"])
            cap = m.area_factor_flag != "ok"
            put(f"factor.{f}.median", m.area_factor.median())
            put(f"factor.{f}.areaw", np.average(m.area_factor, weights=m.A_ref_km2))
            put(f"factor.{f}.n_capped", int(cap.sum()))
            put(f"factor.{f}.pct_area_capped", 100 * m.A_ref_km2[cap].sum() / m.A_ref_km2.sum())

    lines = []
    for k, v in out.items():
        lines.append(f"{k}\t{v}")
    txt = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(txt)
    else:
        print(txt, end="")


if __name__ == "__main__":
    main()
