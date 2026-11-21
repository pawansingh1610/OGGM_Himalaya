#!/usr/bin/env python
"""Check of the modelled 2000-2023 mass balance (and the 2020-2025 extrapolation) against GlaMBIE (2024 dataset,
GlaMBIE Team 2025) for RGI regions 13, 14, 15 and HMA, plus the field glaciers 2020-2024.

GlaMBIE: calendar-year results are the main comparison (the model uses calendar years); the hydrological-year
results (Oct-Sep, offset by about 3 months) are shown as a sensitivity. GlaMBIE ends in 2023. HMA = sum of the
three regions (Gt, area); its error = regional errors in quadrature (independent regions).
Period means of GlaMBIE errors assume independent annual errors (RSS / n) - stated in the table note.
Outputs: tables/glambie_annual.csv, tables/glambie_periods.csv, tables/glambie_field_2020s.csv,
tables/T8_glambie_check.tex, figures/F14_glambie_check.png.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_figures import (FORC, LABEL, COL, LS, INK, INK2, MUTED, AXIS, DOUBLE, write_table, fmt, save,  # noqa: E402
                          TAB, legend_handles)
import matplotlib.pyplot as plt  # noqa: E402

AN = ROOT / "analysis"
GL = ROOT / "data_external/glambie/raw/unz/glambie_results_20240716"
REG = {"RGI60-13": "13_central_asia", "RGI60-14": "14_south_asia_west", "RGI60-15": "15_south_asia_east"}
NAME = {"RGI60-13": "Central Asia (13)", "RGI60-14": "South Asia West (14)", "RGI60-15": "South Asia East (15)",
        "ALL_HMA": "HMA (13+14+15)"}
UNITS = ["RGI60-13", "RGI60-14", "RGI60-15", "ALL_HMA"]
PERIODS = {"2000-2019": (2000, 2019), "2020-2023": (2020, 2023), "2000-2023": (2000, 2023)}


def load_glambie(kind):
    out = []
    for r, f in REG.items():
        d = pd.read_csv(GL / kind / f"{f}.csv")
        d["year"] = np.floor(d.start_dates + 0.3).astype(int)   # calendar: start 2000.0 -> 2000; hydro: 1999.75 -> 2000
        d["unit"] = r
        out.append(d[["unit", "year", "start_dates", "end_dates", "glacier_area", "combined_gt", "combined_gt_errors",
                      "combined_mwe", "combined_mwe_errors"]])
    g = pd.concat(out)
    h = g.groupby("year").agg(glacier_area=("glacier_area", "sum"), combined_gt=("combined_gt", "sum"),
                              combined_gt_errors=("combined_gt_errors", lambda e: np.sqrt((e ** 2).sum())),
                              n=("unit", "count")).reset_index()
    h = h[h.n == 3].drop(columns="n")
    h["combined_mwe"] = h.combined_gt / h.glacier_area * 1000
    h["combined_mwe_errors"] = h.combined_gt_errors / h.glacier_area * 1000
    h["unit"] = "ALL_HMA"
    return pd.concat([g, h], ignore_index=True)


def main():
    gc, gh = load_glambie("calendar_years"), load_glambie("hydrological_years")
    B = pd.read_csv(AN / "B_regional_series.csv")
    B = B[B.level.isin(["hma", "rgi_region"]) & B.year.between(2000, 2025)]
    best = pd.read_csv(TAB / "best_mb_annual.csv")
    best = best[best.level.isin(["hma", "rgi_region"]) & best.year.between(2000, 2025)]

    # ---- annual table (model members + best + GlaMBIE calendar + hydrological)
    mod = B.pivot_table(index=["unit", "year"], columns="forcing", values="B_mwe")
    modg = B.pivot_table(index=["unit", "year"], columns="forcing", values="dm_gt")
    ann = mod.copy()
    ann.columns = [f"B_{c}" for c in ann.columns]
    for c in modg.columns: ann[f"dm_gt_{c}"] = modg[c]
    b = best.set_index(["unit", "year"])
    ann["B_best"], ann["dm_gt_best"], ann["sigma_total_best"] = b.B_best, b.dM_Gt_best, b.sigma_total
    for tag, g in [("cal", gc), ("hyd", gh)]:
        g = g.set_index(["unit", "year"])
        ann[f"glambie_{tag}_mwe"], ann[f"glambie_{tag}_mwe_err"] = g.combined_mwe, g.combined_mwe_errors
        ann[f"glambie_{tag}_gt"], ann[f"glambie_{tag}_gt_err"] = g.combined_gt, g.combined_gt_errors
    ann = ann.reset_index()
    ann.to_csv(TAB / "glambie_annual.csv", index=False)

    # ---- period means / cumulative
    rows = []
    for u in UNITS:
        a = ann[ann.unit == u].set_index("year")
        for p, (y0, y1) in PERIODS.items():
            s = a.loc[y0:y1]
            n = y1 - y0 + 1
            r = dict(unit=u, period=p, n_years=n)
            for src in [f"B_{f}" for f in FORC] + ["B_best", "glambie_cal_mwe", "glambie_hyd_mwe"]:
                v = s[src]
                r[f"{src}_mean"] = v.mean() if v.notna().sum() == n else np.nan
            for src in [f"dm_gt_{f}" for f in FORC] + ["dm_gt_best", "glambie_cal_gt"]:
                v = s[src]
                r[f"{src}_cum"] = v.sum() if v.notna().sum() == n else np.nan
            r["glambie_cal_mwe_err_mean"] = np.sqrt((s.glambie_cal_mwe_err ** 2).sum()) / n
            r["glambie_cal_gt_err_cum"] = np.sqrt((s.glambie_cal_gt_err ** 2).sum())
            r["glambie_hyd_mwe_err_mean"] = np.sqrt((s.glambie_hyd_mwe_err ** 2).sum()) / n
            r["best_minus_glambie_mean"] = r["B_best_mean"] - r["glambie_cal_mwe_mean"]
            rows.append(r)
    P = pd.DataFrame(rows)
    # best estimate per period: members averaged over the period first, then combined (same as T7)
    bp = pd.read_csv(TAB / "best_mb_periods.csv").set_index(["unit", "period"]).B_best
    P["B_best_mean"] = [bp.get((u, p), np.nan) for u, p in zip(P.unit, P.period)]
    P["best_minus_glambie_mean"] = P.B_best_mean - P.glambie_cal_mwe_mean
    P.to_csv(TAB / "glambie_periods.csv", index=False)
    pd.set_option("display.width", 250)
    show = ["unit", "period"] + [f"B_{f}_mean" for f in FORC] + ["B_best_mean", "glambie_cal_mwe_mean",
                                                                  "glambie_cal_mwe_err_mean", "glambie_hyd_mwe_mean",
                                                                  "best_minus_glambie_mean"]
    print(P[show].round(3).to_string(index=False))
    print(P[["unit", "period", "dm_gt_best_cum", "glambie_cal_gt_cum", "glambie_cal_gt_err_cum"]].round(1).to_string(index=False))

    # annual 2020-2025 HMA detail
    h = ann[(ann.unit == "ALL_HMA") & (ann.year >= 2016)]
    print(h[["year"] + [f"B_{f}" for f in ["ERA5", "ERA5-Land", "HARv2"]] + ["B_best", "glambie_cal_mwe",
                                                                             "glambie_cal_mwe_err", "glambie_hyd_mwe"]].round(3).to_string(index=False))

    # ---- field glaciers: 2010-2019 vs 2020-2024 on glaciers with data in both
    pairs = pd.read_csv(AN / "D_validation_pairs.csv").dropna(subset=["rgi_id"])
    pairs["per"] = np.where(pairs.year.between(2010, 2019), "2010-2019", np.where(pairs.year.between(2020, 2024), "2020-2024", None))
    pp = pairs[pairs.per.notna()]
    both = pp.groupby("rgi_id").per.nunique()
    pp = pp[pp.rgi_id.isin(both[both == 2].index)]
    fcols = ["obs", "ERA5", "ERA5-Land", "HARv2"]
    g = pp.groupby(["rgi_id", "per"])[fcols].mean().unstack("per")
    F = pd.DataFrame({c: {"2010-2019": g[(c, "2010-2019")].mean(), "2020-2024": g[(c, "2020-2024")].mean(),
                          "change": (g[(c, "2020-2024")] - g[(c, "2010-2019")]).mean()} for c in fcols}).T
    F["n_glaciers"] = len(g)
    F["n_years_2020s"] = pp[pp.per == "2020-2024"].shape[0]
    yr = pp[pp.per == "2020-2024"].groupby("year")[fcols].mean()
    yr["n"] = pp[pp.per == "2020-2024"].groupby("year").size()
    F.to_csv(TAB / "glambie_field_2020s.csv")
    yr.to_csv(TAB / "glambie_field_2020s_by_year.csv")
    print("field glaciers with data in 2010-2019 and 2020-2024 (glacier means, then mean over glaciers):\n", F.round(3).to_string())
    print(yr.round(3).to_string())

    t8(P, F)
    # figure: scripts/figures/fig14_glambie.py (reads the tables written here)


def t8(P, F):
    rows = []
    srcs = [("glambie_cal_mwe", "GlaMBIE (calendar)"), ("glambie_hyd_mwe", "GlaMBIE (hydrological)"),
            ("B_best", "Best estimate")] + [(f"B_{f}", LABEL[f]) for f in FORC]
    for k, u in enumerate(UNITS):
        if k: rows.append("MID")
        d = P[P.unit == u].set_index("period")
        for i, (s, lab) in enumerate(srcs):
            cells = []
            for p in ["2000-2019", "2020-2023", "2000-2023"]:
                v = d.loc[p, f"{s}_mean"]
                if s == "glambie_cal_mwe" and pd.notna(v):
                    cells.append(f"{fmt(v, 3)} $\\pm$ {d.loc[p, 'glambie_cal_mwe_err_mean']:.3f}")
                else:
                    cells.append(fmt(v, 3))
            cum = d.loc["2000-2023", "glambie_cal_gt_cum"] if s == "glambie_cal_mwe" else (
                d.loc["2000-2023", "dm_gt_best_cum"] if s == "B_best" else (
                    d.loc["2000-2023", f"dm_gt_{s[2:]}_cum"] if s.startswith("B_") else np.nan))
            cumtxt = ("--" if pd.isna(cum) else fmt(cum, 0)) + (
                f" $\\pm$ {d.loc['2000-2023', 'glambie_cal_gt_err_cum']:.0f}" if s == "glambie_cal_mwe" else "")
            rows.append(f"{NAME[u] if i == 0 else ''} & {lab} & " + " & ".join(cells) + f" & {cumtxt}")
    write_table("T8_glambie_check", "Modelled specific mass balance (m w.e. yr$^{-1}$, period means) and cumulative mass change "
                "2000--2023 (Gt) against GlaMBIE (GlaMBIE Team, 2025; dataset GlaMBIE, 2024), calendar years. GlaMBIE errors are "
                "95\\,\\% confidence intervals; period means assume independent annual errors. HMA is the sum of the three "
                "regions. CRU TS4 ends in 2019 and CHELSA-W5E5 in 2016.",
                "tab:glambie", "llrrrr",
                ["Region & Source & 2000--2019 & 2020--2023 & 2000--2023 & 2000--2023 (Gt)"], rows, star=True, size="\\scriptsize")


if __name__ == "__main__":
    main()
