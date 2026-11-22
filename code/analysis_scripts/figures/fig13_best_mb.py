#!/usr/bin/env python
"""Figure file 13 - best-estimate HMA mass balance with +-sigma_forcing and +-sigma_total
and the five members (a); the four uncertainty components and the total (b).

Inputs: tables/best_mb_annual.csv (level == hma; year, B_best, sigma_*), analysis/B_regional_series.csv (members)
"""
import pandas as pd

from style import (AN, COL, DOUBLE, FORC, INK, INK2, LS, TAB, figure, header, legend_handles, save, shade_periods,
                   zero_line)

header(13, "best-estimate mass balance")
a = pd.read_csv(TAB / "best_mb_annual.csv")
a = a[a.level == "hma"].sort_values("year")
scheme = a.scheme.iloc[0]
B = pd.read_csv(AN / "B_regional_series.csv")
Bh = B[B.level == "hma"]
for p in [(1950, 1979), (1980, 1999), (2000, 2019), (2020, 2025)]:
    d = a[a.year.between(*p)]
    print(f"   {p[0]}-{p[1]}: annual B_best mean {d.B_best.mean():+.3f}; mean sigma_total {d.sigma_total.mean():.3f}; "
          f"mean sigma forcing/calib/prior/trend {d.sigma_forcing.mean():.3f}/{d.sigma_calib.mean():.3f}/"
          f"{d.sigma_prior.mean():.3f}/{d.sigma_trend.mean():.3f}")
print(f"   scheme: {scheme}; years {int(a.year.min())}-{int(a.year.max())}")

fig = figure(DOUBLE, 11.5)
axs = fig.subplots(2, 1, sharex=True, height_ratios=[1.5, 1])
ax = axs[0]
shade_periods(ax)
for f in FORC:
    d = Bh[(Bh.forcing == f) & Bh.B_mwe.notna()]
    ax.plot(d.year, d.B_mwe, color=COL[f], ls=LS[f], lw=0.75, alpha=0.8)
ax.fill_between(a.year, a.B_best - a.sigma_total, a.B_best + a.sigma_total, color="#c9c8c2", lw=0, label="±σ total", zorder=1)
ax.fill_between(a.year, a.B_best - a.sigma_forcing.fillna(0), a.B_best + a.sigma_forcing.fillna(0), color="#9a9892",
                lw=0, label="±σ forcing", zorder=1)
ax.plot(a.year, a.B_best, color=INK, lw=1.6, label="best estimate", zorder=4)
zero_line(ax)
ax.set_ylabel("Mass balance (m w.e. yr$^{-1}$)")
ax.set_title("(a) Best-estimate HMA mass balance and members")
yt = ax.get_ylim()[1]
for x, s in [(1965, "not constrained"), (2009.5, "geodetic calibration"), (2022.5, "extrapolation")]:
    ax.text(x, yt, s, ha="center", va="top", color=INK2, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
h2, _ = ax.get_legend_handles_labels()
ax.legend(handles=h2 + legend_handles(lw=0.9), ncol=4, loc="lower left", fontsize=7)
ax = axs[1]
shade_periods(ax)
comp = [("sigma_forcing", "σ forcing", "#2a78d6", "-"), ("sigma_calib", "σ calib", "#eda100", (0, (4, 1.6))),
        ("sigma_prior", "σ prior", "#e87ba4", (0, (5, 1.5, 1, 1.5))), ("sigma_trend", "σ trend", "#1baf7a", (0, (1, 1.2))),
        ("sigma_family", "σ family", "#7a4fd0", (0, (3, 1, 1, 1)))]
for c, lab, col, ls in comp:
    ax.plot(a.year, a[c], color=col, ls=ls, lw=1.2, label=lab)
ax.plot(a.year, a.sigma_total, color=INK, lw=1.4, label="σ total")
ax.set_ylabel("1σ (m w.e. yr$^{-1}$)"); ax.set_ylim(bottom=-0.035); ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
ax.set_title("(b) Uncertainty components (total = quadrature sum)")
ax.legend(ncol=6, loc="upper right", fontsize=6.5)
ax.set_xlim(1948.5, 2026)
# member-set changes (CHELSA-W5E5 enters 1979, HARv2 1980; CHELSA-W5E5 leaves after 2016, CRU TS4 after 2019): steps in the ensemble up to 0.076 m w.e. yr-1
for x in (1978.5, 1979.5, 2016.5, 2019.5):
    for axx in axs:
        axx.axvline(x, color=INK2, lw=0.5, ls=(0, (1, 2)), zorder=0.5)
for x, n in ((1964, 3), (1998, 5), (2022.7, 3)):
    axs[1].text(x, -0.03, f"n = {n}", ha="center", va="bottom", color=INK2, fontsize=6)
save(fig, "fig13_best_mb")
