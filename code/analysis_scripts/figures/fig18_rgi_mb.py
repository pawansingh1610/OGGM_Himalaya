#!/usr/bin/env python
"""Figure 18 - annual area-weighted mass balance 1950-2025 of the three RGI regions (13 Central Asia, 14 South Asia West,
15 South Asia East), one line per forcing; before 1980 shaded (not constrained), 2000-2019 geodetic calibration,
2020-2025 hatched (extrapolation). ERA5-Land from run_07 (corrected Sep 2022 - Feb 2024 precipitation).

Input: analysis/B_regional_series.csv (level == "rgi_region"; columns forcing, unit, year, B_mwe)
"""
import pandas as pd

from style import (AN, COL, DOUBLE, FORC, INK2, LABEL, LS, figure, header, legend_handles, save, shade_periods, zero_line)

header(18, "RGI-region mass balance 1950-2025")
B = pd.read_csv(AN / "B_regional_series.csv")
B = B[B.level == "rgi_region"].copy()
NAME = {"RGI60-13": "(a) Region 13, Central Asia", "RGI60-14": "(b) Region 14, South Asia West",
        "RGI60-15": "(c) Region 15, South Asia East"}
fig = figure(DOUBLE, 13.5)
axs = fig.subplots(3, 1, sharex=True)
for ax, (u, t) in zip(axs, NAME.items()):
    shade_periods(ax)
    for f in FORC:
        d = B[(B.unit == u) & (B.forcing == f) & (B.year >= 1950)]
        d = d[d.year.between(d.dropna(subset=["B_mwe"]).year.min(), d.dropna(subset=["B_mwe"]).year.max())]
        ax.plot(d.year, d.B_mwe, color=COL[f], ls=LS[f])      # NaN (masked ERA5-Land 2022-2024) breaks the line
        c = d[d.year.between(1980, 1999)].B_mwe.mean(); e = d[d.year.between(2000, 2019)].B_mwe.mean()
        print(f"   {u} {LABEL[f]:12s} {int(d.year.min())}-{int(d.year.max())}  mean 1980-1999 {c:+.3f}  2000-2019 {e:+.3f}")
    zero_line(ax)
    ax.set_ylabel("Mass balance (m w.e. yr$^{-1}$)")
    ax.set_title(t)
axs[0].text(1965, axs[0].get_ylim()[1], "1950–1979: not constrained", ha="center", va="top", color=INK2)
axs[0].text(2009.5, axs[0].get_ylim()[1], "geodetic calibration", ha="center", va="top", color=INK2)
axs[0].text(2025.5, axs[0].get_ylim()[1], "extrapolation", ha="right", va="top", color=INK2, fontsize=7)
axs[-1].set_xlim(1948.5, 2026)
fig.legend(handles=legend_handles(), loc="outside lower center", ncol=5)
save(fig, "fig18_rgi_mb")
