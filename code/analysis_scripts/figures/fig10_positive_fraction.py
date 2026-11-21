#!/usr/bin/env python
"""Figure 11 in Draft_2 (file 10) - share of the glacier area with positive annual mass balance, per forcing.

Input: analysis/G_positive_mb_fraction.csv (year, forcing, frac_area)
"""
import pandas as pd

from style import AN, COL, DOUBLE, FORC, LABEL, LS, figure, header, legend_handles, save, shade_periods

header(10, "positive-MB area fraction")
G = pd.read_csv(AN / "G_positive_mb_fraction.csv")
fig = figure(DOUBLE, 6.0)   # full width, longer time axis (author comment on Draft_2 p. 25)
ax = fig.subplots()
shade_periods(ax)
for f in FORC:
    d = G[G.forcing == f]
    ax.plot(d.year, 100 * d.frac_area, color=COL[f], ls=LS[f], lw=1.0)
    per = {p: 100 * d[d.year.between(*p)].frac_area.mean() for p in [(1950, 1979), (1980, 1999), (2000, 2019), (2020, 2025)]}
    print(f"   {LABEL[f]:12s} " + "  ".join(f"{a}-{b}: {v:5.1f} %" for (a, b), v in per.items() if v == v))
ax.set_ylabel("Glacier area with B > 0 (%)")
ax.set_ylim(0, 100); ax.set_xlim(1948.5, 2026)
fig.legend(handles=legend_handles(), loc="outside lower center", ncol=5)
save(fig, "fig10_positive_fraction")
