#!/usr/bin/env python
"""Figure 3 - HMA area-weighted annual mass balance (a) and cumulative ice-volume change relative to the end
of 1999 in % of each forcing's own V_1999 (b), five forcings, 1950-2025.

Input: analysis/B_regional_series.csv (level == "hma"; columns forcing, year, B_mwe, V)
"""
import pandas as pd

from style import (AN, COL, DOUBLE, FORC, INK2, LABEL, LS, figure, header, legend_handles, save,
                   shade_periods, zero_line)

header(3, "HMA mass balance and cumulative volume change")
B = pd.read_csv(AN / "B_regional_series.csv")
h = B[B.level == "hma"].copy()
v99 = h[h.year == 1999].set_index("forcing").V
h["dV_pct"] = 100 * (h.V - h.forcing.map(v99)) / h.forcing.map(v99)
h["dV_km3"] = h.V - h.forcing.map(v99)
print("   V_1999 (km3):", v99.round(0).to_dict())
for f in FORC:
    d = h[(h.forcing == f) & h.B_mwe.notna()]
    print(f"   {LABEL[f]:12s} years {int(d.year.min())}-{int(d.year.max())}  B mean 1980-1999 "
          f"{d[d.year.between(1980, 1999)].B_mwe.mean():+.3f}  2000-2019 {d[d.year.between(2000, 2019)].B_mwe.mean():+.3f}  "
          f"dV(%V1999) first {h[(h.forcing == f) & (h.year == d.year.min() - 1)].dV_pct.iloc[0]:+.2f}  last "
          f"{h[(h.forcing == f) & (h.year == d.year.max())].dV_pct.iloc[0]:+.2f} (km3 {h[(h.forcing == f) & (h.year == d.year.max())].dV_km3.iloc[0]:+.0f})")

fig = figure(DOUBLE, 11.0)
axs = fig.subplots(2, 1, sharex=True)
ax = axs[0]
shade_periods(ax)
for f in FORC:
    d = h[(h.forcing == f) & h.B_mwe.notna()]
    ax.plot(d.year, d.B_mwe, color=COL[f], ls=LS[f])
zero_line(ax)
ax.set_ylabel("Mass balance (m w.e. yr$^{-1}$)")
ax.set_title("(a) Area-weighted annual mass balance, HMA")
yt = ax.get_ylim()[1]
for x, s in [(1965, "1950–1979: not constrained"), (2009.5, "geodetic calibration"), (2022.5, "extrapolation")]:
    ax.text(x, yt, s, ha="center", va="top", color=INK2,
            bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
ax = axs[1]
shade_periods(ax)
for f in FORC:
    d = h[h.forcing == f]
    ax.plot(d.year, d.dV_pct, color=COL[f], ls=LS[f])
zero_line(ax)
ax.set_ylabel("Volume change (% of $V_{1999}$)")
ax.set_title("(b) Cumulative ice-volume change relative to the end of 1999")
ax.set_xlim(1948.5, 2026)
fig.legend(handles=legend_handles(), loc="outside lower center", ncol=5)
save(fig, "fig03_hma_mb")
