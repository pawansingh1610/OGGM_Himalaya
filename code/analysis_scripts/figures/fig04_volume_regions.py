#!/usr/bin/env python
"""Figure file 04 (not in the main text at present) - HMA ice volume (km3) with the consensus-thickness band,
and volume change relative to 1999 (% of V_1999) for RGI regions 13, 14, 15.

Inputs: analysis/B_regional_series.csv (levels hma, rgi_region; V), analysis/B_ensemble_series.csv (V_low_min, V_high_max)
"""
import pandas as pd

from style import (AN, COL, DOUBLE, FORC, LABEL, LS, SHADE_CAL, figure, header, legend_handles, save,
                   shade_periods, zero_line)

header(4, "volume, HMA and RGI regions")
B = pd.read_csv(AN / "B_regional_series.csv")
BE = pd.read_csv(AN / "B_ensemble_series.csv")
fig = figure(DOUBLE, 11.0)
axs = fig.subplots(2, 2, sharex=True)
ax = axs[0, 0]
e = BE[BE.level == "hma"]
ax.fill_between(e.year, e.V_low_min, e.V_high_max, color=SHADE_CAL, lw=0, label="consensus thickness range")
for f in FORC:
    d = B[(B.level == "hma") & (B.forcing == f)]
    ax.plot(d.year, d.V, color=COL[f], ls=LS[f])
    print(f"   HMA {LABEL[f]:12s} V_1999 {d[d.year == 1999].V.iloc[0]:7.0f} km3  V_last {d.dropna(subset=['V']).V.iloc[-1]:7.0f} ({int(d.dropna(subset=['V']).year.iloc[-1])})")
print(f"   consensus band 1999: {e[e.year == 1999].V_low_min.iloc[0]:.0f}-{e[e.year == 1999].V_high_max.iloc[0]:.0f} km3")
ax.set_ylabel("Ice volume (km$^3$)"); ax.set_title("(a) HMA ice volume")
ax.legend(loc="lower left")
r = B[B.level == "rgi_region"]
names = {"RGI60-13": "(b) RGI 13 Central Asia", "RGI60-14": "(c) RGI 14 South Asia West",
         "RGI60-15": "(d) RGI 15 South Asia East"}
for ax, reg in zip([axs[0, 1], axs[1, 0], axs[1, 1]], names):
    shade_periods(ax)
    d0 = r[r.unit == reg]
    for f in FORC:
        d = d0[d0.forcing == f]
        v = d.loc[d.year == 1999, "V"]
        if v.empty:
            continue
        p = 100 * (d.V - v.iloc[0]) / v.iloc[0]
        ax.plot(d.year, p, color=COL[f], ls=LS[f])
        print(f"   {reg} {LABEL[f]:12s} dV to {int(d.year[p.notna() & (d.year <= 2019)].max())} {p[p.notna() & (d.year <= 2019)].iloc[-1]:+.2f} %  first year {p.dropna().iloc[0]:+.2f} %")
    zero_line(ax); ax.set_title(names[reg]); ax.set_ylabel("ΔV (% of $V_{1999}$)")
for ax in axs[1]:
    ax.set_xlim(1948.5, 2026)
fig.legend(handles=legend_handles(), loc="outside lower center", ncol=5)
save(fig, "fig04_volume_regions")
