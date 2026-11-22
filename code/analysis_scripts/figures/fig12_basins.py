#!/usr/bin/env python
"""Figure file 12 - glacier-fed river basins: (a-c) glacier volume change per forcing in % of the
basin's V_1999 for 1980-1999, 2000-2019, 2000-2025; (d) population change (%) 1975-2000 and 2000-2020 for the
whole basin (bars) and the glacierised level-7 units (ticks); 2020 basin population printed on the right.

Inputs (written by scripts/population_basins.py):
  tables/T6_basin_glacier_population.csv  (basin, glacier_area_km2, pop2020, dpop*_pct, dpop*_glac_l7_pct)
  tables/T6_basin_volume_by_forcing.csv   (basin, forcing, window, dV_pct_V1999)
"""
import numpy as np
import pandas as pd

from style import (COL, DOUBLE, FORC, HATCH_EXT, INK, INK2, LABEL, LW_THIN, TAB, Line2D, figure, header, save,
                   zero_line)

header(12, "basins: volume and population")
T = pd.read_csv(TAB / "T6_basin_glacier_population.csv").set_index("basin")
W = pd.read_csv(TAB / "T6_basin_volume_by_forcing.csv")
main = T[(T.glacier_area_km2 > 0) | (T.index == "GBM delta & Meghna")]
order = main.sort_values("glacier_area_km2", ascending=False).index.tolist()
print(main.loc[order, ["glacier_area_km2", "pop2020", "dpop1975_2000_pct", "dpop2000_2020_pct",
                       "dpop1975_2000_glac_l7_pct", "dpop2000_2020_glac_l7_pct"]].round(1).to_string())
print("   volume change range over forcings (% of V_1999):")
print(W.groupby(["basin", "window"]).dV_pct_V1999.agg(["min", "max", "count"]).unstack("window").reindex(order).round(1).to_string())

y = np.arange(len(order))
LBL = {"Tibetan Plateau & Qaidam endorheic": "Tibetan Pl. & Qaidam endorh.",
       "Other Central Asian endorheic": "Other C. Asian endorheic"}
fig = figure(DOUBLE, 10.0)
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.55])
wins = [("1980-1999", "(a) 1980–1999"), ("2000-2019", "(b) 2000–2019"), ("2000-2025", "(c) 2000–2025")]
axs = []
for k, (w, title) in enumerate(wins):
    ax = fig.add_subplot(gs[0, k], sharey=axs[0] if axs else None)
    axs.append(ax)
    if w == "2000-2025":
        ax.axvspan(-60, 15, facecolor="none", edgecolor=HATCH_EXT, hatch="////", lw=0, zorder=0)
    for i, b in enumerate(order):
        d = W[(W.basin == b) & (W.window == w)]
        if d.empty:
            continue
        ax.plot([d.dV_pct_V1999.min(), d.dV_pct_V1999.max()], [i, i], color="#cfcdc6", lw=3.0, solid_capstyle="round",
                zorder=2)
        for _, r in d.iterrows():
            ax.plot(r.dV_pct_V1999, i, marker="D" if r.forcing in ("CRU_TS4", "CHELSA") else "o", ms=3.2,
                    mfc=COL[r.forcing], mec="white", mew=0.4, ls="none", zorder=3)
    zero_line(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-52, 12); ax.set_xticks([-40, -20, 0])
    ax.set_title(title)
    if k == 1:
        ax.set_xlabel("Glacier volume change (% of basin $V_{1999}$)")
    if k:
        ax.tick_params(labelleft=False)
axs[0].set_yticks(y, [LBL.get(n, n) for n in order])
axs[0].tick_params(axis="y", length=0)
axs[0].set_ylim(len(order) - 0.5, -0.7)
ax = fig.add_subplot(gs[0, 3], sharey=axs[0]); ax.tick_params(labelleft=False, axis="y", length=0)
for k, (a_, b_, c) in enumerate([(1975, 2000, "#86b6ef"), (2000, 2020, "#1c5cab")]):
    off = (k - 0.5) * 0.36
    ax.barh(y + off, main.loc[order, f"dpop{a_}_{b_}_pct"], height=0.34, color=c, lw=0, label=f"{a_}–{b_}, whole basin")
    ax.scatter(main.loc[order, f"dpop{a_}_{b_}_glac_l7_pct"], y + off, marker="|", s=30, color=INK, lw=1.0, zorder=3,
               label="glacierised units (level 7)" if k == 0 else None)
for i, n in enumerate(order):
    ax.text(1.01, i, f"{main.loc[n, 'pop2020'] / 1e6:,.0f}", transform=ax.get_yaxis_transform(), ha="left",
            va="center", color=INK2)
ax.text(1.01, -0.95, "2020 (M)", transform=ax.get_yaxis_transform(), ha="left", va="center", color=INK2)
zero_line(ax, "x"); ax.grid(axis="y", visible=False)
ax.set_xlabel("Population change (%)"); ax.set_title("(d) Population (GHS-POP)")
XMAX = 130   # Irrawaddy whole-basin 1975-2000 (+225 %) is clipped and labelled
ax.set_xlim(-25, XMAX)
for i, n in enumerate(order):
    for k, (a_, b_) in enumerate([(1975, 2000), (2000, 2020)]):
        v = main.loc[n, f"dpop{a_}_{b_}_pct"]
        if v > XMAX:
            ax.text(XMAX - 2, i + (k - 0.5) * 0.36, f"{v:.0f} →", ha="right", va="center", color="white", fontsize=7)
            print(f"   clipped in (d): {n} {a_}-{b_} whole basin {v:.1f} %")
pop_h, _ = ax.get_legend_handles_labels()
fig.legend(handles=[Line2D([], [], marker="D" if f in ("CRU_TS4", "CHELSA") else "o", ls="none", mfc=COL[f],
                           mec="white", mew=0.4, ms=4, label=LABEL[f]) for f in FORC] + pop_h,
           loc="outside lower center", ncol=4)
save(fig, "fig12_basins")
print(f"   sum of basin glacier area = {main.glacier_area_km2.sum():,.1f} km2 (HMA RGI total 97,606 km2); "
      f"sum of 2020 basin population = {main.pop2020.sum() / 1e6:,.0f} M")
