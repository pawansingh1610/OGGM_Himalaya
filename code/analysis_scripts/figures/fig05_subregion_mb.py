#!/usr/bin/env python
"""Figure file 05 (not in the main text at present) - annual mass balance of the 15 sub-regions, five forcings.

Input: analysis/B_regional_series.csv (level == "subregion"; unit, forcing, year, B_mwe)
"""
import pandas as pd

from style import (AN, CHELSA_FLAG, COL, DOUBLE, FORC, LS, SUBREG_ORDER, figure, header, legend_handles, save,
                   shade_periods, short, zero_line)

header(5, "sub-region mass balance")
s = pd.read_csv(AN / "B_regional_series.csv")
s = s[s.level == "subregion"]
fig = figure(DOUBLE, 17.0)
axs = fig.subplots(5, 3, sharex=True, sharey=True)
for ax, sr in zip(axs.flat, SUBREG_ORDER):
    shade_periods(ax)
    for f in FORC:
        d = s[(s.unit == sr) & (s.forcing == f) & s.B_mwe.notna()]
        ax.plot(d.year, d.B_mwe, color=COL[f], ls=LS[f], lw=0.75)
    zero_line(ax)
    ax.set_title(short(sr) + (" *" if sr in CHELSA_FLAG else ""), fontsize=7)
    m = s[(s.unit == sr) & s.year.between(2000, 2019)].groupby("forcing").B_mwe.mean()
    print(f"   {short(sr):17s} B 2000-2019 (CHELSA 2000-2016) range over forcings {m.min():+.2f} .. {m.max():+.2f}")
for ax in axs[:, 0]:
    ax.set_ylabel("m w.e. yr$^{-1}$")
axs[0, 0].set_xlim(1948.5, 2026)
fig.legend(handles=legend_handles(), loc="outside lower center", ncol=5)
save(fig, "fig05_subregion_mb")
print("   * CHELSA-W5E5 flagged (unreachable calibration area >= 9 %)")
