#!/usr/bin/env python
"""Figure 2 (Sect. 2.2) - annual temperature and precipitation of the five climate products over the glaciers of HMA,
1950-2025, each series cut at its last complete calendar year. Also writes the Supplement version.

Main:        (a) temperature lapse-corrected to the median glacier elevation, (b) precipitation.
Supplement:  (a) temperature at the grid-cell elevation (the height the product represents), (b) precipitation (same).
Both: area-weighted mean over all 95,535 modelled glaciers (RGI 6.0 area) of the series at each glacier's own grid
cell, i.e. the cells and unit conversions used in the calibration; precipitation is the product's value, not multiplied
by the calibrated precipitation factor.

ERA5-Land precipitation uses the corrected months Sep 2022 - Feb 2024 (scripts/era5land_tp_fix.py).
Input: tables/climate_series_<forcing>.csv (made by scripts/climate_series.py from the run_06 metadata and the climate files).
"""
import pandas as pd

from style import (COL, DOUBLE, FORC, INK2, LABEL, LS, TAB, figure, header, legend_handles, save)

header(15, "Climate of the five products over the HMA glaciers")
D = pd.concat([pd.read_csv(TAB / f"climate_series_{f}.csv") for f in FORC], ignore_index=True)
D = D[D.year >= 1950].copy()
for f in FORC:
    d = D[D.forcing == f]
    print(f"   {LABEL[f]:12s} {int(d.year.min())}-{int(d.year.max())}  T_grid mean {d.T_grid.mean():+.2f}  T_zmed mean "
          f"{d.T_zmed.mean():+.2f}  P mean {d.P.mean():.0f} mm  (lapse offset {d.lapse_offset_K.iloc[0]:+.2f} K, "
          f"{int(d.n_cells.iloc[0])} cells)")
    c = d[d.year.between(1980, 2016)]
    print(f"      1980-2016: T_zmed {c.T_zmed.mean():+.2f}  P {c.P.mean():.0f}")


def draw(tcol, tlabel, ttitle, name):
    fig = figure(DOUBLE, 10.5)
    axs = fig.subplots(2, 1, sharex=True)
    for ax, col in zip(axs, (tcol, "P")):
        for f in FORC:
            d = D[D.forcing == f]
            ax.plot(d.year, d[col], color=COL[f], ls=LS[f])
    axs[0].set_ylabel(tlabel)
    axs[0].set_title(ttitle)
    axs[1].set_ylabel("Precipitation (mm yr$^{-1}$)")
    axs[1].set_title("(b) Annual total precipitation, as given by the product")
    axs[1].set_xlim(1948.5, 2026)
    fig.legend(handles=legend_handles(), loc="outside lower center", ncol=5)
    save(fig, name)


draw("T_zmed", "Temperature ($^\\circ$C)",
     "(a) Annual mean 2 m temperature, lapse-corrected to the median glacier elevation", "fig15_climate")
draw("T_grid", "Temperature ($^\\circ$C)",
     "(a) Annual mean 2 m temperature at the grid-cell elevation", "figS_climate_gridcell")
