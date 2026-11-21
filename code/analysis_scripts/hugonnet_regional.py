"""Hugonnet et al. (2021), Extended Data Table 1, 2000-2019 regional mass change (RGI 6.0 regions 13-15).
Transcribed from data_external/Hugonnet2021_ExtendedDataTable1.jpg (source in data_external/SOURCES.md).
Uncertainties in the table are 95 % confidence intervals; 1 sigma = CI / 1.96."""
import numpy as np
import pandas as pd

TABLE = pd.DataFrame({            # region: area km2, mass change Gt/yr, 95 % CI Gt/yr
    "rgi_region": ["RGI60-13", "RGI60-14", "RGI60-15"],
    "name": ["Central Asia", "South Asia West", "South Asia East"],
    "area_km2": [49303.0, 33568.0, 14734.0],
    "dmdt_gt_yr": [-9.6, -4.6, -6.9],
    "dmdt_ci95_gt_yr": [2.1, 1.7, 1.4],
}).set_index("rgi_region")
Z = 1.96


def regional():
    """per region + HMA: specific MB (m w.e./yr) and 1-sigma. HMA sigma given both ways:
    'indep' = regions combined in quadrature, 'corr' = linear sum (fully correlated)."""
    t = TABLE.copy()
    t["B_mwe"] = t.dmdt_gt_yr / t.area_km2 * 1000
    t["sigma1_mwe"] = t.dmdt_ci95_gt_yr / Z / t.area_km2 * 1000
    a = t.area_km2.sum()
    h = pd.Series({"name": "HMA (13+14+15)", "area_km2": a, "dmdt_gt_yr": t.dmdt_gt_yr.sum(),
                   "dmdt_ci95_gt_yr": np.sqrt((t.dmdt_ci95_gt_yr ** 2).sum()),
                   "B_mwe": t.dmdt_gt_yr.sum() / a * 1000,
                   "sigma1_mwe": np.sqrt((t.dmdt_ci95_gt_yr ** 2).sum()) / Z / a * 1000,
                   "sigma1_mwe_corr": t.dmdt_ci95_gt_yr.sum() / Z / a * 1000}, name="ALL_HMA")
    return pd.concat([t, h.to_frame().T])


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(regional().to_string())
