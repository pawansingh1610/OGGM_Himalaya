#!/usr/bin/env python
"""Area consistency of calibration and mass integration

Hugonnet et al. (2021) per-glacier specific rates (our calibration targets) are mass change / fixed RGI 6.0 area
(checked: sum(dmdtda * A_RGI) reproduces their Extended Data Table 1 regional Gt/yr for regions 13-15).
The model matches the specific rate B over the calibration period but integrates mass with the shrinking
V-A area, dM = B * A(t). Mass loss in the calibration period is therefore underestimated by mean(A_t)/A_RGI.

This script quantifies, per forcing and per RGI region / HMA (in-sample glaciers):
  ratio      = sum_t sum_i A_start(i,t) / (n_years * sum_i A_RGI(i)) over the calibration period
  mass       = modelled mass change rate (Gt/yr) vs the target mass rate sum_i target_i * A_RGI,i
  correction = first-order estimate of a recalibration to an area-consistent (mass) target: each glacier's B is
               shifted by the constant delta_i = (target_i * A_RGI,i - mean_t[B(i,t) * A(i,t)]) / mean_t A(i,t), which
               makes its calibration-period mass equal the target mass (a change of dT shifts B almost uniformly;
               delta_i also absorbs the covariance of B and A in time). The window volume change shifts by
               sum_t sum_i delta_i * A(i,t) / 0.9. Area feedback of the shift is ignored (first order).
Outputs: tables/area_consistency_ratio.csv, tables/area_consistency_windows.csv (and printed summaries).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paths import PER_FORCING  # noqa: E402
from volume_ref import windows_with_v1999  # noqa: E402

FORC = ["ERA5", "ERA5-Land", "HARv2", "CRU_TS4", "CHELSA"]
CAL = {f: (2000, 2019) for f in FORC}
CAL["CHELSA"] = (2000, 2009)
WIN = {"recon_1950-1979": (1950, 1979), "recon_1950-1999": (1950, 1999), "recon_1980-1999": (1980, 1999),
       "full_1980-2016": (1980, 2016), "fwd_2000-2016": (2000, 2016), "fwd_2000-2019": (2000, 2019),
       "fwd_2000-2025": (2000, 2025), "full_1950-2019": (1950, 2019), "full_1950-2025": (1950, 2025)}
TAB = ROOT / "tables"


def main():
    W = windows_with_v1999()
    rat, win = [], []
    for f in FORC:
        m = pd.read_csv(PER_FORCING / f / "metadata.csv", usecols=["rgi_id", "A_ref_km2", "geodetic_target",
                                                                   "calib_status", "backward_vanished", "region"])
        m = m[(m.calib_status == "ok") & ~m.backward_vanished.astype(str).str.lower().isin(["true", "1"])]
        y = pd.read_parquet(PER_FORCING / f / "yearly.parquet", columns=["rgi_id", "year", "area_start_km2", "dm_gt"])
        y = y[y.rgi_id.isin(m.rgi_id) & y.area_start_km2.notna()]
        a0, a1 = CAL[f]
        n = a1 - a0 + 1
        c = y[y.year.between(a0, a1)]
        abar = c.groupby("rgi_id").area_start_km2.sum() / n          # mean A over the calibration period
        dmc = c.groupby("rgi_id").dm_gt.sum() / n
        m = m.set_index("rgi_id")
        m["abar"], m["dm_rate"] = abar, dmc
        m["target_gt"] = m.geodetic_target * m.A_ref_km2 / 1000        # m w.e. * km2 / 1000 = Gt
        # shift of B that makes the modelled calibration-period mass equal the target mass, glacier by glacier
        m["delta"] = np.where(m.abar > 0, (m.target_gt - m.dm_rate) * 1000 / m.abar, 0.0)
        for u, g in list(m.groupby("region")) + [("ALL_HMA", m)]:
            rat.append(dict(forcing=f, unit=u, calib_period=f"{a0}-{a1}", A_RGI_km2=g.A_ref_km2.sum(),
                            mean_A_calib_km2=g.abar.sum(), ratio=g.abar.sum() / g.A_ref_km2.sum(),
                            target_mass_Gt_yr=g.target_gt.sum(), model_mass_Gt_yr=g.dm_rate.sum(),
                            deficit_Gt_yr=g.dm_rate.sum() - g.target_gt.sum(),
                            deficit_pct=100 * (g.dm_rate.sum() / g.target_gt.sum() - 1),
                            area_weighted_delta_mwe=np.average(g.delta, weights=g.A_ref_km2)))
        # first-order window shift (HMA)
        y = y.merge(m[["delta"]], left_on="rgi_id", right_index=True)
        y["shift_km3"] = y.delta * y.area_start_km2 / 1000 / 0.9
        for w, (w0, w1) in WIN.items():
            ref = W[(W.forcing == f) & (W.level == "hma") & (W.window == w)]
            if ref.empty: continue
            r = ref.iloc[0]
            s = y[y.year.between(w0, w1)].shift_km3.sum()
            # backward windows: a more negative B before 2000 means MORE ice in the past, i.e. the change
            # V(end) - V(start) becomes more negative as well -> same sign convention as forward.
            win.append(dict(forcing=f, window=w, dV_km3=r.dV_km3, shift_km3=s, dV_corr_km3=r.dV_km3 + s,
                            V_1999_km3=r.V_1999_km3, dV_pct_V1999=r.dV_pct_V1999,
                            dV_corr_pct_V1999=100 * (r.dV_km3 + s) / r.V_1999_km3,
                            shift_pp=100 * s / r.V_1999_km3))
        print(f"{f} done", flush=True)
    R = pd.DataFrame(rat); Wd = pd.DataFrame(win)
    R.to_csv(TAB / "area_consistency_ratio.csv", index=False)
    Wd.to_csv(TAB / "area_consistency_windows.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(4).to_string(index=False))
    print(Wd.pivot(index="window", columns="forcing", values="shift_pp").round(2).to_string())
    print(Wd.pivot(index="window", columns="forcing", values="dV_corr_pct_V1999").round(2).to_string())


if __name__ == "__main__":
    main()
