#!/usr/bin/env python
"""Verify the compiled run_06_areacons set before it is used.

1. Every glacier x forcing: prior_sigma_t == 4, area_consistent == True; counts of area_factor_flag
   (ok / capped / other / missing) per forcing x region. Glaciers absent from metadata (failed) are listed.
2. Mass consistency per forcing x region (and HMA) over the calibration period (2000-2019; CHELSA 2000-2009):
   modelled mass  = sum_i sum_t dm_gt(i,t) / n_years                       (Gt/yr)
   target mass    = sum_i B_hug,i * A_RGI,i / 1000                          (Gt/yr)
   residual %     = 100 * (model / target - 1)
   Reported for (a) calib_status ok and not backward_vanished (in-sample), (b) calib ok, (c) all glaciers.
Reads run_06 directly (not paths.py). Writes tables/run06_verify_flags.csv and tables/run06_mass_check.csv.
"""
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from config import OGGM_ROOT  # noqa: E402
from pathlib import Path

import pandas as pd

RUN06 = OGGM_ROOT / "compiled_v3_run_06_areacons" / "per_forcing"
TAB = Path(__file__).resolve().parents[1] / "tables"
FORC = ["ERA5", "ERA5-Land", "HARv2", "CRU_TS4", "CHELSA"]
CAL = {f: (2000, 2019) for f in FORC}
CAL["CHELSA"] = (2000, 2009)
TRUE = {"true", "1", "1.0"}


def main():
    flags, mass = [], []
    ids_all = None
    for f in FORC:
        m = pd.read_csv(RUN06 / f / "metadata.csv")
        ids_all = set(m.rgi_id) if ids_all is None else ids_all | set(m.rgi_id)
        assert m.rgi_id.is_unique, f
        bad_sig = (m.prior_sigma_t != 4.0).sum()
        bad_ac = (~m.area_consistent.astype(str).str.lower().isin(TRUE)).sum()
        m["flag"] = m.area_factor_flag.fillna("missing").astype(str)
        for r, g in list(m.groupby("region")) + [("ALL_HMA", m)]:
            c = g.flag.value_counts()
            flags.append(dict(forcing=f, region=r, n=len(g), sigma_t_not4=int((g.prior_sigma_t != 4.0).sum()),
                              area_consistent_false=int((~g.area_consistent.astype(str).str.lower().isin(TRUE)).sum()),
                              ok=int(c.get("ok", 0)), capped=int(sum(v for k, v in c.items() if k.startswith("cap"))),
                              missing=int(c.get("missing", 0)),
                              other=int(sum(v for k, v in c.items() if k not in ("ok", "missing")
                                            and not k.startswith("cap"))),
                              other_values=";".join(sorted(k for k in c.index if k not in ("ok", "missing")
                                                           and not k.startswith("cap")))))
        print(f, "sigma_T!=4:", bad_sig, "area_consistent False:", bad_ac, flush=True)

        a0, a1 = CAL[f]
        n = a1 - a0 + 1
        y = pd.read_parquet(RUN06 / f / "yearly.parquet", columns=["rgi_id", "year", "dm_gt"],
                            filters=[("year", ">=", a0), ("year", "<=", a1)])
        dm = y.groupby("rgi_id").dm_gt.sum() / n
        m = m.set_index("rgi_id")
        m["dm_rate"] = dm.reindex(m.index).fillna(0.0)
        m["target_hug_gt"] = m.geodetic_target_hugonnet * m.A_ref_km2 / 1000
        ok = m.calib_status == "ok"
        bv = m.backward_vanished.astype(str).str.lower().isin(TRUE)
        for sample, sel in [("in_sample", ok & ~bv), ("calib_ok", ok), ("all", m.target_hug_gt.notna())]:
            s = m[sel & m.target_hug_gt.notna()]
            for r, g in list(s.groupby("region")) + [("ALL_HMA", s)]:
                mass.append(dict(forcing=f, region=r, sample=sample, period=f"{a0}-{a1}", n=len(g),
                                 model_Gt_yr=g.dm_rate.sum(), target_Gt_yr=g.target_hug_gt.sum(),
                                 residual_pct=100 * (g.dm_rate.sum() / g.target_hug_gt.sum() - 1)))
    F = pd.DataFrame(flags)
    M = pd.DataFrame(mass)
    F.to_csv(TAB / "run06_verify_flags.csv", index=False)
    M.to_csv(TAB / "run06_mass_check.csv", index=False)
    pd.set_option("display.width", 250)
    print(F.to_string(index=False))
    print(M.pivot_table(index=["sample", "region"], columns="forcing", values="residual_pct").round(2).to_string())
    print("glaciers in metadata (union):", len(ids_all))


if __name__ == "__main__":
    main()
