"""Percent convention (decision B, 2026-11-09): every volume change is also given as % of the
consensus volume at the end of 1999 (V_1999) of the same glacier sample, one reference for all
windows. V_1999 = V at year 1999 in analysis/B_regional_series.csv (forcing x level x unit); it is
the Farinotti et al. (2019) consensus volume (Bahr fallback for 7 glaciers) of the in-sample glaciers.
km3 ice and Gt stay the primary numbers."""
from pathlib import Path

import pandas as pd

AN = Path(__file__).resolve().parents[1] / "analysis"


def v1999(series=None):
    s = series if series is not None else pd.read_csv(AN / "B_regional_series.csv")
    return (s[s.year == 1999].set_index(["forcing", "level", "unit"]).V.rename("V_1999_km3"))


def windows_with_v1999(windows=None, series=None):
    """C_volume_change_windows + V_1999_km3 + dV_pct_V1999 (= dV / V_1999 * 100)."""
    w = windows if windows is not None else pd.read_csv(AN / "C_volume_change_windows.csv")
    w = w.join(v1999(series), on=["forcing", "level", "unit"])
    w["dV_pct_V1999"] = w.dV_km3 / w.V_1999_km3 * 100
    return w


if __name__ == "__main__":
    w = windows_with_v1999()
    h = w[w.level == "hma"].pivot(index="window", columns="forcing", values=["dV_km3", "dM_Gt", "dV_pct_V1999", "dV_pct"])
    pd.set_option("display.width", 250)
    print(v1999().xs("hma", level="level").round(1).to_string())
    print(h.round(2).to_string())
