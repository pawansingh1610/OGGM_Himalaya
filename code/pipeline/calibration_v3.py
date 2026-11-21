"""
================================================================================
CALIBRATION + GEOMETRY  (v3)
================================================================================
Replaces the 2-parameter differential-evolution calibration and the
volume–area integration of v2.

[1] Calibration — well-posed instead of arbitrary
    One geodetic number cannot constrain (temp_bias, prcp_fac): every pair on
    a curve fits equally well, and differential evolution returned a random
    point on that curve (seed-dependent). The pre-2000 reconstruction depends
    on which point is picked, so v2 injected noise into exactly the period
    the study is about.

    v3: for any prcp_fac the MB is monotonic in temp_bias, so we solve
    temp_bias exactly (Brent root) and choose, among all exact solutions,
    the one closest to a stated prior:
        minimise  (dT/sigma_T)^2 + (ln(pf/pf0)/sigma_lnP)^2
        subject to  mean MB(dT, pf) == geodetic target
    Deterministic, reproducible, ~10x fewer MB evaluations than DE×5 seeds.
    Every glacier gets calib_status:
        'ok'          exact solution found inside bounds
        'unreachable' no (dT, pf) in bounds reaches the target -> best
                      bounded compromise returned, residual reported
    The prior is a modelling choice: test sensitivity (PRIOR_PF0 = 1.0 and
    2.5) on a subsample and report it.

[2] Geometry — mass-conserving, can actually reach zero
    * dV_ice = B [m w.e.] * A [km2] * 1e-3 / (rho_ice/rho_w)   (v2 forgot /0.9)
    * All rows = END-of-year state (calendar year in OGGM 1.6). V(end of 1999) = V_ref (RGI
      epoch). Forward:  V(t)   = V(t-1) + dV(B_t, A(t-1))
                Backward: V(t-1) = V(t)   - dV(B_t, A(t))
    * No 1 % floor. Forward: V <= 0 -> glacier gone (extinct_year), stays 0,
      MB = NaN afterwards. Backward: if V would drop <= 0 going back in time
      (i.e. glacier would have "grown from nothing" since then) the run is
      flagged backward_vanished=True and those years are marked invalid.
    * dm_gt (= km3 w.e.) and area_start_km2 are written for every year so
      regional MB can be computed exactly as sum(dm)/sum(A_start).
================================================================================
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

RHO_RATIO = 0.900          # rho_ice / rho_water (OGGM uses 900 kg m-3)
GAMMA_VA  = 1.375          # Bahr et al. (1997)
C_VA_GLOBAL = 0.034

# ── Calibration bounds: SINGLE source of truth, written to metadata ──────────
BOUNDS = {
    # +-10 degC: with the correct ref_hgt, CRU TS and CHELSA-W5E5 (W5E5 is
    # bias-adjusted to CRU) are ~10 degC warmer than ERA5/HARv2 over HMA
    # glaciers — a real product bias the calibration must be allowed to see.
    # The prior (PRIOR_SIGMA_T) still pulls solutions towards 0.
    'temp_bias': (-10.0, 10.0),
    'prcp_fac':  (0.3, 6.0),
}
# ── Prior (centre of the admissible solution curve) ──────────────────────────
# Override from the shell for sensitivity runs, e.g.
#   OGGM_V3_PRIOR_SIGMA_T=4 OGGM_V3_RUN=run_03_sT4 python main_calibration_parallel_v2.py
import os as _os
PRIOR_T0      = float(_os.environ.get('OGGM_V3_PRIOR_T0', 0.0))       # degC
# 4 degC chosen from the test runs (2026-11-05): with 1.5 degC the warm-biased
# forcings (CRU, CHELSA-W5E5) compensated with pf ~5-6 instead of dT; with 4 degC
# pf drops to ~1.8, field RMSE improves for 4/5 forcings (CHELSA 0.97 -> 0.74).
PRIOR_SIGMA_T = float(_os.environ.get('OGGM_V3_PRIOR_SIGMA_T', 4.0))  # degC
PRIOR_PF0     = float(_os.environ.get('OGGM_V3_PRIOR_PF0', 1.5))
PRIOR_SIGMA_LNP = np.log(float(_os.environ.get('OGGM_V3_PRIOR_PF_FACTOR', 2.0)))
RESID_TOL = 0.01                 # m w.e. yr-1 — "exact" fit tolerance


# ─────────────────────────────────────────────────────────────────────────────
# MB model construction (same as calibration_functions.create_mass_balance_model
# but the global params written to mb_calib.json are the ones actually used)
# ─────────────────────────────────────────────────────────────────────────────

def make_mb_model(gdir, temp_bias: float, prcp_fac: float, melt_f: float,
                  climate_source: str = 'custom'):
    from oggm import cfg
    from oggm.core.massbalance import MultipleFlowlineMassBalance
    gdir.write_json({
        'rgi_id': gdir.rgi_id,
        'bias': 0,
        'temp_bias': float(temp_bias),
        'prcp_fac': float(prcp_fac),
        'melt_f': float(melt_f),
        'baseline_climate_source': climate_source,
        'mb_global_params': {k: cfg.PARAMS[k] for k in
                             ['temp_default_gradient', 'temp_all_solid',
                              'temp_all_liq', 'temp_melt']},
    }, 'mb_calib')
    return MultipleFlowlineMassBalance(
        gdir, use_inversion_flowlines=True,
        fls=gdir.read_pickle('inversion_flowlines'),
        check_calib_params=False)


# ─────────────────────────────────────────────────────────────────────────────
# Fast parameter setting on an OGGM MultipleFlowlineMassBalance
# ─────────────────────────────────────────────────────────────────────────────

def set_mb_params(mb_model, temp_bias: float, prcp_fac: float) -> None:
    """
    OGGM >= 1.6: MultipleFlowlineMassBalance exposes temp_bias / prcp_fac
    setters that propagate to every flowline model, so we do not need to
    rewrite mb_calib.json and rebuild the model at every evaluation.
    """
    mb_model.temp_bias = float(temp_bias)
    mb_model.prcp_fac = float(prcp_fac)


def period_mean_mb(mb_model, years) -> float:
    """Mean specific MB (m w.e. yr-1) over an explicit list of years."""
    vals = []
    for y in years:
        try:
            vals.append(mb_model.get_specific_mb(year=int(y)) / 1000.0)
        except Exception:
            return np.nan
    return float(np.mean(vals)) if vals else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_regularized(mb_fun, target: float, sigma_target: float = None,
                          bounds: dict = None,
                          t0: float = PRIOR_T0, sigma_t: float = PRIOR_SIGMA_T,
                          pf0: float = PRIOR_PF0,
                          sigma_lnp: float = PRIOR_SIGMA_LNP) -> dict:
    """
    Parameters
    ----------
    mb_fun : callable (temp_bias, prcp_fac) -> mean MB over calibration years
             [m w.e. yr-1]. Must be decreasing in temp_bias and increasing
             in prcp_fac (true for any temperature-index model).
    target : geodetic MB [m w.e. yr-1]

    Returns dict with temp_bias, prcp_fac, modeled_mb, residual,
    calib_status, at_tb_bound, at_pf_bound, n_mb_evals.
    """
    b = bounds or BOUNDS
    tlo, thi = b['temp_bias']
    plo, phi = b['prcp_fac']
    n_eval = [0]

    def f(tb, pf):
        n_eval[0] += 1
        return mb_fun(tb, pf)

    def solve_tb(pf):
        """Exact temp_bias for this pf, or None if not bracketed."""
        g_lo = f(tlo, pf) - target      # coldest -> most positive MB
        g_hi = f(thi, pf) - target      # warmest -> most negative MB
        if not (np.isfinite(g_lo) and np.isfinite(g_hi)):
            return None, np.nan
        if g_lo < 0:                     # even coldest is too negative
            return None, g_lo
        if g_hi > 0:                     # even warmest is too positive
            return None, g_hi
        tb = brentq(lambda t: f(t, pf) - target, tlo, thi,
                    xtol=1e-3, maxiter=60)
        return tb, 0.0

    def prior_cost(tb, pf):
        return ((tb - t0) / sigma_t) ** 2 + \
               (np.log(pf / pf0) / sigma_lnp) ** 2

    # 1) Coarse scan in log(pf) to find the feasible interval
    pf_grid = np.exp(np.linspace(np.log(plo), np.log(phi), 9))
    feas = []
    for pf in pf_grid:
        tb, _ = solve_tb(pf)
        if tb is not None:
            feas.append((pf, tb, prior_cost(tb, pf)))

    if feas:
        # 2) Refine on the feasible log(pf) sub-interval around the best node
        i_best = int(np.argmin([c for _, _, c in feas]))
        pf_feas = [p for p, _, _ in feas]
        lo_i = max(0, i_best - 1)
        hi_i = min(len(pf_feas) - 1, i_best + 1)
        a, c = np.log(pf_feas[lo_i]), np.log(pf_feas[hi_i])

        def obj(lnp):
            tb, _ = solve_tb(np.exp(lnp))
            return 1e6 if tb is None else prior_cost(tb, np.exp(lnp))

        if c > a:
            res = minimize_scalar(obj, bounds=(a, c), method='bounded',
                                  options={'xatol': 0.01})
            pf_opt = float(np.exp(res.x))
            tb_opt, _ = solve_tb(pf_opt)
            if tb_opt is None:                       # numerical edge
                pf_opt, tb_opt = feas[i_best][0], feas[i_best][1]
        else:
            pf_opt, tb_opt = feas[i_best][0], feas[i_best][1]
        status = 'ok'
    else:
        # 3) Unreachable: target outside what the bounds allow.
        #    Return the corner that gets closest and flag it.
        corners = [(tlo, phi), (thi, plo)]
        errs = [abs(f(t, p) - target) for t, p in corners]
        tb_opt, pf_opt = corners[int(np.argmin(errs))]
        status = 'unreachable'

    mod = f(tb_opt, pf_opt)
    resid = mod - target
    if status == 'ok' and abs(resid) > max(RESID_TOL, 0.05 * (sigma_target or 0)):
        status = 'unreachable'

    eps = 1e-3
    return {
        'temp_bias':   float(tb_opt),
        'prcp_fac':    float(pf_opt),
        'modeled_mb':  float(mod),
        'target_mb':   float(target),
        'residual':    float(resid),
        'calib_status': status,
        'at_tb_bound': bool(tb_opt <= tlo + eps or tb_opt >= thi - eps),
        'at_pf_bound': bool(pf_opt <= plo + eps or pf_opt >= phi - eps),
        'n_mb_evals':  int(n_eval[0]),
        'bounds_temp_bias': list(b['temp_bias']),
        'bounds_prcp_fac':  list(b['prcp_fac']),
        'prior': {'t0': t0, 'sigma_t': sigma_t, 'pf0': pf0,
                  'sigma_lnp': float(sigma_lnp)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

def _integrate(years, mb, V0, A0, C, reverse: bool):
    """
    years, mb : arrays sorted ascending, for the direction being integrated
                forward: years >= ref_year+1 ... ; backward: years <= ref_year
    Returns dict year -> (V_end, A_end, A_start, dm_gt, valid)
    """
    out = {}
    V, A = V0, A0
    gone = False
    if not reverse:
        for y, b in zip(years, mb):
            A_start = A
            if gone:                                # stays gone, no mass
                out[y] = (0.0, 0.0, 0.0, 0.0, True)
                continue
            if not np.isfinite(b):                  # missing MB year
                out[y] = (V, A, A_start, np.nan, False)
                continue
            dm = b * A * 1e-3                       # km3 w.e. == Gt
            V_new = V + dm / RHO_RATIO              # km3 ice
            if V_new <= 0.0:
                dm = -V * RHO_RATIO                 # only what was left
                V, A, gone = 0.0, 0.0, True
            else:
                V = V_new
                A = (V / C) ** (1.0 / GAMMA_VA)
            out[y] = (V, A, A_start, dm, True)
    else:
        # state at END of year y is known; step to END of y-1
        vanished = False
        for y, b in zip(years[::-1], mb[::-1]):
            # record: year y-1 end state
            if vanished or not np.isfinite(b):
                out[y - 1] = (np.nan, np.nan, np.nan, np.nan, False)
                vanished = vanished or not np.isfinite(b)
                continue
            dm = b * A * 1e-3            # A(t) ~ A at end of year y
            V_prev = V - dm / RHO_RATIO
            if V_prev <= 0.0:
                vanished = True
                out[y - 1] = (np.nan, np.nan, np.nan, np.nan, False)
                continue
            V = V_prev
            A_prev = (V / C) ** (1.0 / GAMMA_VA)
            out[y - 1] = (V, A_prev, np.nan, np.nan, True)
            A = A_prev
        out['_vanished'] = vanished
    return out


def compute_geometry_timeseries_v3(mb_rows: list, V_ref: float, A_ref: float,
                                   u_thick: float, ref_year: int = 2000,
                                   glacier_flags: dict = None) -> pd.DataFrame:
    """
    mb_rows : [{'year': y, 'mb': B_y [m w.e.]}, ...] mass-balance years
    V_ref, A_ref : state at the RGI epoch, assigned to END of (ref_year-1)
    u_thick : relative thickness uncertainty -> low/high volume scenarios

    Output: one row per year from (first_mb_year-1) .. last_mb_year with
        volume_km3[_low,_high], area_km2[_low,_high], area_start_km2,
        dm_gt (best), mb_annual_mwe, is_valid, reconstructed, extinct,
        extinct_year, backward_vanished
    """
    df_mb = pd.DataFrame(mb_rows).sort_values('year')
    yrs = df_mb['year'].to_numpy(int)
    mb = df_mb['mb'].to_numpy(float)

    fwd = yrs >= ref_year
    bwd = yrs <= ref_year - 1

    scen = {
        'best': V_ref,
        'low':  V_ref * (1.0 - u_thick),
        'high': V_ref * (1.0 + u_thick),
    }
    res = {}
    for k, V0 in scen.items():
        C = V0 / A_ref ** GAMMA_VA
        rf = _integrate(yrs[fwd], mb[fwd], V0, A_ref, C, reverse=False)
        rb = _integrate(yrs[bwd], mb[bwd], V0, A_ref, C, reverse=True) \
            if bwd.any() else {'_vanished': False}
        res[k] = (rf, rb)

    rf_b, rb_b = res['best']
    backward_vanished = bool(rb_b.get('_vanished', False))
    mb_map = dict(zip(yrs, mb))

    all_years = sorted(set([y for y in rf_b]) |
                       set([y for y in rb_b if y != '_vanished']) |
                       {ref_year - 1})
    rows = []
    extinct_year = None
    for y in all_years:
        r = {'year': int(y)}
        for k in ('best', 'low', 'high'):
            rf, rb = res[k]
            sfx = '' if k == 'best' else f'_{k}'
            if y == ref_year - 1:
                V, A = scen[k], A_ref
            elif y in rf:
                V, A = rf[y][0], rf[y][1]
            elif y in rb:
                V, A = rb[y][0], rb[y][1]
            else:
                V, A = np.nan, np.nan
            r[f'volume_km3{sfx}'] = V
            r[f'area_km2{sfx}'] = A
        if y in rf_b:
            _, _, A_start, dm, valid = rf_b[y]
            r['area_start_km2'] = A_start
            r['dm_gt'] = dm
            r['is_valid'] = bool(valid)
            # no specific MB for a glacier that had no area this year
            r['mb_annual_mwe'] = (mb_map.get(y, np.nan)
                                  if (valid and A_start > 0) else np.nan)
            if extinct_year is None and rf_b[y][0] == 0.0:
                extinct_year = int(y)
        else:
            # backward / reference year: start-of-year area = end of y-1
            r['is_valid'] = bool(np.isfinite(r['volume_km3']))
            r['mb_annual_mwe'] = mb_map.get(y, np.nan) if r['is_valid'] else np.nan
            r['area_start_km2'] = np.nan
            r['dm_gt'] = np.nan
        r['reconstructed'] = bool(y < ref_year - 1)
        rows.append(r)

    df = pd.DataFrame(rows)

    # Backward rows: area at start of year y is the end-state of y-1,
    # dm_gt = B_y * A_start
    df = df.sort_values('year').reset_index(drop=True)
    prev_A = df['area_km2'].shift(1)
    m = df['year'] <= ref_year - 1
    df.loc[m, 'area_start_km2'] = prev_A[m]
    df.loc[m, 'dm_gt'] = df.loc[m, 'mb_annual_mwe'] * df.loc[m, 'area_start_km2'] * 1e-3

    df['mean_thickness_m'] = np.where(df['area_km2'] > 0,
                                      df['volume_km3'] / df['area_km2'] * 1e3, 0.0)
    df['length_km'] = 1.60 * np.clip(df['area_km2'], 0, None) ** 0.60
    df['extinct'] = df['volume_km3'] == 0.0
    df['extinct_year'] = extinct_year if extinct_year is not None else np.nan
    df['backward_vanished'] = backward_vanished
    for c in ['volume_km3', 'volume_km3_low', 'volume_km3_high']:
        df[c] = df[c].round(7)
    for c in ['area_km2', 'area_km2_low', 'area_km2_high', 'area_start_km2']:
        df[c] = df[c].round(6)
    if glacier_flags:
        for k, v in glacier_flags.items():
            df[k] = v
    return df


def consensus_volume_km3(gdir, A_rgi_km2: float):
    """
    V_2000 directly from the Farinotti et al. (2019) consensus thickness
    (what the paper describes), NOT from an OGGM inversion run with an
    uncalibrated MB model as in v2.

    oggm.shop.bedtopo.add_consensus_thickness rescales the gridded field so
    that its sum equals the consensus volume, so
        V = sum(thickness) * dx^2
    Returns (V_km3, mean_thick_m = V / A_RGI) or (None, None).
    """
    import xarray as xr
    try:
        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            if 'consensus_ice_thickness' not in ds:
                return None, None
            thk = ds['consensus_ice_thickness'].values
        V = float(np.nansum(thk)) * gdir.grid.dx ** 2 * 1e-9     # km3
        if not np.isfinite(V) or V <= 0:
            return None, None
        return V, V / A_rgi_km2 * 1e3
    except Exception:
        return None, None
