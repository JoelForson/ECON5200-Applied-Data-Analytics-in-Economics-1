import numpy as np
import pandas as pd
from types import SimpleNamespace

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

import ruptures as rpt


def run_stl(series: pd.Series, period: int = 12, log_transform: bool = True, robust: bool = True):
    """Apply STL decomposition with optional log-transform and return the fitted result.
    Returns an object with attributes: trend, seasonal, resid (each a pandas Series).
    """
    s = series.copy()
    if log_transform:
        if (s <= 0).any():
            raise ValueError("Series contains non-positive values. Cannot log-transform.")
        s = np.log(s)

    stl = STL(s, period=period, robust=robust)
    res = stl.fit()

    # Ensure attributes are pandas Series with original index
    trend = pd.Series(res.trend, index=series.index)
    seasonal = pd.Series(res.seasonal, index=series.index)
    resid = pd.Series(res.resid, index=series.index)

    return SimpleNamespace(trend=trend, seasonal=seasonal, resid=resid)


def run_mstl(series: pd.Series, periods: list[int], log_transform: bool = True):
    """A simple MSTL approximation: apply STL repeatedly to extract multiple seasonal components.
    Returns an object with attributes: trend, seasonal (DataFrame of components), resid.
    Note: This is a lightweight approximation for teaching/demonstration purposes.
    """
    s = series.copy()
    if log_transform:
        if (s <= 0).any():
            raise ValueError("Series contains non-positive values. Cannot log-transform.")
        s = np.log(s)

    seasonal_components = pd.DataFrame(index=series.index)
    remainder = s.copy()

    for i, p in enumerate(periods):
        stl = STL(remainder, period=p, robust=True)
        res = stl.fit()
        seasonal_components[f'seasonal_{p}'] = res.seasonal
        remainder = remainder - res.seasonal

    # Final trend from last STL fit
    trend = pd.Series(res.trend, index=series.index)
    seasonal_sum = seasonal_components.sum(axis=1)
    resid = pd.Series((s - trend - seasonal_sum), index=series.index)

    return SimpleNamespace(trend=trend, seasonal=seasonal_components, resid=resid)


def test_stationarity(series: pd.Series, alpha: float = 0.05) -> dict:
    """Run ADF + KPSS and return test stats and a simple verdict."""
    # Use 'ct' regression which is common for macro series
    try:
        adf_res = adfuller(series.dropna(), regression='ct', autolag='AIC')
        adf_stat, adf_p = adf_res[0], adf_res[1]
    except Exception:
        adf_stat, adf_p = np.nan, np.nan

    try:
        kpss_stat, kpss_p, _, _ = kpss(series.dropna(), regression='ct', nlags='auto')
    except Exception:
        kpss_stat, kpss_p = np.nan, np.nan

    adf_reject = False if np.isnan(adf_p) else (adf_p < alpha)
    kpss_reject = False if np.isnan(kpss_p) else (kpss_p < alpha)

    if adf_reject and not kpss_reject:
        verdict = 'stationary'
    elif not adf_reject and kpss_reject:
        verdict = 'non-stationary'
    elif adf_reject and kpss_reject:
        verdict = 'contradictory'
    else:
        verdict = 'inconclusive'

    return {
        'adf_stat': adf_stat,
        'adf_p': adf_p,
        'kpss_stat': kpss_stat,
        'kpss_p': kpss_p,
        'verdict': verdict
    }


def detect_breaks(series: pd.Series, pen: float = 10) -> list:
    """Detect structural breaks using PELT (ruptures). Returns list of pd.Timestamp break dates."""
    signal = series.values
    algo = rpt.Pelt(model='rbf').fit(signal)
    bkps = algo.predict(pen=pen)

    break_dates = []
    for b in bkps:
        if b < len(series):
            break_dates.append(series.index[b])
    return break_dates


def block_bootstrap_trend(series: pd.Series, n_bootstrap: int = 200, block_size: int = 8, period: int = 12, log_transform: bool = True, confidence_level: float = 0.9):
    """Simple block bootstrap for STL trend: resample residual blocks and recompute trend to get CI bands.
    Returns (lower_series, upper_series, trends_df) where lower/upper are Series, trends_df is DataFrame of bootstrap trends.
    """
    # Fit baseline STL
    base = run_stl(series, period=period, log_transform=log_transform, robust=True)
    trend0 = base.trend
    resid0 = base.resid.dropna()

    rng = np.random.default_rng()
    n = len(series)
    trends = pd.DataFrame(index=series.index)

    # Create concatenated residuals for circular block sampling
    resid_vals = resid0.values
    m = len(resid_vals)
    if m == 0:
        raise ValueError("Not enough residual data for bootstrap.")

    for i in range(n_bootstrap):
        # sample start positions
        starts = rng.integers(0, m, size=int(np.ceil(n / block_size)))
        sample = []
        for s in starts:
            block = resid_vals[s:s+block_size]
            if len(block) < block_size:
                # wrap-around
                extra = block_size - len(block)
                block = np.concatenate([block, resid_vals[0:extra]])
            sample.append(block)
        sample = np.concatenate(sample)[:n]

        # create synthetic series = trend + sampled residuals
        if log_transform:
            synth = np.exp(trend0) * np.exp(sample)  # approximate multiplicative
        else:
            synth = trend0 + sample

        # recompute trend on synthetic series
        try:
            res = run_stl(pd.Series(synth, index=series.index), period=period, log_transform=False, robust=True)
            trends[f'b_{i}'] = res.trend
        except Exception:
            # fallback: use base trend
            trends[f'b_{i}'] = trend0

    lower = trends.quantile((1 - confidence_level) / 2, axis=1)
    upper = trends.quantile(1 - (1 - confidence_level) / 2, axis=1)

    # Return as Series aligned with original index
    return pd.Series(lower.values, index=series.index), pd.Series(upper.values, index=series.index), trends
