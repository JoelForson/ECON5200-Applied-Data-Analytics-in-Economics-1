def run_stl(
    series: pd.Series,
    period: int = 12,
    log_transform: bool = True,
    robust: bool = True
):
    """Apply STL decomposition with optional log-transform."""
    if log_transform:
        if (series <= 0).any():
            raise ValueError("Series contains non-positive values. Cannot log-transform.")
        series_to_decompose = np.log(series)
    else:
        series_to_decompose = series.copy()
    
    stl = STL(series_to_decompose, period=period, robust=robust)
    result = stl.fit()
    return result

def test_stationarity(series: pd.Series, alpha: float = 0.05) -> dict:
    """Run ADF + KPSS and return the 2x2 decision table verdict."""
    # Determine appropriate regression based on series characteristics
    # For most economic series, use 'ct' (constant + trend)
    regression = 'ct'
    
    # ADF test (H0: unit root)
    adf_stat, adf_p, _, _, _, _ = adfuller(series, regression=regression, autolag='AIC')
    
    # KPSS test (H0: stationary)
    kpss_stat, kpss_p, _, _ = kpss(series, regression=regression, nlags='auto')
    
    # 2x2 decision logic
    adf_reject = adf_p < alpha
    kpss_reject = kpss_p < alpha
    
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
    """Detect structural breaks using the PELT algorithm."""
    # Convert to numpy array for ruptures
    signal = series.values
    
    # Apply PELT algorithm
    algo = rpt.Pelt(model='rbf').fit(signal)
    breakpoint_indices = algo.predict(pen=pen)
    
    # Convert indices to dates, excluding the final index (end of series)
    break_dates = []
    for bp_idx in breakpoint_indices:
        if bp_idx < len(series):
            break_dates.append(series.index[bp_idx])
    
    return break_dates