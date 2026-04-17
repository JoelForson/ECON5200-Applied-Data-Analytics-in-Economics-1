"""
Time Series Analysis Dashboard
Interactive Streamlit app for decomposition, stationarity testing, and structural break detection.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
import warnings
warnings.filterwarnings('ignore')

# Import our custom module (make sure decompose.py is in the same directory)
try:
    from decompose import run_stl, run_mstl, test_stationarity, detect_breaks, block_bootstrap_trend
except ImportError:
    st.error("Please ensure decompose.py is in the same directory as this app.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Time Series Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Time Series Analysis Dashboard")
st.markdown("""
**Interactive platform for advanced time series decomposition and diagnostics**

This app demonstrates key econometric concepts:
- **STL vs MSTL**: Single vs multiple seasonal decomposition
- **Stationarity Testing**: ADF + KPSS with 2×2 decision table
- **Structural Breaks**: PELT algorithm with penalty tuning
- **Bootstrap Uncertainty**: Block resampling for trend confidence bands
""")

# Sidebar for inputs
st.sidebar.header("📊 Data & Parameters")

# FRED API setup
st.sidebar.subheader("Data Source")
fred_api_key = st.sidebar.text_input(
    "FRED API Key",
    value="19333dfa76b436c7f74dfdbbd46a9676",
    help="Get free key at https://fred.stlouisfed.org/docs/api/api_key.html"
)

# Series selection with common examples
series_examples = {
    "Real GDP": "GDPC1",
    "Retail Sales": "RSXFSN", 
    "Unemployment Rate": "UNRATE",
    "CPI (All Items)": "CPIAUCSL",
    "10-Year Treasury": "GS10",
    "Industrial Production": "INDPRO"
}

selected_example = st.sidebar.selectbox(
    "Choose Example or Enter Custom:",
    ["Custom"] + list(series_examples.keys())
)

if selected_example != "Custom":
    fred_series = series_examples[selected_example]
else:
    fred_series = st.sidebar.text_input("FRED Series ID", value="GDPC1")

start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.Timestamp("2000-01-01")
)

# Analysis parameters
st.sidebar.subheader("Analysis Parameters")

decomp_method = st.sidebar.selectbox(
    "Decomposition Method",
    ["STL", "MSTL", "Classical"],
    help="STL=robust, MSTL=multiple seasonalities"
)

if decomp_method == "MSTL":
    periods_input = st.sidebar.text_input(
        "Periods (comma-separated)",
        value="12,4",
        help="e.g., '24,168' for hourly data with daily+weekly cycles"
    )
    try:
        periods = [int(p.strip()) for p in periods_input.split(",")]
    except:
        periods = [12]
else:
    period = st.sidebar.slider("Seasonal Period", 2, 52, 12)

log_transform = st.sidebar.checkbox(
    "Log Transform",
    value=True,
    help="Use for multiplicative seasonality (growing amplitude)"
)

robust = st.sidebar.checkbox(
    "Robust Fitting",
    value=True,
    help="Downweight outliers in STL"
)

# Structural breaks
st.sidebar.subheader("Structural Breaks")
penalty = st.sidebar.slider(
    "PELT Penalty",
    1.0, 50.0, 10.0,
    help="Higher = fewer breaks (bias-variance tradeoff)"
)

# Bootstrap parameters
st.sidebar.subheader("Bootstrap Confidence")
show_bootstrap = st.sidebar.checkbox("Show Bootstrap Bands")
if show_bootstrap:
    n_bootstrap = st.sidebar.slider("Bootstrap Replications", 50, 500, 200)
    block_size = st.sidebar.slider("Block Size", 2, 20, 8)
    confidence = st.sidebar.slider("Confidence Level", 0.8, 0.99, 0.9)

# Load and process data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_fred_data(series_id, api_key, start_date):
    """Load data from FRED with error handling."""
    try:
        fred = Fred(api_key=api_key)
        data = fred.get_series(series_id, observation_start=start_date)
        data = data.dropna()
        
        # Auto-detect frequency
        if len(data) > 1:
            freq_days = (data.index[1] - data.index[0]).days
            if freq_days <= 7:
                freq = 'D'
            elif freq_days <= 40:
                freq = 'MS'
            elif freq_days <= 100:
                freq = 'QS'
            else:
                freq = 'AS'
            
            data.index = pd.DatetimeIndex(data.index)
            data.index.freq = freq
        
        return data, None
    except Exception as e:
        return None, str(e)

# Main analysis
if fred_api_key != "YOUR_FRED_API_KEY_HERE":
    with st.spinner("Loading data from FRED..."):
        data, error = load_fred_data(fred_series, fred_api_key, start_date)
    
    if error:
        st.error(f"Error loading data: {error}")
        st.stop()
    
    if data is None or len(data) == 0:
        st.error("No data available for the specified series and date range.")
        st.stop()
    
    st.success(f"✅ Loaded {len(data)} observations for {fred_series}")
    
    # Data overview
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Start Date", data.index[0].strftime("%Y-%m-%d"))
        st.metric("Observations", len(data))
    with col2:
        st.metric("End Date", data.index[-1].strftime("%Y-%m-%d"))
        st.metric("Frequency", str(data.index.freq) if data.index.freq else "Unknown")
    
    # Decomposition analysis
    st.header("🔍 Time Series Decomposition")
    
    try:
        if decomp_method == "STL":
            decomp_result = run_stl(data, period=period, log_transform=log_transform, robust=robust)
            seasonal_data = decomp_result.seasonal
        elif decomp_method == "MSTL":
            decomp_result = run_mstl(data, periods=periods, log_transform=log_transform)
            seasonal_data = decomp_result.seasonal.sum(axis=1)  # Sum all seasonal components
        else:  # Classical
            from statsmodels.tsa.seasonal import seasonal_decompose
            if log_transform:
                decomp_result = seasonal_decompose(np.log(data), model='additive', period=period)
            else:
                decomp_result = seasonal_decompose(data, model='multiplicative', period=period)
            seasonal_data = decomp_result.seasonal
        
        # Create decomposition plot
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
            vertical_spacing=0.08
        )
        
        # Original series
        fig.add_trace(go.Scatter(
            x=data.index, y=data.values,
            name='Original', line=dict(color='blue', width=1)
        ), row=1, col=1)
        
        # Trend
        fig.add_trace(go.Scatter(
            x=decomp_result.trend.index, y=decomp_result.trend.values,
            name='Trend', line=dict(color='orange', width=2)
        ), row=2, col=1)
        
        # Seasonal
        fig.add_trace(go.Scatter(
            x=seasonal_data.index, y=seasonal_data.values,
            name='Seasonal', line=dict(color='green', width=1)
        ), row=3, col=1)
        
        # Residual
        fig.add_trace(go.Scatter(
            x=decomp_result.resid.index, y=decomp_result.resid.values,
            name='Residual', line=dict(color='red', width=1)
        ), row=4, col=1)
        
        # Add bootstrap confidence bands if requested
        if show_bootstrap and decomp_method in ["STL"]:
            with st.spinner("Computing bootstrap confidence bands..."):
                try:
                    lower, upper, _ = block_bootstrap_trend(
                        data, n_bootstrap=n_bootstrap, block_size=block_size, 
                        period=period, log_transform=log_transform, confidence_level=confidence
                    )
                    
                    # Add confidence band
                    fig.add_trace(go.Scatter(
                        x=upper.index, y=upper.values,
                        fill=None, mode='lines', line_color='rgba(0,0,0,0)',
                        showlegend=False, hoverinfo="skip"
                    ), row=2, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=lower.index, y=lower.values,
                        fill='tonexty', fillcolor='rgba(0,100,200,0.2)',
                        mode='lines', line_color='rgba(0,0,0,0)',
                        name=f'{int(confidence*100)}% Bootstrap CI', showlegend=True
                    ), row=2, col=1)
                    
                except Exception as e:
                    st.warning(f"Bootstrap failed: {e}")
        
        fig.update_layout(
            height=800,
            title=f"{decomp_method} Decomposition: {fred_series}",
            showlegend=True
        )
        
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Decomposition insights
        st.subheader("📊 Decomposition Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trend_change = (decomp_result.trend.iloc[-1] - decomp_result.trend.iloc[0]) / decomp_result.trend.iloc[0] * 100
            st.metric("Trend Change", f"{trend_change:.2f}%")
        
        with col2:
            seasonal_amplitude = seasonal_data.max() - seasonal_data.min()
            st.metric("Seasonal Amplitude", f"{seasonal_amplitude:.3f}")
        
        with col3:
            residual_std = decomp_result.resid.std()
            st.metric("Residual Std Dev", f"{residual_std:.3f}")
        
    except Exception as e:
        st.error(f"Decomposition failed: {e}")
    
    # Stationarity testing
    st.header("🧪 Stationarity Tests")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Series")
        try:
            stationarity_result = test_stationarity(data)
            
            # Create results table
            results_df = pd.DataFrame({
                'Test': ['ADF', 'KPSS'],
                'Statistic': [stationarity_result['adf_stat'], stationarity_result['kpss_stat']],
                'P-Value': [stationarity_result['adf_p'], stationarity_result['kpss_p']],
                'Null Hypothesis': ['Unit Root (Non-stationary)', 'Stationary'],
                'Reject H0': [stationarity_result['adf_p'] < 0.05, stationarity_result['kpss_p'] < 0.05]
            })
            
            st.dataframe(results_df)
            
            # Verdict with color coding
            verdict = stationarity_result['verdict'].upper()
            if verdict == 'STATIONARY':
                st.success(f"**Verdict: {verdict}** ✅")
            elif verdict == 'NON-STATIONARY':
                st.info(f"**Verdict: {verdict}** ℹ️")
            else:
                st.warning(f"**Verdict: {verdict}** ⚠️")
                
        except Exception as e:
            st.error(f"Stationarity test failed: {e}")
    
    with col2:
        st.subheader("First Differences")
        try:
            diff_data = data.diff().dropna()
            diff_stationarity = test_stationarity(diff_data)
            
            diff_results_df = pd.DataFrame({
                'Test': ['ADF', 'KPSS'],
                'Statistic': [diff_stationarity['adf_stat'], diff_stationarity['kpss_stat']],
                'P-Value': [diff_stationarity['adf_p'], diff_stationarity['kpss_p']],
                'Null Hypothesis': ['Unit Root (Non-stationary)', 'Stationary'],
                'Reject H0': [diff_stationarity['adf_p'] < 0.05, diff_stationarity['kpss_p'] < 0.05]
            })
            
            st.dataframe(diff_results_df)
            
            diff_verdict = diff_stationarity['verdict'].upper()
            if diff_verdict == 'STATIONARY':
                st.success(f"**Verdict: {diff_verdict}** ✅")
            elif diff_verdict == 'NON-STATIONARY':
                st.info(f"**Verdict: {diff_verdict}** ℹ️")
            else:
                st.warning(f"**Verdict: {diff_verdict}** ⚠️")
                
        except Exception as e:
            st.error(f"Differenced series stationarity test failed: {e}")
    
    # Structural breaks
    st.header("🔴 Structural Break Detection")
    
    try:
        breaks = detect_breaks(data, pen=penalty)
        
        if len(breaks) > 0:
            st.write(f"**{len(breaks)} structural breaks detected:**")
            for i, break_date in enumerate(breaks, 1):
                st.write(f"{i}. {break_date.strftime('%Y-%m-%d')}")
            
            # Plot series with breaks
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data.index, y=data.values,
                mode='lines', name=fred_series,
                line=dict(color='blue', width=1)
            ))
            
            # Add break points
            for break_date in breaks:
                fig.add_vline(
                    x=break_date, line_dash="dash", line_color="red",
                    annotation_text=f"Break: {break_date.strftime('%Y-%m')}"
                )
            
            fig.update_layout(
                title=f"Structural Breaks in {fred_series} (Penalty = {penalty})",
                xaxis_title="Date",
                yaxis_title="Value",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No structural breaks detected with current penalty parameter.")
            
        # Parameter sensitivity note
        st.info("""
        **Parameter Sensitivity Note:** 
        - Lower penalty → more breaks (higher sensitivity to changes)
        - Higher penalty → fewer breaks (less sensitivity, more robust)
        - Try different penalty values to see how results change
        """)
        
    except Exception as e:
        st.error(f"Structural break detection failed: {e}")
    
    # Parameter sensitivity analysis
    st.header("⚙️ Parameter Sensitivity Analysis")
    
    with st.expander("What this app reveals about parameter choices"):
        st.markdown("""
        **Key Insights from Interactive Analysis:**
        
        1. **Log Transformation**: 
           - Toggle to see how multiplicative vs additive seasonality affects decomposition
           - Retail sales, GDP: need log transform (growing seasonal amplitude)
           - Interest rates, unemployment: usually don't need log transform
        
        2. **Seasonal Period**:
           - Monthly data: period=12 captures annual cycles
           - Quarterly data: period=4 for annual patterns
           - Wrong period → seasonal component looks random
        
        3. **PELT Penalty**:
           - Low penalty (1-5): detects many breaks, may overfit to noise
           - High penalty (20-50): conservative, misses some true breaks
           - Optimal range often 5-15 for economic data
        
        4. **Block Size for Bootstrap**:
           - Too small (1-2): destroys autocorrelation, overconfident bands
           - Too large (20+): insufficient variation, too conservative
           - Sweet spot: 6-12 for quarterly data, 12-24 for monthly
        
        5. **Robust vs Non-robust STL**:
           - Robust: handles outliers (financial crises, COVID)
           - Non-robust: sensitive to extreme values
           - Compare 2008-2009 period with robust on/off
        """)

else:
    st.warning("Please enter your FRED API key in the sidebar to begin analysis.")
    st.markdown("""
    **Get started:**
    1. Obtain free API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)
    2. Enter key in sidebar
    3. Choose a series (or try the examples)
    4. Explore different decomposition methods and parameters
    """)

# Footer
st.markdown("---")
st.markdown("**Built for ECON 5200 | Time Series Analysis Dashboard**")

