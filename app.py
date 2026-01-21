import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from engine import RegressionEngine, StatsEngine
import yfinance as yf
from datetime import datetime
import calendar

# Page Config
st.set_page_config(
    page_title="PolyPredict Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1e1e2f 0%, #121212 100%);
        color: #ffffff;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    div[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #00bcd4;
        color: white;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0097a7;
        box-shadow: 0 0 15px rgba(0, 188, 212, 0.4);
    }
    .month-card {
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Nifty 50 Tickers and Sectors Mapping
NIFTY_50_TICKERS = {
    'ADANIENT.NS': 'Emerging Business', 'ADANIPORTS.NS': 'Infrastructure', 'APOLLOHOSP.NS': 'Healthcare',
    'ASIANPAINT.NS': 'Consumer Goods', 'AXISBANK.NS': 'Banking', 'BAJAJ-AUTO.NS': 'Automobile',
    'BAJAJFINSV.NS': 'Financial Services', 'BAJFINANCE.NS': 'Financial Services', 'BHARTIARTL.NS': 'Telecom',
    'BPCL.NS': 'Energy', 'BRITANNIA.NS': 'Consumer Goods', 'CIPLA.NS': 'Healthcare',
    'COALINDIA.NS': 'Energy', 'DIVISLAB.NS': 'Healthcare', 'DRREDDY.NS': 'Healthcare',
    'EICHERMOT.NS': 'Automobile', 'GRASIM.NS': 'Cement', 'HCLTECH.NS': 'IT',
    'HDFCBANK.NS': 'Banking', 'HDFCLIFE.NS': 'Financial Services', 'HEROMOTOCO.NS': 'Automobile',
    'HINDALCO.NS': 'Metals', 'HINDUNILVR.NS': 'Consumer Goods', 'ICICIBANK.NS': 'Banking',
    'INDUSINDBK.NS': 'Banking', 'INFY.NS': 'IT', 'ITC.NS': 'Consumer Goods',
    'JSWSTEEL.NS': 'Metals', 'KOTAKBANK.NS': 'Banking', 'LTIM.NS': 'IT',
    'LT.NS': 'Infrastructure', 'M&M.NS': 'Automobile', 'MARUTI.NS': 'Automobile',
    'NESTLEIND.NS': 'Consumer Goods', 'NTPC.NS': 'Energy', 'ONGC.NS': 'Energy',
    'POWERGRID.NS': 'Energy', 'RELIANCE.NS': 'Energy', 'SBILIFE.NS': 'Financial Services',
    'SBIN.NS': 'Banking', 'SUNPHARMA.NS': 'Healthcare', 'TATACONSUM.NS': 'Consumer Goods',
    'TATAMOTORS.NS': 'Automobile', 'TATASTEEL.NS': 'Metals', 'TCS.NS': 'IT',
    'TECHM.NS': 'IT', 'TITAN.NS': 'Consumer Goods', 'ULTRACEMCO.NS': 'Cement',
    'WIPRO.NS': 'IT'
}

# App Header
st.title("🚀 PolyPredict Pro")
st.markdown("---")

@st.cache_data
def run_stock_screener(month_list):
    # Fetch 10 years of data for all Nifty 50
    tickers = list(NIFTY_50_TICKERS.keys())
    # Use interval="1mo" for speed
    data = yf.download(tickers, period="10y", interval="1mo", group_by='ticker')
    
    upcoming_months = month_list
    results = []
    
    # Reuse reliability logic
    def get_internal_reliability(returns_group):
        mean = returns_group.mean()
        median = returns_group.median()
        win_rate = (returns_group > 0).mean()
        volatility = returns_group.std()
        
        score = 0
        if win_rate >= 0.65: score += 2
        elif win_rate >= 0.50: score += 1
        
        if volatility < abs(mean) and abs(mean) > 1: score += 1
        
        skew_factor = abs(mean - median)
        if skew_factor < 1.0: score += 2
        elif skew_factor < 2.5: score += 1
        return min(5, score)

    for ticker in tickers:
        try:
            # Monthly returns
            close_data = data[ticker]['Close'].dropna()
            returns = close_data.pct_change().dropna() * 100
            
            # Group by month
            df_ret = returns.to_frame(name='Return')
            df_ret['Month'] = df_ret.index.month
            
            ticker_stats = []
            for m in upcoming_months:
                m_returns = df_ret[df_ret['Month'] == m]['Return']
                if len(m_returns) < 5: continue
                
                mean_ret = m_returns.mean()
                median_ret = m_returns.median()
                win_rate = (m_returns > 0).mean() * 100
                rel_score = get_internal_reliability(m_returns)
                
                ticker_stats.append({
                    'Month': m,
                    'Ticker': ticker,
                    'Sector': NIFTY_50_TICKERS[ticker],
                    'Median': median_ret,
                    'Mean': mean_ret,
                    'Win_Rate': win_rate,
                    'Reliability': rel_score,
                    'Worst_Case': np.percentile(m_returns, 10),
                    'Best_Case': np.percentile(m_returns, 90),
                    'Outlier': abs(mean_ret - median_ret) > (1.5 * m_returns.std()) if m_returns.std() != 0 else False
                })
            
            if ticker_stats:
                results.append({
                    'Ticker': ticker,
                    'Sector': NIFTY_50_TICKERS[ticker],
                    'Reliability': np.mean([s['Reliability'] for s in ticker_stats]),
                    'Win_Rate': np.mean([s['Win_Rate'] for s in ticker_stats]),
                    'Median': np.mean([s['Median'] for s in ticker_stats]),
                    'Mean': np.mean([s['Mean'] for s in ticker_stats]),
                    'Worst_Case': np.mean([s['Worst_Case'] for s in ticker_stats]),
                    'Best_Case': np.mean([s['Best_Case'] for s in ticker_stats]),
                    'Outlier': any([s['Outlier'] for s in ticker_stats])
                })
        except:
            continue
            
    return pd.DataFrame(results)

@st.cache_data
def fetch_multi_stock_data(tickers, period):
    try:
        # Tickers should be a list
        data = yf.download(tickers, period=period)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

@st.cache_data
def calculate_seasonality_stats(price_df, ticker):
    if price_df is None:
        return None, None, None
        
    prices = price_df['Close']
    if isinstance(prices, pd.DataFrame):
        if ticker in prices.columns:
            prices = prices[ticker]
        else:
            return None, None, None

    # Calculate monthly returns
    monthly_prices = prices.resample('ME').last()
    monthly_returns = monthly_prices.pct_change() * 100
    
    df_returns = monthly_returns.reset_index()
    df_returns.columns = ['Date', 'Return']
    df_returns['Year'] = df_returns['Date'].dt.year
    df_returns['Month'] = df_returns['Date'].dt.month
    df_returns['Month_Name'] = df_returns['Date'].dt.month_name()
    
    # Filter out NaNs from return calculation
    df_returns = df_returns.dropna()
    
    # Pivot for heatmap
    heatmap_df = df_returns.pivot(index='Year', columns='Month', values='Return')
    heatmap_df.columns = [calendar.month_name[m] for m in heatmap_df.columns]
    
    # Advanced Stats per month
    def get_reliability_score(group):
        mean = group.mean()
        median = group.median()
        win_rate = (group > 0).mean()
        volatility = group.std()
        
        # Reliability logic:
        # 1. Win Rate must be high (>60% for 5 stars, >50% for 4, etc.)
        # 2. Volatility should be low relative to mean
        # 3. Mean and Median should be close (if mean >> median, there's a positive outlier skew)
        
        score = 0
        if win_rate >= 0.65: score += 2
        elif win_rate >= 0.50: score += 1
        
        if volatility < abs(mean) and abs(mean) > 1: score += 1
        
        # Skewness check: If mean and median differ by more than 2x or 3%, it's unreliable
        skew_factor = abs(mean - median)
        if skew_factor < 1.0: score += 2
        elif skew_factor < 2.5: score += 1
        
        # Max score is 5
        return min(5, score)

    month_stats = df_returns.groupby('Month')['Return'].agg([
        'mean', 'median', 'std', 'count', 
        lambda x: (x > 0).sum()
    ])
    month_stats.columns = ['Mean', 'Median', 'Volatility', 'Count', 'Wins']
    month_stats['Win_Rate'] = (month_stats['Wins'] / month_stats['Count']) * 100
    month_stats['Reliability'] = df_returns.groupby('Month')['Return'].apply(get_reliability_score)
    month_stats['Month_Name'] = [calendar.month_name[m] for m in month_stats.index]
    
    # Outlier detection (Z-score > 2.5)
    df_returns['Z_Score'] = df_returns.groupby('Month')['Return'].transform(lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else 1))
    outliers = df_returns[df_returns['Z_Score'].abs() > 2.5]
    
    # Financial Truth: CAGR (Geometric Mean Monthly Return)
    # Total Growth = Price_End / Price_Start
    # CAGR_Monthly = (Total Growth ^ (1/n_months)) - 1
    total_months = len(df_returns)
    if total_months > 0:
        total_growth = prices.iloc[-1] / prices.iloc[0]
        cagr_monthly = (pow(total_growth, 1/total_months) - 1) * 100
    else:
        cagr_monthly = 0
    
    return heatmap_df, month_stats, df_returns, outliers, cagr_monthly

@st.cache_data
def process_data(df, x_col, y_col, use_log=False):
    # Robust data cleaning using Pandas
    # Use loc to avoid SettingWithCopyWarning
    temp_df = df[[x_col, y_col]].copy()
    
    # Handle X column (potentially dates)
    is_x_date = False
    base_date = None
    try:
        # Try to convert to datetime
        temp_df[x_col] = pd.to_datetime(temp_df[x_col], errors='raise')
        is_x_date = True
        # Convert to ordinal for regression (numeric)
        base_date = temp_df[x_col].min()
        X_series = (temp_df[x_col] - base_date).dt.days
    except Exception:
        # Fallback to standard numeric conversion
        is_x_date = False
        X_series = pd.to_numeric(temp_df[x_col], errors='coerce')
        
    # Handle Y column (must be numeric)
    y_series = pd.to_numeric(temp_df[y_col], errors='coerce')
    
    # Combined cleaner DataFrame
    clean_df = pd.DataFrame({'x_val': X_series, 'y_val': y_series})
    clean_df = clean_df.dropna()
    
    # Financial Mode: Log Transformation
    if use_log:
        clean_df = clean_df[clean_df['y_val'] > 0]
        clean_df['y_val'] = np.log(clean_df['y_val'].astype(float))
    
    return clean_df, is_x_date, base_date

@st.cache_resource
def train_model(X, y, auto_degree, manual_degree, alpha=1.0):
    engine = RegressionEngine(degree=manual_degree if not auto_degree else 2, alpha=alpha)
    if auto_degree:
        engine.find_best_degree(X, y)
    engine.fit(X, y)
    return engine

# Sidebar for controls
with st.sidebar:
    st.header("🛠 Configuration")
    
    data_mode = st.radio("Select Data Source", ["Upload CSV", "Stock Market", "Sample Data"], index=1)
    
    df = None
    x_col_default = "x"
    y_col_default = "y"
    
    if data_mode == "Upload CSV":
        upload_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
        if upload_file:
            df = pd.read_csv(upload_file)
    elif data_mode == "Stock Market":
        ticker_input = st.text_input("Ticker Symbols (comma-separated for Correlation)", "ITC.NS, TATAELXSI.NS")
        tickers_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        period = st.selectbox("Market History Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=4)
        with st.spinner("Fetching market data..."):
            raw_data = fetch_multi_stock_data(tickers_list, period)
            
        if raw_data is not None:
            # For Tab 1 (Regression), we pick one ticker to focus on
            focal_ticker = st.selectbox("Select Ticker for Detailed Analysis", tickers_list)
            
            # Simplified df for regression logic (compatible with legacy)
            if isinstance(raw_data['Close'], pd.Series):
                df = raw_data[['Close']].copy().reset_index()
                df.columns = ['Date', 'Close']
            else:
                df = raw_data['Close'][[focal_ticker]].copy().reset_index()
                df.columns = ['Date', 'Close']
                
            x_col_default = "Date"
            y_col_default = "Close"
        else:
            st.error("No data found for tickers.")
    else:
        # Sample Data Mode
        if 'sample_data' not in st.session_state:
            sample_x = np.linspace(0, 10, 50)
            sample_y = 2 * (sample_x**2) - 5 * sample_x + 10 + np.random.normal(0, 10, 50)
            st.session_state['sample_data'] = pd.DataFrame({'x': sample_x, 'y': sample_y})
        df = st.session_state['sample_data']
        x_col_default = "x"
        y_col_default = "y"

    st.markdown("---")
    st.markdown("### Model Parameters")
    auto_degree = st.checkbox("Auto-optimize Polynomial Degree", value=True)
    manual_degree = st.slider("Manual Degree", 1, 10, 2, disabled=auto_degree)
    
    st.markdown("### Stability & Financial Mode")
    use_log = st.checkbox("Financial Mode (Log-Regression)", value=True, help="Ensures positive predictions.")
    alpha = st.slider("Regularization (Ridge Alpha)", 0.0, 10.0, 1.0, step=0.1)

    st.markdown("---")
    st.markdown("### Columns Mapping")
    x_col = st.text_input("X Axis Column", x_col_default)
    y_col = st.text_input("Y Axis Column", y_col_default)

# --- TABS ---
predictive_tab, seasonality_tab, screener_tab = st.tabs(["🔮 Predictive Insights", "📅 Seasonality Analysis", "🔍 Stock Screener"])
with predictive_tab:
    if df is not None:
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        if x_col in df.columns and y_col in df.columns:
            with st.spinner("Processing data..."):
                clean_df, is_x_date, base_date = process_data(df, x_col, y_col, use_log=use_log)
            
            # Explicitly ensure they are floats for the engine
            X = clean_df['x_val'].astype(float).values
            y = clean_df['y_val'].astype(float).values
            
            if len(X) < 2:
                st.error("Error: Not enough valid numeric data points for regression after cleaning.")
                st.stop()
                
            # Initialize and train Engine (Cached)
            with st.spinner("Training Model..."):
                engine = train_model(X, y, auto_degree, manual_degree, alpha=alpha)
            
            y_pred_numeric = engine.predict(X)
            
            # Inverse transform for metrics if in log mode
            y_actual_display = y
            y_pred_display = y_pred_numeric
            if use_log:
                y_actual_display = np.exp(y)
                y_pred_display = np.exp(y_pred_numeric)
                
            stats = StatsEngine.compute_all(y_actual_display, y_pred_display, X.reshape(-1, 1))
            
            # Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R² Score", f"{stats['r2']:.4f}")
            col2.metric("Adj. R²", f"{stats['adj_r2']:.4f}")
            col3.metric("MSE", f"{stats['mse']:.4f}")
            col4.metric("RMSE", f"{stats['rmse']:.4f}")
            
            # Visualization
            st.subheader("📈 Regression Analysis")
            
            # Create regression line data
            min_x, max_x = float(np.min(X)), float(np.max(X))
            line_x, line_y_numeric = engine.get_line_data(min_x, max_x)
            
            # Inverse transform for line if in log mode
            line_y = line_y_numeric
            if use_log:
                line_y = np.exp(line_y_numeric).tolist()
            else:
                line_y = np.maximum(0, line_y).tolist()

            # Mapping numeric X back to dates for visualization if needed
            X_plot = X
            line_x_plot = line_x
            if is_x_date:
                X_plot = [base_date + pd.Timedelta(days=int(d)) for d in X]
                line_x_plot = [base_date + pd.Timedelta(days=int(d)) for d in line_x]

            fig = go.Figure()
            
            # Scatter for original data
            fig.add_trace(go.Scatter(
                x=X_plot, y=y_actual_display, 
                mode='markers', 
                name='Original Data',
                marker=dict(color='#00bcd4', size=8, opacity=0.6)
            ))
            
            # Line for regression
            fig.add_trace(go.Scatter(
                x=line_x_plot, y=line_y, 
                mode='lines', 
                name=f'Fit (Degree {engine.degree})',
                line=dict(color='#ff4081', width=3)
            ))
            
            fig.update_layout(
                template="plotly_dark",
                xaxis_title=x_col if not is_x_date else f"{x_col} (Time Series)",
                yaxis_title=y_col,
                height=600,
                hovermode="x unified",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction Tool
            st.markdown("---")
            st.subheader("🔮 Predictive Insights")
            predict_col1, predict_col2 = st.columns([1, 2])
            
            with predict_col1:
                if is_x_date:
                    # Use date input for prediction
                    last_date = base_date + pd.Timedelta(days=int(X.max()))
                    target_date = st.date_input("Select Date for Prediction", value=last_date + pd.Timedelta(days=365))
                    input_val = (pd.to_datetime(target_date) - base_date).days
                else:
                    input_val = st.number_input(f"Enter {x_col} to predict", value=float(max_x + (max_x - min_x) * 0.1))
                
                if st.button("Predict Future Value"):
                    pred_numeric = engine.predict(np.array([[input_val]]))[0]
                    pred = np.exp(pred_numeric) if use_log else max(0, pred_numeric)
                    
                    if is_x_date:
                        st.success(f"Predicted **{y_col}** for {target_date}: `{pred:.4f}`")
                    else:
                        st.success(f"Predicted **{y_col}** for {input_val}: `{pred:.4f}`")
            
            with predict_col2:
                st.info("The model uses Ridge Regression for stability. 'Financial Mode' ensures stock prices remain positive during extrapolation.")

        else:
            st.warning(f"Waiting for valid columns: '{x_col}' and '{y_col}'. You can adjust them in the sidebar.")
    else:
        st.info("👋 Welcome! Please select a data source in the sidebar to get started.")

with seasonality_tab:
    if data_mode == "Stock Market" and raw_data is not None:
        st.subheader(f"📅 Seasonality Analysis: {focal_ticker}")
        
        # New signature with cagr_monthly
        heatmap_df, month_stats, raw_returns, outliers, cagr_monthly = calculate_seasonality_stats(raw_data, focal_ticker)
        
        # Win-Rate & Volatility Row
        st.markdown("### 📊 Performance Summary")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        # Find best/worst month
        best_month = month_stats['Mean'].idxmax()
        worst_month = month_stats['Mean'].idxmin()
        best_winrate = month_stats['Win_Rate'].idxmax()
        
        # Combined Arithmetic and Compounded Metric
        m_col1.metric("Avg. Monthly Return", f"{month_stats['Mean'].mean():.2f}%", help="Arithmetic Mean: Simple average of monthly swings.")
        m_col1.markdown(f"**Compounded (CAGR):** `{cagr_monthly:.2f}%` 💎", help="Geometric Mean: The true growth rate your wealth feels after accounting for volatility drag.")
        
        m_col2.metric("Best Month (Avg)", f"{calendar.month_name[best_month]}", f"{month_stats.loc[best_month, 'Mean']:.2f}%")
        m_col3.metric("Worst Month (Avg)", f"{calendar.month_name[worst_month]}", f"{month_stats.loc[worst_month, 'Mean']:.2f}%", delta_color="inverse")
        m_col4.metric("Highest Win-Rate", f"{calendar.month_name[best_winrate]}", f"{month_stats.loc[best_winrate, 'Win_Rate']:.1f}%")
        
        st.info("💡 **Why two averages?** Arithmetic Mean shows the average monthly swing. **Compounded (CAGR)** shows the real rate of growth. If a stock is highly volatile (like SBIN), the Arithmetic Mean is usually higher, but the CAGR is what actually builds wealth.")
        
        # --- RED FLAG DETECTION SYSTEM ---
        st.markdown("---")
        st.markdown("#### 🚩 Reliability Insights & Red Flags")
        
        flagged_months = month_stats[abs(month_stats['Mean'] - month_stats['Median']) > 2.5]
        
        if not flagged_months.empty or not outliers.empty:
            for idx, row in flagged_months.iterrows():
                m_name = calendar.month_name[idx]
                st.warning(f"""
                **⚠️ Reliability Warning: {m_name}**
                * **Skew Detected:** Average return is `{row['Mean']:.2f}%` but Median is only `{row['Median']:.2f}%`.
                * **Reason:** This month's average is likely inflated/deflated by a specific outlier year.
                * **Recommendation:** Look at Win Rate (**{row['Win_Rate']:.1f}%**) for a truer picture of consistency.
                """)
            
            if not outliers.empty:
                with st.expander("🔍 View Specific Outliers (Historical Anomalies)"):
                    st.write("These specific months had returns more than 2.5 standard deviations from the mean:")
                    st.dataframe(outliers[['Date', 'Return', 'Z_Score']].sort_values(by='Z_Score', ascending=False), use_container_width=True)
        else:
            st.success("✅ Seasonal patterns appear statistically consistent. (Low divergence between Mean and Median)")

        # --- OPTION 4: PERFORMANCE BREAKDOWN TABLE ---
        st.markdown("---")
        st.markdown("#### 📋 Month-by-Month Reliability Breakdown")
        
        breakdown_df = month_stats.copy()
        breakdown_df['Reliability_Stars'] = breakdown_df['Reliability'].apply(lambda x: "⭐" * int(x) if x > 0 else "⚠️")
        
        # Presentation formatting
        disp_df = breakdown_df[['Month_Name', 'Mean', 'Median', 'Win_Rate', 'Volatility', 'Reliability_Stars']].copy()
        disp_df.columns = ['Month', 'Avg Return %', 'Median %', 'Win Rate %', 'Volatility %', 'Reliability']
        
        st.dataframe(
            disp_df.style.background_gradient(subset=['Avg Return %'], cmap='RdYlGn')
            .format({
                'Avg Return %': '{:.2f}',
                'Median %': '{:.2f}',
                'Win Rate %': '{:.1f}',
                'Volatility %': '{:.2f}'
            }),
            use_container_width=True
        )
        
        st.caption("**Legend:** ⭐⭐⭐⭐⭐ = Highly reliable | ⚠️ = Highly Skewed / Low History")

        # Rolling Window Comparison
        st.markdown("---")
        st.markdown("### 🕒 Pattern Evolution (Rolling Windows)")
        
        if len(raw_returns['Year'].unique()) > 3:
            recent_3y = raw_returns[raw_returns['Year'] >= raw_returns['Year'].max() - 2]
            r3y_stats = recent_3y.groupby('Month')['Return'].mean()
            
            comparison_df = pd.DataFrame({
                'Month': [calendar.month_name[i] for i in range(1, 13)],
                'Full History Avg (%)': [month_stats.loc[i, 'Mean'] for i in range(1, 13)],
                'Recent 3Y Avg (%)': [r3y_stats.get(i, np.nan) for i in range(1, 13)]
            })
            comparison_df['Shift (%)'] = comparison_df['Recent 3Y Avg (%)'] - comparison_df['Full History Avg (%)']
            
            st.dataframe(comparison_df.style.background_gradient(subset=['Shift (%)'], cmap='RdYlGn'), use_container_width=True)
            st.info("💡 A positive 'Shift' means the month is performing better in the last 3 years compared to its long-term history.")
        else:
            st.info("Not enough history for Rolling Comparison (requires >3 years of data).")

        # Visuals
        st.markdown("---")
        chart_col1, chart_col2 = st.columns([2, 1])
        
        with chart_col1:
            st.markdown("#### 🌡️ Monthly Return Heatmap (%)")
            fig_heat = px.imshow(
                heatmap_df,
                labels=dict(x="Month", y="Year", color="Return %"),
                x=heatmap_df.columns,
                y=heatmap_df.index,
                color_continuous_scale="RdYlGn",
                aspect="auto",
                text_auto=".1f"
            )
            fig_heat.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig_heat, use_container_width=True)
            
        with chart_col2:
            st.markdown("#### 📦 Return Distribution")
            fig_box = px.box(
                raw_returns,
                x="Month_Name",
                y="Return",
                points="all",
                color="Month_Name",
                category_orders={"Month_Name": [calendar.month_name[i] for i in range(1, 13)]}
            )
            fig_box.update_layout(template="plotly_dark", showlegend=False, height=500, xaxis_title="")
            st.plotly_chart(fig_box, use_container_width=True)
            
        # Rolling & Multi-Stock Logic
        st.markdown("---")
        st.markdown("### 🔍 Advanced Insights")
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            # Volatility is now a bar chart but we can use our new stats
            st.markdown("#### 📉 Risk Level (Volatility) per Month")
            fig_vol = px.bar(
                month_stats.reset_index(),
                x="Month_Name",
                y="Volatility",
                color="Volatility",
                color_continuous_scale="Purples",
                category_orders={"Month_Name": [calendar.month_name[i] for i in range(1, 13)]}
            )
            fig_vol.update_layout(template="plotly_dark", height=400, xaxis_title="")
            st.plotly_chart(fig_vol, use_container_width=True)
            
        with adv_col2:
             if len(tickers_list) > 1:
                st.markdown("#### 🛰️ Multi-Stock Correlation")
                # Calculate correlation of daily returns for the selected period
                corr_df = raw_data['Close'].pct_change().corr()
                fig_corr = px.imshow(
                    corr_df,
                    text_auto=".2f",
                    color_continuous_scale="Viridis",
                    aspect="auto"
                )
                fig_corr.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_corr, use_container_width=True)
             else:
                st.info("Add more tickers in the sidebar to see the Correlation Matrix.")
                
        # Export logic
        st.markdown("---")
        csv_data = month_stats.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Seasonality CSV Report",
            data=csv_data,
            file_name=f"{focal_ticker}_seasonality.csv",
            mime="text/csv",
        )
with screener_tab:
    st.subheader("🔍 Smart Stock Screener (Nifty 50)")
    st.markdown("""
    This screener identifies stocks with the strongest **historical seasonal patterns** for the **selected months**. 
    It ranks them using a composite **Smart Score** that balances returns, consistency, and risk.
    """)
    
    # Month Selection UI
    current_month_idx = datetime.now().month
    month_options = list(calendar.month_name)[1:]
    
    # Default to next 3 months
    default_months = [month_options[(current_month_idx + i - 1) % 12] for i in range(1, 4)]
    
    selected_month_names = st.multiselect(
        "Select Months to Analyze",
        options=month_options,
        default=default_months,
        help="The screener will calculate performance based on the historical data for these specific months."
    )
    
    selected_month_indices = [month_options.index(m) + 1 for m in selected_month_names]
    
    # Session State for persistence
    if 'screener_res' not in st.session_state:
        st.session_state.screener_res = None
    if 'screener_months' not in st.session_state:
        st.session_state.screener_months = None

    if st.button("🚀 Run Seasonal Screener (Analyzes 10Y Data)"):
        if not selected_month_indices:
            st.error("Please select at least one month to analyze.")
        else:
            with st.spinner("Analyzing Nifty 50 historical patterns... This may take a minute."):
                st.session_state.screener_res = run_stock_screener(selected_month_indices)
                st.session_state.screener_months = selected_month_names
                
    if st.session_state.screener_res is not None:
        screener_df = st.session_state.screener_res
        st.success(f"Analysis complete for: **{', '.join(st.session_state.screener_months)}**")
        
        # --- TABLE 1: TOP 10 BY RELIABILITY ---
        st.markdown("### 🏆 Table 1: Top 10 Stocks by Reliability & Median")
        
        # Ranking Logic: Reliability (Stars) -> Median -> Win Rate
        top_reliable = screener_df.sort_values(by=['Reliability', 'Median', 'Win_Rate'], ascending=False).head(10).copy()
        top_reliable.insert(0, 'Rank', range(1, 11))
        
        # Convert Reliability to Stars
        top_reliable['Reliability_Stars'] = top_reliable['Reliability'].apply(lambda x: "⭐" * int(round(x)) if x >= 1 else "⚠️")
        top_reliable['Outlier Flag'] = top_reliable['Outlier'].apply(lambda x: "🚩" if x else "✅")
        
        st.dataframe(
            top_reliable[['Rank', 'Ticker', 'Sector', 'Reliability_Stars', 'Median', 'Win_Rate', 'Worst_Case', 'Best_Case', 'Outlier Flag']]
            .style.background_gradient(subset=['Median', 'Win_Rate'], cmap='RdYlGn')
            .format({'Median': '{:.2f}%', 'Win_Rate': '{:.1f}%', 'Worst_Case': '{:.2f}%', 'Best_Case': '{:.2f}%'}),
            use_container_width=True,
            hide_index=True
        )
        st.caption("**Ranking Strategy:** Prioritizes consistency (Reliability) and median performance over simple averages.")
        
        # --- TABLE 2: TOP 10 BY WIN RATE ---
        st.markdown("### 📈 Table 2: Top 10 Stocks by Win Rate (Consistency Leaders)")
        top_win = screener_df[screener_df['Win_Rate'] >= 60].sort_values(by=['Win_Rate', 'Reliability'], ascending=False).head(10).copy()
        if not top_win.empty:
            top_win.insert(0, 'Rank', range(1, len(top_win) + 1))
            top_win['Reliability_Stars'] = top_win['Reliability'].apply(lambda x: "⭐" * int(round(x)) if x >= 1 else "⚠️")
            top_win['Outlier Flag'] = top_win['Outlier'].apply(lambda x: "🚩" if x else "✅")
            
            st.dataframe(
                top_win[['Rank', 'Ticker', 'Sector', 'Win_Rate', 'Reliability_Stars', 'Median', 'Worst_Case', 'Outlier Flag']]
                .style.background_gradient(subset=['Win_Rate', 'Median'], cmap='Greens')
                .format({'Median': '{:.2f}%', 'Win_Rate': '{:.1f}%', 'Worst_Case': '{:.2f}%'}),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No stocks found with a Win Rate ≥ 60% for the upcoming period.")

        # --- TABLE 3: TOP 5 SECTORS ---
        st.markdown("### 🏢 Table 3: Top 5 Sectors by Seasonal Performance")
        
        sector_stats = screener_df.groupby('Sector').agg({
            'Win_Rate': 'mean',
            'Median': 'mean',
            'Reliability': 'mean',
            'Ticker': 'count'
        }).reset_index()
        
        # Find top performers per sector based on reliability
        def get_top_performers(sector):
            t_performers = screener_df[screener_df['Sector'] == sector].sort_values(by=['Reliability', 'Median'], ascending=False).head(3)['Ticker'].tolist()
            return ", ".join(t_performers)
        
        sector_stats['Top Performers'] = sector_stats['Sector'].apply(get_top_performers)
        sector_stats = sector_stats.sort_values(by=['Reliability', 'Win_Rate'], ascending=False).head(5).copy()
        sector_stats.insert(0, 'Rank', range(1, 6))
        
        disp_sectors = sector_stats.rename(columns={
            'Win_Rate': 'Sect. Win Rate %',
            'Median': 'Sect. Median %',
            'Reliability': 'Sect. Reliability',
            'Ticker': 'Count'
        })
        
        st.dataframe(
            disp_sectors.style.background_gradient(subset=['Sect. Win Rate %', 'Sect. Median %'], cmap='Blues')
            .format({'Sect. Win Rate %': '{:.1f}', 'Sect. Median %': '{:.2f}', 'Sect. Reliability': '{:.1f}'}),
            use_container_width=True,
            hide_index=True
        )
        
        # Download insights
        st.markdown("---")
        csv_screener = screener_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Screener Report (CSV)",
            data=csv_screener,
            file_name="nifty_seasonal_screener.csv",
            mime="text/csv",
        )
    else:
        st.info("Click the button above to start the multi-year seasonal analysis for Nifty 50 stocks.")
