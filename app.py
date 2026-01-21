import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from engine import RegressionEngine, StatsEngine
import io
import yfinance as yf

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
</style>
""", unsafe_allow_html=True)

# App Header
st.title("🚀 PolyPredict Pro")
st.markdown("---")

@st.cache_data
def fetch_stock_data(ticker, period):
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            return None
        # Handle multi-level columns if any, and simplify
        df = data[['Close']].copy()
        df = df.reset_index()
        df.columns = ['Date', 'Close']
        return df
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

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
        ticker = st.text_input("Ticker Symbol (e.g. ITC.NS, AAPL, RELIANCE.NS)", "ITC.NS")
        period = st.selectbox("Market History Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=4)
        with st.spinner("Fetching market data..."):
            df = fetch_stock_data(ticker, period)
        x_col_default = "Date"
        y_col_default = "Close"
    else:
        # Sample Data Mode
        if 'sample_data' not in st.session_state:
            sample_x = np.linspace(0, 10, 50)
            sample_y = 2 * (sample_x**2) - 5 * sample_x + 10 + np.random.normal(0, 10, 50)
            st.session_state['sample_data'] = pd.DataFrame({'x': sample_x, 'y': sample_y})
        df = st.session_state['sample_data']
        x_col_default = "x"
        y_col_default = "y"

    st.markdown("### Model Parameters")
    auto_degree = st.checkbox("Auto-optimize Polynomial Degree", value=True)
    manual_degree = st.slider("Manual Degree", 1, 10, 2, disabled=auto_degree)
    
    st.markdown("---")
    st.markdown("### Columns Mapping")
    x_col = st.text_input("X Axis Column", x_col_default)
    y_col = st.text_input("Y Axis Column", y_col_default)

@st.cache_data
def process_data(df, x_col, y_col):
    # Robust data cleaning using Pandas
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
    except:
        # Fallback to standard numeric conversion
        is_x_date = False
        X_series = pd.to_numeric(temp_df[x_col], errors='coerce')
        
    # Handle Y column (must be numeric)
    y_series = pd.to_numeric(temp_df[y_col], errors='coerce')
    
    # Combined cleaner DataFrame
    clean_df = pd.DataFrame({'x_val': X_series, 'y_val': y_series})
    clean_df = clean_df.dropna()
    
    return clean_df, is_x_date, base_date

@st.cache_resource
def train_model(X, y, auto_degree, manual_degree):
    engine = RegressionEngine(degree=manual_degree if not auto_degree else 2)
    if auto_degree:
        engine.find_best_degree(X, y)
    engine.fit(X, y)
    return engine

# Main Logic
if df is not None:
    st.subheader("📊 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    if x_col in df.columns and y_col in df.columns:
        with st.spinner("Processing data..."):
            clean_df, is_x_date, base_date = process_data(df, x_col, y_col)
        
        # Explicitly ensure they are floats for the engine
        X = clean_df['x_val'].astype(float).values
        y = clean_df['y_val'].astype(float).values
        
        if len(X) < 2:
            st.error("Error: Not enough valid numeric data points for regression after cleaning.")
            st.stop()
            
        # Initialize and train Engine (Cached)
        with st.spinner("Training Model..."):
            engine = train_model(X, y, auto_degree, manual_degree)
        
        y_pred = engine.predict(X)
        stats = StatsEngine.compute_all(y, y_pred, X.reshape(-1, 1))
        
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
        line_x, line_y = engine.get_line_data(min_x, max_x)
        
        # Mapping numeric X back to dates for visualization if needed
        X_plot = X
        line_x_plot = line_x
        if is_x_date:
            X_plot = [base_date + pd.Timedelta(days=int(d)) for d in X]
            line_x_plot = [base_date + pd.Timedelta(days=int(d)) for d in line_x]

        fig = go.Figure()
        
        # Scatter for original data
        fig.add_trace(go.Scatter(
            x=X_plot, y=y, 
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
                last_date = base_date + pd.Timedelta(days=int(max_x))
                target_date = st.date_input("Select Date for Prediction", value=last_date + pd.Timedelta(days=30))
                input_val = (pd.to_datetime(target_date) - base_date).days
            else:
                input_val = st.number_input(f"Enter {x_col} to predict", value=float(max_x + (max_x - min_x) * 0.1))
            
            if st.button("Predict Future Value"):
                pred = engine.predict(np.array([[input_val]]))[0]
                if is_x_date:
                    st.success(f"Predicted **{y_col}** for {target_date}: `{pred:.4f}`")
                else:
                    st.success(f"Predicted **{y_col}** for {input_val}: `{pred:.4f}`")
        
        with predict_col2:
            st.info("Input a value to see the model's projection beyond the current dataset range. This use polynomial logic to estimate values.")

    else:
        st.warning(f"Waiting for valid columns: '{x_col}' and '{y_col}'. You can adjust them in the sidebar.")
else:
    st.info("👋 Welcome! Please select a data source in the sidebar to get started.")
