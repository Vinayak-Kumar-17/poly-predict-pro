import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from engine import RegressionEngine, StatsEngine
import io

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

# Sidebar for controls
with st.sidebar:
    st.header("🛠 Configuration")
    upload_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    
    st.markdown("### Model Parameters")
    auto_degree = st.checkbox("Auto-optimize Polynomial Degree", value=True)
    manual_degree = st.slider("Manual Degree", 1, 10, 2, disabled=auto_degree)
    
    st.markdown("---")
    st.markdown("### Columns Mapping")
    x_col = st.text_input("X Axis Column", "x")
    y_col = st.text_input("Y Axis Column", "y")

# Main Logic
data_source = upload_file
if 'sample_data' in st.session_state and upload_file is None:
    data_source = io.StringIO(st.session_state['sample_data'])

if data_source is not None:
    df = pd.read_csv(data_source)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    if x_col in df.columns and y_col in df.columns:
        # Robust data cleaning
        temp_df = df[[x_col, y_col]].copy()
        
        # Handle X column (potentially dates)
        is_x_date = False
        try:
            # Try to convert to datetime
            temp_df[x_col] = pd.to_datetime(temp_df[x_col])
            is_x_date = True
            # Convert to ordinal for regression (numeric)
            base_date = temp_df[x_col].min()
            X_numeric = (temp_df[x_col] - base_date).dt.days.values
        except:
            # Fallback to standard numeric conversion
            temp_df[x_col] = pd.to_numeric(temp_df[x_col], errors='coerce')
            X_numeric = temp_df[x_col].values
            
        # Handle Y column (must be numeric)
        temp_df[y_col] = pd.to_numeric(temp_df[y_col], errors='coerce')
        y_numeric = temp_df[y_col].values
        
        # Filter NaN values
        mask = ~np.isnan(X_numeric) & ~np.isnan(y_numeric)
        X = X_numeric[mask]
        y = y_numeric[mask]
        
        if len(X) < 2:
            st.error("Error: Not enough valid numeric data points for regression.")
            st.stop()
            
        # Initialize Engine
        engine = RegressionEngine(degree=manual_degree if not auto_degree else 2)
        
        if auto_degree:
            with st.spinner("Optimizing degree..."):
                engine.find_best_degree(X, y)
                st.info(f"✅ Best Degree Found: **{engine.degree}**")
        
        engine.fit(X, y)
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
        st.error(f"Error: Columns '{x_col}' or '{y_col}' not found in the dataset.")
else:
    st.info("👋 Welcome! Please upload a CSV file to get started.")
    st.markdown("""
    ### Sample Data Format
    Your CSV should have at least two numerical columns. For example:
    | x | y |
    |---|---|
    | 1 | 2.1 |
    | 2 | 3.9 |
    | 3 | 9.2 |
    """)
    
    # Generate random sample data button
    if st.button("Generate & Use Sample Data"):
        sample_x = np.linspace(0, 10, 50)
        sample_y = 2 * (sample_x**2) - 5 * sample_x + 10 + np.random.normal(0, 10, 50)
        sample_df = pd.DataFrame({'x': sample_x, 'y': sample_y})
        
        # Save to buffer
        csv_buffer = io.StringIO()
        sample_df.to_csv(csv_buffer, index=False)
        st.session_state['sample_data'] = csv_buffer.getvalue()
        st.rerun()

# Handle sample data if generated
if 'sample_data' in st.session_state:
    df = pd.read_csv(io.StringIO(st.session_state['sample_data']))
    # Re-run logic for sample data...
    # (Actually it's better to just set use the buffer in the uploader-like logic)
    # For now, if sample_data is in state, we use it as if it was uploaded
    # ... logic above will handle it if we mock the upload_file
