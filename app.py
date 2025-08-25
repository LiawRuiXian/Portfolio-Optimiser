import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data import fetch_price_data, price_to_returns
from optimiser import optimize_portfolio
from backtest import run_dca_backtest, calculate_performance_metrics
from visual.plot import plot_portfolio_vs_benchmark, plot_efficient_frontier
import plotly.express as px

# --- Page Config ---
st.set_page_config(
    page_title="Portfolio Optimizer Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Dark Theme CSS ---
st.markdown("""
<style>
.stApp { background-color: #0F111A; color: #E0E0E0; font-family: 'Segoe UI', sans-serif; }
.main-header { font-size:3rem; color:#FF7F50; text-align:center; margin-bottom:1rem; font-weight:700; text-shadow:0px 2px 4px rgba(0,0,0,0.6);}
.sub-header { font-size:1.8rem; color:#FF7F50; margin-bottom:1rem; font-weight:600; border-bottom:2px solid #FF7F50; padding-bottom:0.5rem; }
.metric-card { background-color:#1A1C2E; padding:1.5rem; border-radius:0.75rem; box-shadow:0 6px 10px rgba(0,0,0,0.5); margin-bottom:1rem; border-left:5px solid #FF7F50; color:#FF7F50; }
.stButton>button { width:100%; background-color:#FF7F50; color:#0F111A; font-weight:600; padding:0.8rem; border-radius:0.5rem; border:none; transition:all 0.3s ease;}
.stButton>button:hover { background-color:#FF9F7F; transform:translateY(-2px); box-shadow:0 6px 12px rgba(255,127,80,0.4); }
.divider { height:2px; background:linear-gradient(to right, transparent, #FF7F50, transparent); margin:1.5rem 0; }
.dataframe { border-radius:0.5rem; background-color:#1A1C2E !important; color:#E0E0E0 !important; }
th { background-color:#FF7F50 !important; color:#0F111A !important; font-weight:700; }
td { background-color:#1A1C2E !important; color:#E0E0E0 !important; border-bottom:1px solid #444 !important; }
tr:hover { background-color:#2A2A44 !important; }
.streamlit-expanderHeader { background-color:#1A1C2E; color:#FF7F50; border-radius:0.5rem; padding:0.5rem 1rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 class='main-header'>📊 Portfolio Optimiser Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# --- Sidebar Inputs ---
with st.sidebar:
    st.markdown("### ⚙️ Portfolio Configuration")
    tickers_input = st.text_input("Enter comma-separated tickers (e.g., AAPL, MSFT, GOOGL)", "AAPL, MSFT, GOOG, AMZN", label_visibility="collapsed")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"), min_value=pd.to_datetime("2015-01-01"), max_value=pd.to_datetime("today"))
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("today"), min_value=pd.to_datetime("2015-01-01"), max_value=pd.to_datetime("today"))
    
    freq = st.selectbox("Return Frequency", ["Daily", "Weekly", "Monthly"])
    
    dca_interval = st.selectbox("DCA Interval", ["Monthly", "Weekly", "Daily"])

    with st.expander("🔧 Optimization Settings", expanded=True):
        objective = st.selectbox("Objective", [
            "Maximize Sharpe Ratio",
            "Minimize Volatility",
            "Minimize Volatility for Target Return",
            "Maximize Sortino Ratio",
            "Minimize CVaR",
            "Minimize Maximum Drawdown"
        ])
        target_return = None
        if objective == "Minimize Volatility for Target Return":
            target_return = st.slider("Target Annual Return (%)", 0.0, 50.0, 10.0, 0.5)/100

    st.info("Backtesting uses Dollar-Cost Averaging (DCA) only")

# --- Run Button ---
run_opt = st.button("🚀 Run Portfolio Optimization", use_container_width=True)

interval_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}

# --- Helper Functions ---
@st.cache_data
def fetch_prices_safe(tickers, start, end):
    return fetch_price_data(tickers, start=start, end=end)

def validate_tickers(prices, tickers):
    if prices is None or prices.empty:
        return [], tickers
    
    valid = [t for t in tickers if t in prices.columns]
    invalid = [t for t in tickers if t not in prices.columns]
    return valid, invalid

# --- Run Optimization ---
if run_opt and tickers:
    try:
        with st.spinner("Fetching data and optimizing portfolio..."):
            prices = fetch_prices_safe(tickers, start=start_date, end=end_date)
            
            # Check if prices is None or empty
            if prices is None or prices.empty:
                st.error("Failed to fetch price data. Please check your tickers and date range.")
                st.stop()
            
            # Validate tickers
            valid_tickers, invalid_tickers = validate_tickers(prices, tickers)

            if invalid_tickers:
                st.warning(f"The following tickers are invalid or have no data: {', '.join(invalid_tickers)}")
            
            if not valid_tickers:
                st.error("No valid tickers to optimize. Please check your input.")
                st.stop()

            prices = prices[valid_tickers]
            rets = price_to_returns(prices, interval_map[freq])
            weights_res = optimize_portfolio(rets, objective=objective, target_return=target_return, freq=freq)

            # Benchmark
            benchmark_prices = fetch_prices_safe(["VOO"], start=start_date, end=end_date)

            if benchmark_prices is None or benchmark_prices.empty:
                benchmark_prices = None

        st.success("✅ Optimization completed successfully!")
        tabs = st.tabs(["📊 Portfolio Overview", "📈 Backtest Results", "📐 Efficient Frontier"])

        # --- Portfolio Overview ---
        with tabs[0]:
            st.markdown("<h2 class='sub-header'>Optimal Portfolio Allocation</h2>", unsafe_allow_html=True)
            # Convert weights to DataFrame
            weights_df = pd.DataFrame(weights_res["weights"], index=valid_tickers, columns=["Weight"])
            weights_df["Weight"] = pd.to_numeric(weights_df["Weight"], errors='coerce')

            # Create a pie chart
            fig = px.pie(weights_df, 
                        names=weights_df.index, 
                        values="Weight", 
                        color=weights_df.index,  # optional: different colors per ticker
                        color_discrete_sequence=px.colors.qualitative.Pastel)  # optional color palette

            fig.update_traces(textinfo='percent+label')  # show both percentage and label
            fig.update_layout(
                margin=dict(t=50, b=50, l=50, r=50),  # increase margin in pixels
                height=600,  # optional: adjust chart height
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Backtest ---
        with tabs[1]: 
            
            st.markdown("<h3 class='sub-header'>Performance metrics</h3>", unsafe_allow_html=True)

            # --- Simulate DCA once ---
            prices, net_worth_df, portfolio_df = run_dca_backtest(
                rets,
                weights_res["weights"],
                amount=10000,
                interval=interval_map[dca_interval]
            )

            # --- Calculate metrics on portfolio total ---
            metrics = calculate_performance_metrics(portfolio_df, freq=interval_map[freq])
            metric_keys = [
                ("Annual Return", f"{metrics['Annual Return']:.2%}"),
                ("Annual Volatility", f"{metrics['Annual Volatility']:.2%}"),
                ("Max Drawdown", f"{metrics['Max Drawdown']:.2%}"),
                ("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}"),
                ("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}"),
                ("Calmar Ratio", f"{metrics.get('Calmar Ratio', np.nan):.2f}")
            ]

            st.markdown("<p></p>", unsafe_allow_html=True)
            cols = st.columns(len(metric_keys))
            for idx, (key, val) in enumerate(metric_keys):
                with cols[idx]:
                    st.markdown(f"<div class='metric-card'>{key}: {val}</div>", unsafe_allow_html=True)

            st.markdown("<h3 class='sub-header'>Portfolio vs Benchmark</h3>", unsafe_allow_html=True)

            if benchmark_prices is not None and not benchmark_prices.empty:
                fig_bt = plot_portfolio_vs_benchmark(
                    net_worth_df,
                    portfolio_df["Portfolio"],
                    benchmark_prices=benchmark_prices,
                    amount=10000,
                    interval=interval_map[dca_interval]
                )
                st.plotly_chart(fig_bt, use_container_width=True)
            else:
                st.info("No benchmark data available to plot.")

        # --- Efficient Frontier ---
        with tabs[2]:
            st.markdown("<h2 class='sub-header'>Efficient Frontier</h2>", unsafe_allow_html=True)
            
            benchmark_returns = (
                price_to_returns(benchmark_prices, interval_map[freq])
                if benchmark_prices is not None and getattr(benchmark_prices, "empty", True) == False
                else None
            )
            
            ef_fig = plot_efficient_frontier(
                rets,
                weights_res["weights"],
                benchmark_returns=benchmark_returns
            )
            st.plotly_chart(ef_fig, use_container_width=True)
            
            with st.expander("ℹ️ About the Efficient Frontier"):
                st.markdown("The efficient frontier is a concept in modern portfolio theory representing the set of optimal portfolios that offer the highest expected return for a given level of risk (or the lowest risk for a given return). " \
                "Portfolios on this frontier are considered efficient, while any portfolio below it is suboptimal because it provides less return for the same risk.")
                st.markdown("Hover over points to see each portfolio's return, volatility, and weights. The red diamond shows VOO benchmark.")
        
    except Exception as e:
        st.error(f"❌ Error: {e}")

# --- Footer ---
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #6c757d; margin-top: 2rem;'>Created by Liaw Rui Xian • Not A Financial Advice</div>", unsafe_allow_html=True)
