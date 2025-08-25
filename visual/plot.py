import numpy as np
import pandas as pd
import plotly.graph_objects as go
from backtest import simulate_dca,get_scaled_amount

def plot_portfolio_vs_benchmark(net_worth_df, portfolio_series, benchmark_prices=None, amount=10000.0, interval='M'):
    """
    Plot DCA portfolio vs benchmark using precomputed results.
    """
    portfolio_norm = portfolio_series / portfolio_series.iloc[0] * amount

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=portfolio_norm.index,
        y=portfolio_norm.values,
        mode='lines',
        name='Portfolio',
        line=dict(color='#FFD700', width=3),
        hovertemplate="Date: %{x}<br>Portfolio: $%{y:,.2f}<extra></extra>"
    ))

    # Individual assets
    for col in net_worth_df.columns:
        asset_norm = net_worth_df[col] / net_worth_df[col].iloc[0] * amount
        fig.add_trace(go.Scatter(
            x=net_worth_df.index,
            y=asset_norm,
            mode='lines',
            name=col,
            line=dict(width=1),
            opacity=0.5,
            showlegend=False,
            hovertemplate=f"Date: %{{x}}<br>{col}: $%{{y:,.2f}}<extra></extra>"
        ))

    # Benchmark (normalized to same starting value)
    if benchmark_prices is not None and not benchmark_prices.empty:
        bench_amount = get_scaled_amount(amount, interval)
        benchmark_net, benchmark_series = simulate_dca(benchmark_prices, np.array([1.0]), bench_amount, interval)
        benchmark_norm = benchmark_series / benchmark_series.iloc[0] * amount

        fig.add_trace(go.Scatter(
            x=benchmark_norm.index,
            y=benchmark_norm.values,
            mode='lines',
            name=benchmark_prices.columns[0],
            line=dict(color='#FF4B4B', width=3, dash='dash'),
            hovertemplate="Date: %{x}<br>Benchmark: $%{y:,.2f}<extra></extra>"
        ))

    fig.update_layout(
        title="DCA Portfolio vs Benchmark",
        xaxis_title="Date",
        yaxis_title="Net Worth ($)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_efficient_frontier(returns: pd.DataFrame, optimal_weights, benchmark_returns: pd.Series = None, points=100):
    np.random.seed(42)

    returns = returns.apply(pd.to_numeric, errors='coerce')
    returns.index = pd.to_datetime(returns.index)

    n_assets = len(returns.columns)
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # Generate random portfolios
    results = []
    for _ in range(points):
        w = np.random.random(n_assets)
        w /= np.sum(w)
        port_return = np.dot(w, mean_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        results.append({"Return": port_return, "Volatility": port_vol, "Weights": w.copy()})

    results_df = pd.DataFrame(results)

    # Optimal portfolio
    if isinstance(optimal_weights, dict):
        opt_weights_arr = np.array([float(optimal_weights.get(t, 0)) for t in returns.columns])
    else:
        opt_weights_arr = np.array([float(x) for x in optimal_weights], dtype=float)

    opt_return = float(np.dot(opt_weights_arr, mean_returns))
    opt_vol = float(np.sqrt(np.dot(opt_weights_arr.T, np.dot(cov_matrix, opt_weights_arr))))

    # Benchmark 
    benchmark_point = None
    if benchmark_returns is not None:
        benchmark_mean = float(benchmark_returns.mean() * 252)
        benchmark_vol = float(benchmark_returns.std() * np.sqrt(252))
        benchmark_point = (benchmark_vol, benchmark_mean)

    fig = go.Figure()

    # Random portfolios
    hover_texts = []
    for r, v, w in zip(results_df['Return'], results_df['Volatility'], results_df['Weights']):
        r, v = float(r), float(v)
        w_formatted = ', '.join([f"{float(x):.2f}" for x in w])
        hover_texts.append(f"Return: {r:.2%}<br>Volatility: {v:.2%}<br>Weights: {w_formatted}")

    fig.add_trace(go.Scatter(
        x=results_df['Volatility'].astype(float),
        y=results_df['Return'].astype(float),
        mode='markers',
        marker=dict(color='rgba(255,127,80,0.6)', size=8),
        name='Random Portfolios',
        hovertext=hover_texts,
        hoverinfo="text"
    ))

    # Optimal portfolio
    opt_hover = f"Return: {opt_return:.2%}<br>Volatility: {opt_vol:.2%}<br>Weights: {', '.join([f'{x:.2f}' for x in opt_weights_arr])}"
    fig.add_trace(go.Scatter(
        x=[opt_vol],
        y=[opt_return],
        mode='markers',
        marker=dict(color='#00FF7F', size=14, symbol='star'),
        name='Optimal Portfolio',
        hovertext=[opt_hover],
        hoverinfo="text"
    ))

    # Benchmark
    if benchmark_point is not None:
        bench_vol, bench_ret = benchmark_point
        fig.add_trace(go.Scatter(
            x=[bench_vol],
            y=[bench_ret],
            mode='markers',
            marker=dict(color='red', size=14, symbol='diamond'),
            name='Benchmark (VOO)',
            hovertext=[f"Return: {bench_ret:.2%}<br>Volatility: {bench_vol:.2%}"],
            hoverinfo="text"
        ))

    fig.update_layout(
        title="Efficient Frontier with Benchmark",
        xaxis_title="Volatility",
        yaxis_title="Expected Return",
        plot_bgcolor='#0F111A',
        paper_bgcolor='#0F111A',
        font_color='#E0E0E0',
        hovermode="closest"
    )

    return fig