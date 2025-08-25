import numpy as np
import pandas as pd

def downside_vol(rets, thresh=0.0, k=252):
    d = np.minimum(0.0, rets-thresh)
    return float(np.sqrt(np.mean(d**2)*k))

def cvar_loss(rets, alpha=0.95):
    losses = -rets.values
    q = np.quantile(losses, alpha)
    tail = losses[losses>=q]
    if tail.size==0: return 0.0
    return float(tail.mean())

def max_drawdown(series):
    """Compute the maximum drawdown of a series."""
    cumulative_max = series.cummax()
    drawdowns = (series - cumulative_max) / cumulative_max
    return abs(drawdowns.min())

def calculate_performance_metrics(df: pd.DataFrame, freq: str = "D"):
    """
    Calculate key performance metrics for a portfolio.
    
    Parameters:
        df (pd.DataFrame): Must contain 'Portfolio' column with portfolio values or returns.
        freq (str): Sampling frequency ("D", "W", "M"). Default daily.
        
    Returns:
        dict: Dictionary with performance metrics.
    """
    df = df.copy()

    if "Portfolio" not in df.columns:
        raise ValueError("DataFrame must contain a 'Portfolio' column")

    # ---- 1. Convert to returns if needed ----
    # crude check: if values are prices (>2), convert to pct_change
    if (df["Portfolio"].abs() > 2).any():
        returns = df["Portfolio"].pct_change().dropna()
    else:
        returns = df["Portfolio"].dropna()

    # ---- 2. Annualization factor ----
    freq_map = {"D": 252, "W": 52, "M": 12}
    ann_factor = freq_map.get(freq.upper(), 252)

    # ---- 3. Cumulative and annualized returns ----
    cumulative_return = (1 + returns).prod() - 1
    annual_return = (1 + cumulative_return) ** (ann_factor / len(returns)) - 1

    # ---- 4. Annualized volatility ----
    annual_volatility = returns.std(ddof=1) * np.sqrt(ann_factor)

    # ---- 5. Sharpe Ratio ----
    # Use risk-free rate = 0
    sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else np.nan

    # ---- 6. Sortino Ratio ----
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std(ddof=1) * np.sqrt(ann_factor)
    sortino_ratio = annual_return / downside_volatility if downside_volatility != 0 else np.nan

    # ---- 7. Max Drawdown ----
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = cumulative / peak - 1
    max_drawdown = drawdown.min()

    # ---- 8. Calmar Ratio ----
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    # ---- 9. Return metrics ----
    metrics = {
        "Annual Return": float(annual_return),
        "Annual Volatility": float(annual_volatility),
        "Sharpe Ratio": float(sharpe_ratio),
        "Sortino Ratio": float(sortino_ratio),
        "Max Drawdown": float(max_drawdown),
        "Calmar Ratio": float(calmar_ratio)
    }

    return metrics