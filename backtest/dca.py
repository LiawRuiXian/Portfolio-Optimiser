import pandas as pd
import numpy as np

def get_scaled_amount(amount: float, interval: str) -> float:
    """
    Scale a base monthly amount according to the DCA interval.
    Args:
        amount (float): Base monthly investment amount.
        interval (str): 'D', 'W', or 'M' for daily, weekly, monthly.
    Returns:
        float: Scaled amount per period.
    """
    if interval == 'D':
        return amount / 21  # assume 21 trading days per month
    elif interval == 'W':
        return amount / 4   # assume 4 weeks per month
    else:
        return amount       # monthly

def simulate_dca(prices: pd.DataFrame, weights: np.ndarray, amount: float = 10000.0, interval: str = 'M'):
    """
    Simulate Dollar-Cost Averaging (DCA).
    Returns:
        net_worth_df: DataFrame of per-asset net worths
        portfolio_series: Series of total portfolio value
    """
    px = prices.copy().dropna()
    dates = px.index
    holdings = np.zeros(px.shape[1])
    net_worths = []
    last_period = None

    amount_per_period = get_scaled_amount(amount, interval)

    for dt in dates:
        current_period = dt.to_period(interval)
        if last_period is None or current_period != last_period:
            alloc = weights * amount_per_period
            shares = np.divide(
                alloc,
                px.loc[dt].values,
                out=np.zeros_like(alloc),
                where=px.loc[dt].values > 0
            )
            holdings += shares
            last_period = current_period

        net_worths.append(holdings * px.loc[dt].values)

    net_worth_df = pd.DataFrame(net_worths, index=dates, columns=prices.columns)
    portfolio_series = net_worth_df.sum(axis=1)
    return net_worth_df, portfolio_series

def run_dca_backtest(returns_df: pd.DataFrame, weights: np.ndarray, amount: float = 10000.0, interval: str = 'M'):
    """
    Run DCA backtest and return per-asset and total portfolio values.
    """
    prices = (1 + returns_df).cumprod()
    net_worth_df, portfolio_series = simulate_dca(prices, weights, amount, interval)
    portfolio_df = pd.DataFrame(portfolio_series, columns=["Portfolio"])
    return prices, net_worth_df, portfolio_df
