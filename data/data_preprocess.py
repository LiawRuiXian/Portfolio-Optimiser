def resample_prices(prices, freq):
    """按频率重采样"""
    if freq=="Daily":
        return prices
    elif freq=="Weekly":
        return prices.resample('W-FRI').last()
    elif freq=="Monthly":
        return prices.resample('M').last()
    return prices

def price_to_returns(prices, freq="Daily"):
    """价格转收益率"""
    px = resample_prices(prices, freq)
    rets = px.pct_change().dropna(how='all')
    return rets

def annualized_stats(rets, rf_annual=0.02, freq="Daily"):
    """计算年化收益、协方差、无风险利率"""
    k = {"Daily":252, "Weekly":52, "Monthly":12}.get(freq, 252)
    mu = rets.mean() * k
    Sigma = rets.cov() * k
    rf = rf_annual
    return mu.values, Sigma.values, rf
