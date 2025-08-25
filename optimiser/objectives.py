import numpy as np
from backtest.performance import max_drawdown, downside_vol, cvar_loss

def portfolio_perf(w, mu, Sigma, rf):
    """计算投资组合收益、波动、夏普"""
    ret = float(w @ mu)
    vol = float(np.sqrt(max(1e-12, w @ Sigma @ w)))
    sharpe = (ret - rf)/vol if vol>0 else -np.inf
    return ret, vol, sharpe

def portfolio_mdd(w, rets):
    port = rets @ w
    return -max_drawdown(port)

def neg_sortino(w, rets, rf_annual, freq):
    k = {"Daily":252,"Weekly":52,"Monthly":12}.get(freq,252)
    rf_p = (1+rf_annual)**(1/k)-1
    port = rets @ w
    ann_ret = port.mean()*k
    dsv = downside_vol(port, rf_p, k)
    if dsv<=0:
        return 1e6
    return - (ann_ret - rf_annual)/dsv

def cvar_obj(w, rets, alpha=0.95):
    port = rets @ w
    return cvar_loss(port, alpha)
