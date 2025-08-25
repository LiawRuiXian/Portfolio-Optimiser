from scipy.optimize import minimize
import numpy as np
from .objectives import portfolio_perf, portfolio_mdd, neg_sortino, cvar_obj
from .constraints import constraint_sum_to_one, constraint_target_return, bounds_factory, initial_weights

def optimize_portfolio(returns_df, objective=None, target_return=None,
                       allow_short=False, max_weight=1.0, min_short=0.0,
                       rf=0.02, freq="Daily", cvar_alpha=0.95):
    mu = returns_df.mean() * {"Daily":252,"Weekly":52,"Monthly":12}[freq]
    Sigma = returns_df.cov() * {"Daily":252,"Weekly":52,"Monthly":12}[freq]
    n = len(mu)
    cons = [constraint_sum_to_one(n)]
    if objective=="Minimize Volatility for Target Return" and target_return:
        cons.append(constraint_target_return(mu.values, target_return))
    bnds = bounds_factory(n, allow_short, max_weight, min_short)
    w0 = initial_weights(n)

    if objective=="Maximize Sharpe Ratio":
        res = minimize(lambda w: -portfolio_perf(w, mu.values, Sigma.values, rf)[2],
                       w0,bounds=bnds,constraints=cons,method='SLSQP')
    elif objective in ["Minimize Volatility","Minimize Volatility for Target Return"]:
        res = minimize(lambda w: np.sqrt(max(1e-12, w@Sigma.values@w)),
                       w0,bounds=bnds,constraints=cons,method='SLSQP')
    elif objective=="Minimize Maximum Drawdown":
        res = minimize(lambda w: portfolio_mdd(w, returns_df), w0,bounds=bnds,constraints=cons,method='SLSQP')
    elif objective=="Maximize Sortino Ratio":
        res = minimize(lambda w: neg_sortino(w, returns_df, rf, freq), w0,bounds=bnds,constraints=cons,method='SLSQP')
    elif objective=="Minimize CVaR":
        res = minimize(lambda w: cvar_obj(w, returns_df, cvar_alpha), w0,bounds=bnds,constraints=cons,method='SLSQP')
    else:
        raise ValueError("Unknown Objective")
    w = res.x / res.x.sum() if res.success else w0
    return {"weights": w, "res": res}
