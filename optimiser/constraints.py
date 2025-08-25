import numpy as np
def constraint_sum_to_one(n):
    return {'type':'eq','fun': lambda w: w.sum()-1.0}

def constraint_target_return(mu, target):
    return {'type':'eq','fun': lambda w: float(w @ mu)-target}

def bounds_factory(n, allow_short=False, max_weight=1.0, min_short=0.0):
    lb = -abs(min_short) if allow_short else 0.0
    ub = max(1.0, max_weight)
    return tuple((lb, ub) for _ in range(n))

def initial_weights(n):
    return np.ones(n)/n
