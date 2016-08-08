from sympy import *

def dist(y,y_):
    return y-y_

def derived_func(derive_c,y,y_):
    pred = Symbol("y")
    target = Symbol("y_")
    delta = y
    for j in range(len(y)):
        for k in range(len(delta[j])):
            delta[j][k] = derive_c.subs({pred: y[j][k], target: y_[j][k]})
    return delta


cost = {
    "dist": dist,
}

