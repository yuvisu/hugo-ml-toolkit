import numpy as np

def l1(X):
    return np.abs(X).sum()

def l2(X):
    return np.sqrt(X.T.dot(X))

penalty = {
    "l1": l1,
    "l2": l2,
}