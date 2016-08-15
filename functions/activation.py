import numpy as np
#old_settings = np.seterr(all='ignore')

def softmax(values):
    exp_scores = np.exp(values)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs

def tanh(values):
    return np.tanh(values)

def tanhGrad(values):
    return 1 - np.power(values, 2)

def logistic(values):
    exp_scores = np.exp(-values)
    return 1./(1+exp_scores)

def logisticGrad(values):
    return values - np.power(values, 2)

def relu(values):
    return values * (values > 0)

def reluGrad(values):
    return (values > 0)

activation = {
    "logistic": logistic,
    "softmax": softmax,
    "tanh": tanh,
    "relu": relu,
}

activation_Grad = {
    "logistic": logisticGrad,
    "softmax": softmax,
    "tanh": tanhGrad,
    "relu": reluGrad,
}

