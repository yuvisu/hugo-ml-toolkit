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

def sigmoid(values):
    exp_scores = np.exp(-values)
    result = 1./ 1+exp_scores
    return result

def relu(values):
    return values * (values > 0)

def reluGrad(values):
    return (values > 0)

activation = {
    "sigmoid": sigmoid,
    "softmax": softmax,
    "tanh": tanh,
    "relu": relu,
}

activation_Grad = {
    "sigmoid": sigmoid,
    "softmax": softmax,
    "tanh": tanhGrad,
    "relu": reluGrad,
}

