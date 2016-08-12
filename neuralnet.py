import numpy as np
import scipy as sp
import functions.activation as af
import functions.optimizer as opt
import functions.cost as cf
from copy import deepcopy
from sympy import *
from timeit import default_timer as timer

class multilayer_perceptron():

    def __init__(self,layers,n_iter=10000,reg_lambda=0.01,epsilon=0.01,random_state = 0,cost = 'dist',expr = None,
                 opt_function = 'gredientDescent',momentum = 0.1,decay = 0.1,print_loss=True):
        self.x = []
        self.y = []
        self.model = {}
        self.num_examples = 0
        self.nn_input_dim = 0
        self.layers = layers # neural net layers list
        self.num_layers = len(layers) # number of neural net layers
        self.epsilon = epsilon # learning rate
        self.reg_lambda = reg_lambda # regularization
        self.n_iter = n_iter # n iteration
        self.print_loss = print_loss # print loss
        self.random_state = random_state # random seed
        self.cost = cost #
        self.expr = expr
        self.optimizer = opt.optimizer(momentum= momentum,decay=decay)
        self.opt_function = opt_function
        self.momentum = momentum
        self.decay = decay
        self.act_type = [] # list of activation type
        self.output_dim = [] # list of ouput dimensional per layer
        for layer in layers:
            self.act_type.append(layer.get_activation())
            self.output_dim.append(layer.get_n_units())

        if expr is not None:
            self.pred = Symbol("y")
            self.target = Symbol("y_")
            self.derive_c = (expr).diff(self.pred)

    def logloss(self):
        probs = self.predict_proba(self.x)
        # Calculating the loss
        epsilon = 1e-15
        probs = sp.maximum(epsilon, probs)
        probs = sp.minimum(1 - epsilon, probs)
        ll = sum(self.y * sp.log(probs) + sp.subtract(1, self.y) * sp.log(sp.subtract(1, probs)))
        ll = ll * -1.0 / len(self.y)

        return ll[0]

    def predict(self,xx):
        weights = self.model["weight"]
        bias = self.model["bias"]
        a = [None] * self.num_layers
        z = [None] * self.num_layers
        a[0] = xx
        probs = 0;
        for idx in range(self.num_layers):
            z[idx] = a[idx].dot(weights[idx]) + bias[idx]
            if idx + 1 < self.num_layers:
                a[idx + 1] = af.activation[self.act_type[idx]](z[idx])
            else:
                probs = af.activation[self.act_type[idx]](z[idx])
        return np.argmax(probs, axis=1)

    def predict_proba(self,xx):
        weights = self.model["weight"]
        bias = self.model["bias"]
        a = [None] * self.num_layers
        z = [None] * self.num_layers
        a[0] = xx
        probs = 0;
        for idx in range(self.num_layers):
            z[idx] = a[idx].dot(weights[idx]) + bias[idx]
            if idx + 1 < self.num_layers:
                a[idx + 1] = af.activation[self.act_type[idx]](z[idx])
            else:
                probs = af.activation[self.act_type[idx]](z[idx])
        return probs

    def fit(self,X,Y):
        self.x = X
        self.y = deepcopy(Y)
        self.nn_input_dim = len(X[0])
        self.num_examples = len(X)

        weights = [None]*self.num_layers
        derive_weights = [None]*self.num_layers
        bias = [None]*self.num_layers
        derive_bias = [None]*self.num_layers
        a = [None]*self.num_layers
        z = [None]*self.num_layers

        delta = [None] * self.num_layers

        a[0] = X # set training data

        np.random.seed(self.random_state)

        for idx in range(self.num_layers):
            if idx is 0:
                weights[idx] = (np.random.randn(self.nn_input_dim, self.output_dim[idx]) / np.sqrt(self.nn_input_dim))
            else:
                weights[idx] = (np.random.randn(self.output_dim[idx - 1], self.output_dim[idx]) / np.sqrt(self.output_dim[idx - 1]))
            bias[idx] = np.zeros((1, self.output_dim[idx]))

        for i in range(0,self.n_iter):
            probs = None
            # Forward propagation
            for idx in range(self.num_layers):
                z[idx] = a[idx].dot(weights[idx]) + bias[idx]
                if idx+1 < self.num_layers: a[idx+1] = af.activation[self.act_type[idx]](z[idx])
                else: probs = af.activation[self.act_type[idx]](z[idx])

            # Backpropagation
            for idx in reversed(range(self.num_layers)):
                if idx + 1 < self.num_layers:
                    delta[idx] = delta[idx+1].dot(weights[idx+1].T) * af.activation_Grad[self.act_type[idx]](a[idx+1])
                else:
                    if self.expr is None: delta[idx] = cf.cost[self.cost](probs,self.y)
                    else: delta[idx] = cf.derived_func(self.derive_c,probs,self.y)
                derive_weights[idx] = (a[idx].T).dot(delta[idx])
                derive_bias[idx] = np.sum(delta[idx], axis=0, keepdims=True)

            # Add regularization terms
            for idx in range(self.num_layers):
                derive_weights[idx] += self.reg_lambda * weights[idx]

            # Gradient descent parameter update
            for idx in range(self.num_layers):
                weights[idx] += self.optimizer.function[self.opt_function](self.epsilon,derive_weights[idx])
                bias[idx] += self.optimizer.function[self.opt_function](self.epsilon,derive_bias[idx])

            self.model = {'weight': weights, 'bias': bias}

            if self.print_loss and i % 100 == 0:
                print("Loss after iteration ",i, self.logloss())

        return self.model

class layer():
    def __init__(self, activation="tanh", n_units=3):
        self.activation = activation
        self.n_units = n_units

    def get_n_units(self):
        return self.n_units

    def get_activation(self):
        return self.activation