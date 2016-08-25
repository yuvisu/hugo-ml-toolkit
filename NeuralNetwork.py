import random
import numpy as np
import scipy as sp
import functions.activation as af
import functions.optimizer as opt
import functions.cost as cf
import compoent.neuron as nr
from sympy import *

class multilayerperceptron():

    def __init__(self,layers,n_iter=1000,num_batch = 50,reg_lambda=0.01,epsilon=0.01,random_state = 0,cost = 'dist',expr = None,
                 opt_function = 'gredientDescent',momentum = 0.9,drop_out = 1,print_loss=True):
        self.x = []
        self.y = []
        self.model = {}
        self.num_examples = 0
        self.nn_input_dim = 0
        self.layers = layers # neural net layers list
        self.num_layers = len(layers) # number of neural net layers
        self.num_batch = num_batch # number of batch
        self.epsilon = epsilon # learning rate
        self.reg_lambda = reg_lambda # regularization
        self.n_iter = n_iter # n iteration
        self.print_loss = print_loss # print loss
        self.random_state = random_state # random seed
        self.cost = cost # cost function
        self.expr = expr # expression of cost function
        self.optimizer = opt.optimizer(momentum= momentum,epsilon = epsilon) # optimizer
        self.opt_function = opt_function #optimization function
        self.momentum = momentum # momentum
        self.drop_out = drop_out
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
                probs = af.activation[self.act_type[idx]](z[idx]) * self.drop_out
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
                probs = af.activation[self.act_type[idx]](z[idx]) * self.drop_out
        return probs

    def fit(self,X,Y):
        np.random.seed(self.random_state)
        self.x = X
        self.y = Y
        self.nn_input_dim = len(self.x[0])
        self.num_examples = len(self.x)

        neurons = []
        drop = [None]*self.num_examples
        weights = [None]* self.num_layers
        bias = [None]* self.num_layers

        for i in range(self.num_layers):
            tmp = nr.neuron()
            neurons.append(tmp)

        for idx in range(self.num_layers): # initialization
            if idx is 0:
                neurons[idx].weights = (np.random.randn(self.nn_input_dim, self.output_dim[idx]) / np.sqrt(self.nn_input_dim))
            else:
                neurons[idx].weights = (np.random.randn(self.output_dim[idx - 1], self.output_dim[idx]) / np.sqrt(self.output_dim[idx - 1]))
            neurons[idx].bias = np.zeros((1, self.output_dim[idx]))

        for i in range(0,self.n_iter):
            rn = random.randint(0, self.num_examples-self.num_batch) # get a random integer
            train_x = X[rn:rn+self.num_batch] # select a batch from all data
            train_y = self.y[rn:rn+self.num_batch]

            neurons[0].a = train_x # set training data
            probs = None # forward result

            # Forward propagation
            for idx in range(self.num_layers):
                neurons[idx].z = neurons[idx].a.dot(neurons[idx].weights)+neurons[idx].bias # z = ax + b
                if idx+1 < self.num_layers: # if not the last layer
                    neurons[idx+1].a = af.activation[self.act_type[idx]](neurons[idx].z) # using activation function
                    drop[idx] = np.random.binomial(1,self.drop_out,size=neurons[idx].z.shape) # dropout
                    neurons[idx+1].a *= drop[idx] # the next layer input product dropout
                else:
                    probs = af.activation[self.act_type[idx]](neurons[idx].z) # the last layer's activation function

            # Backpropagation
            for idx in reversed(range(self.num_layers)):
                if idx + 1 < self.num_layers: # if not the last layer
                    neurons[idx].delta = neurons[idx+1].delta.dot(neurons[idx+1].weights.T) * af.activation_Grad[self.act_type[idx]](neurons[idx+1].a) # calculate the delta
                else:
                    if self.expr is None:
                        neurons[idx].delta = cf.cost[self.cost](probs,train_y)
                    else: # customize cost function
                        neurons[idx].delta = cf.derived_func(self.derive_c,probs,train_y)
                neurons[idx].weights_derivative = (neurons[idx].a.T).dot(neurons[idx].delta)
                neurons[idx].bias_derivative = np.sum(neurons[idx].delta,axis = 0,keepdims = True)

            # Add regularization terms
            for idx in range(self.num_layers):
                neurons[idx].weights_derivative += self.reg_lambda * neurons[idx].weights

            # Gradient descent parameter update
            neurons = self.optimizer.function(self.opt_function,neurons)

            # Get All weights and bias
            for idx in range(self.num_layers):
                weights[idx] = neurons[idx].weights
                bias[idx] = neurons[idx].bias

            self.model = {'weight': weights, 'bias': bias}

            if self.print_loss and i % 100 == 0:
                print("Loss after iteration ",i, self.logloss()," batch ",rn)

        return self.model

class layer():
    def __init__(self, activation="tanh", n_units=3):
        self.activation = activation
        self.n_units = n_units

    def get_n_units(self):
        return self.n_units

    def get_activation(self):
        return self.activation