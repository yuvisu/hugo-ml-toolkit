import numpy as np
class optimizer():
    def __init__(self, momentum=0.9,epsilon = 0.1):
        self.momentum = momentum
        self.epsilon = epsilon

    def gredientDescent(self,gredient,m):
        return -self.epsilon * gredient

    def momentum(self,gredient,m):
        if m is None:
            m=np.array([[0.0 for i in range(gredient.shape[1])] for j in range(gredient.shape[0])])
        update=self.momentum*m
        return -self.epsilon * gredient+update

    def function(self,method,neurons):
        for neuron in neurons:
            neuron.weights += self.methods[method](self,neuron.weights_derivative,neuron.weights_momentum)
            neuron.bias += self.methods[method](self,neuron.bias_derivative,neuron.bias_momentum)
        return neurons

    methods={
        'gredientDescent': gredientDescent,
        'momentum': momentum
    }