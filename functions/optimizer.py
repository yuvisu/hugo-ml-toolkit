import numpy as np
class optimizer():
    def __init__(self, momentum=0.9, decay=0.1):
        self.momentum = momentum
        self.decay = 0.1

    def gredientDescent(self,epsilon,gredient,m):
        return -epsilon * gredient

    def momentum(self,epsilon,gredient,m):
        if m==None:
            m=np.array([[0.0 for i in range(gredient.shape[1])] for j in range(gredient.shape[0])])
        update=self.momentum*m
        return -epsilon * gredient+update

    def function(self,method,epsilon,gredient,m):
        return self.methods[method](self,epsilon,gredient,m)

    methods={
        'gredientDescent': gredientDescent,
        'momentum': momentum
    }