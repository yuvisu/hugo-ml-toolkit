import numpy as np
class optimizer():
    def __init__(self, momentum=0.9):
        self.momentum = momentum

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