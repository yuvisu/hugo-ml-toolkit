class optimizer():
    def __init__(self, momentum=0.1, decay=0.1):
        self.momentum = momentum
        self.decay = 0.1

    def gredientDescent(self,epsilon,gredient):
        return -epsilon * gredient

    def function(self,method,epsilon,gredient):
        return self.methods[method](self,epsilon,gredient)

    methods={
        'gredientDescent': gredientDescent
    }