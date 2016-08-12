class optimizer():
    def __init__(self, momentum=0.1, decay=0.1):
        self.momentum = momentum
        self.decay = 0.1

    def gredientDescent(epsilon,gredient):
        return -epsilon * gredient

    function={
        'gredientDescent': gredientDescent
    }