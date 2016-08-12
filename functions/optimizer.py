
class optimizer:
    def gredientDescent(epsilon,gredient):
        return -epsilon * gredient

    optimizer={
        'gredientDescent': gredientDescent
    }