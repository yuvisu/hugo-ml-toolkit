class neuron():
    def __init__(self):
        self.weights = []
        self.weights_momentum = None
        self.weights_derivative = []
        self.bias = []
        self.bias_momentum = None
        self.bias_derivative = []
        self.a = []
        self.z = []
        self.delta = []

    def printInfo(self):
        print("weight:",self.weights)
        print("weights_momentum:",self.weights_momentum)
        print("weights_derivative:",self.weights_derivative)
        print("bias:",self.bias)
        print("bias_momentum:",self.bias_momentum)
        print("bias_derivative:",self.bias_derivative)
        print("a:",self.a)
        print("z:",self.z)
        print("delta:",self.delta)