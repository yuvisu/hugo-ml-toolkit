import numpy as np

class multilayer_perceptron():

    def __init__(self):
        self.x = []
        self.y = []
        self.nn_input_dim = 0;
        self.nn_onput_dim = 0;
        self.num_examples = 0
        self.epsilon = 0.01
        self.reg_lambda = 0.01
        self.model = {}

    def calculate_loss(self,model):
        W1,b1,W2,b2 = model['W1'],model['b1'],model['W2'],model['b2']
        z1 = self.x.dot(W1)+b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2)+b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores,axis=1,keepdims=True)

        #Calculating the loss
        corect_logprobs = -np.log(probs[range(self.num_examples),self.y])
        data_loss = np.sum(corect_logprobs)

        #Add regulatization term to loss
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1. / self.num_examples * data_loss

    def predict(self,xx):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'],self.model['b2']
        # Forward propagation
        z1 = xx.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def fit(self,X,Y,nn_h_dim = 3,num_passes=10000,reg_lambda=0.01,epsilon=0.01,print_loss=False):

        self.x = X
        self.y = Y
        self.nn_input_dim = len(X[0])
        self.nn_output_dim = len(set(Y))
        self.reg_lambda = reg_lambda
        self.epsilon = epsilon
        self.num_examples = len(X)

        np.random.seed(0)
        W1 = np.random.randn(self.nn_input_dim, nn_h_dim) / np.sqrt(self.nn_input_dim)
        b1 = np.zeros((1, nn_h_dim))
        W2 = np.random.randn(nn_h_dim, self.nn_output_dim) / np.sqrt(nn_h_dim)
        b2 = np.zeros((1, self.nn_output_dim))

        for i in range(0,num_passes):
            # Forward propagation
            z1 = X.dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            delta3 = probs
            delta3[range(len(X)), Y] -= 1
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms
            dW2 += self.reg_lambda * W2
            dW1 += self.reg_lambda * W1

            # Gradient descent parameter update
            W1 += -self.epsilon * dW1
            b1 += -self.epsilon * db1
            W2 += -self.epsilon * dW2
            b2 += -self.epsilon * db2

            self.model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

            if print_loss and i % 1000 == 0:
                print("Loss after iteration ",i, self.calculate_loss(self.model))

        return self.model
