import numpy as np
import functions.activation as af

class logisticregression():
    def __init__(self,tolerance=0.0001,random_state = 0,n_iter = 100,epsilon=0.01):
        self.tolerance = tolerance
        self.random_state = random_state
        self.n_iter = n_iter
        self.epsilon = epsilon

    def predict(self, X):
        features = (X - self.mean_x) / self.std_x
        probs = af.logistic(features.dot(self.w))

        for p in probs:
            if p[0] >= 0.5: p[0] = 1
            else: p[0] = 0
        return probs

    def predict_proba(self, X):
        features = (X - self.mean_x) / self.std_x
        probs = af.logistic(features.dot(self.w))
        return probs

    def fit(self, X, Y):
        np.random.seed(self.random_state)
        self.mean_x = X.mean(axis=0)
        self.std_x = X.std(axis=0)
        self.features = (X - self.mean_x) / self.std_x
        self.labels = Y.reshape(Y.size, 1)
        self.w = np.zeros((X.shape[1] , 1))

        previous_likelihood = self.log_likelihood()
        difference = self.tolerance + 1

        for i in range(0, self.n_iter):
            if difference > self.tolerance :
                self.w += self.epsilon * self.log_likelihood_gradient()
                temp = self.log_likelihood()
                difference = np.abs(temp - previous_likelihood)
                previous_likelihood = temp

        self.coef_ = self.w


    def log_likelihood(self):
        # Get Probablities
        p = af.logistic(self.features.dot(self.w))
        # Get Log Likelihood For Each Row of Dataset
        loglikelihood = self.labels * np.log(p + 1e-24) + (1 - self.labels) * np.log(1 - p + 1e-24)
        # Return Sum
        return -1 * loglikelihood.sum()

    def log_likelihood_gradient(self):
        error = self.labels - af.logistic(self.features.dot(self.w))
        product = error * self.features
        return product.sum(axis=0).reshape(self.w.shape)
