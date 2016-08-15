import numpy as np
import pandas as pd
import neuralnet as nn
import matplotlib.pyplot as plt
from sklearn import datasets
from timeit import default_timer as timer

def generateData(seed = 0):
    np.random.seed(seed)
    X, y = datasets.make_moons(500, noise=0.20)
    return X,y

def plot_decision_boundary(pred_func,X,y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

def main():
    X, y = generateData()
    start = timer()
    test_x,test_y = generateData(1992)
    clf = nn.multilayer_perceptron(layers=[
        nn.layer("tanh",4),
        nn.layer("softmax",2)
    ],opt_function='momentum',drop_out=0.9)
    yt = pd.get_dummies(y,prefix='class').values
    tyt = pd.get_dummies(test_y,prefix='class').values
    clf.fit(test_x, tyt)

    timeusage = timer() - start
    print("%f seconds"%timeusage)

    y_ = clf.predict(X)
    print(y_)
    count = 0;
    for i in range(len(y)):
        if (y[i] == y_[i]): count += 1
    print(count / len(y))

    # Plot the decision boundary
    plot_decision_boundary(lambda x: clf.predict(x),X,y)
    plt.title("Multilayer Perceptron")
    plt.show()

main()