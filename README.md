###  implement neural network(BP)

- 为了练习写的library；
- 简单易用；
- 目前没有gpu加速；
- 以后再慢慢实作其它内容
# 用法

**code**

> clf = nn.multilayer_perceptron(layers=[
        nn.layer("tanh",4),
        nn.layer("softmax",2)
    ])
    yt = pd.get_dummies(y,prefix='class').values
    clf.fit(X, yt)