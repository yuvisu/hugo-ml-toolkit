#  Introduction

 - implement machine learning algorithms；
 - aim to training personal skill and create a 'ease to use' library
 - current version not support GPU ；
 - current algorithms:
    -- neural network(BP)
    -- logistic regression


    
 
#  Guide

**code**

```python
clf = nn.multilayer_perceptron(layers=[
        nn.layer("tanh",4),
        nn.layer("softmax",2)
    ])
yt = pd.get_dummies(y,prefix='class').values
clf.fit(X, yt)
```