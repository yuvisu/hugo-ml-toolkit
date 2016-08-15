import numpy as np
from sympy import *
import sympy as sp

x = Symbol("x")
ex = (1. / (1 + np.e ** -x))
derive_c = ex.diff(x)

expression = "(0.5*(y - y_) ** 2)"

e = S(expression)

a = sympify(expression)

print (derive_c)

