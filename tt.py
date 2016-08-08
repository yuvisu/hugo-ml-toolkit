#import numbapro

#numbapro.check_cuda()

from sympy import *
import sympy as sp

pred = Symbol("y")
target = Symbol("y_")
ex = (1 / 2 * (pred - target) ** 2)
derive_c = ex.diff(pred)

expression = "(0.5*(y - y_) ** 2)"

e = S(expression)

a = sympify(expression)

dd = a.diff(pred)

print (ex)
print (dd)

