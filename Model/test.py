#!~/anaconda/bin/python
# -*- coding: UTF-8 -*-
from functools import reduce
__author__ = "Kris Peng"

# for some test code. based on python3


# map reduce
def f(x):
    return x * x
r = map(f, [1, 2, 3, 4, 5])
print(list(r))

def add(x, y):
    return x + y
print(reduce(add, [1, 2, 3, 4, 5]))

# lambda
g = lambda x: x+1
print(g(2))
