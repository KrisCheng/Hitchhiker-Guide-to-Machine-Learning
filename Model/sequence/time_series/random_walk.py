#!/usr/bin/python
# -*- coding: utf-8 -*-

from random import seed
from random import randrange
from matplotlib import pyplot
seed(1)
series = [randrange(10) for i in range(1000)]
print(series)
# pyplot.plot(series)
# pyplot.show()