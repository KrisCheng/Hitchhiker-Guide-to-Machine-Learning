#!/usr/bin/python
# -*- coding: utf-8 -*-

# how to transform with pandas

from pandas import Series
from matplotlib import pyplot

series = Series.from_csv('airline-passengers.csv', header = 0)
pyplot.figure(1)

pyplot.subplot(211)
pyplot.plot(series)

pyplot.subplot(212)
pyplot.hist(series)

pyplot.show()