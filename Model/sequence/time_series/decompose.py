#!/usr/bin/python
# -*- coding: utf-8 -*-

from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
series = Series.from_csv('data/airline-passengers.csv', header=0)
result = seasonal_decompose(series, model='additive')
result.plot()
pyplot.show()