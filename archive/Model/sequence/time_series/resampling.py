#!/usr/bin/python
# -*- coding: utf-8 -*-

# how to interpolate with pandas

from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot

def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')

series = read_csv('shampoo-sales.csv', header = 0, parse_dates = [0], index_col = 0,
                  squeeze = True, date_parser = parser)
resample = series.resample('A')
yearly_mean_sales = resample.sum()
# interpolated = upsampled.interpolate(method = 'spline', order = 1)
# print(len(interpolated))
print(yearly_mean_sales)
yearly_mean_sales.plot()
pyplot.show()