#!/usr/bin/python
# -*- coding: utf-8 -*-

# moving average smoothing as data preparation
from pandas import Series
from pandas import DataFrame
from matplotlib import pyplot
from pandas import concat
series = Series.from_csv( "data/daily-total-female-births-in-cal.csv", header=0)
df = DataFrame(series.values)
width = 3
lag1 = df.shift(1)
lag3 = df.shift(width - 1)
window = lag3.rolling(window=width)
means = window.mean()
dataframe = concat([means, lag1, df], axis=1)
dataframe.columns = ['mean','t','t+1']
print(dataframe.head(10))