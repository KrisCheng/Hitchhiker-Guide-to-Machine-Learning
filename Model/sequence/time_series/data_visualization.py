#!/usr/bin/python
# -*- coding: utf-8 -*-

from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from pandas import concat
from matplotlib import pyplot
from pandas.tools.plotting import lag_plot
from pandas.tools.plotting import autocorrelation_plot
series = Series.from_csv('daily-minimum-temperatures.csv', header = 0)

# box and whisker plot

# Monthly
# one_year = series['1990']
# groups = one_year.groupby(TimeGrouper('M'))
# months = concat([DataFrame(x[1].values) for x in groups], axis = 1)
# months = DataFrame(months)
# months.columns = range(1, 13)
# months.boxplot()

# Yearly
# groups = series.groupby(TimeGrouper('A'))
# years = DataFrame()
# for name, group in groups:
#     years[name.year] = group.values
# years.boxplot()

# heatmap plot

# Yearly
# groups = series.groupby(TimeGrouper('A'))
# years = DataFrame()
# for name, group in groups:
#     years[name.year] = group.values
# years = years.T
# pyplot.matshow(years, interpolation = None, aspect = 'auto')

# Monthly
# one_year = series['1990']
# groups = one_year.groupby(TimeGrouper('M'))
# months = concat([DataFrame(x[1].values) for x in groups], axis = 1)
# months = DataFrame(months)
# months.columns = range(1, 13)
# pyplot.matshow(months, interpolation = None, aspect = 'auto')

# lag_plot(series)

# values = DataFrame(series.values)
# lags = 7
# columns = [values]
# for i in range(1, (lags + 1)):
#     columns.append(values.shift(i))
# dataframe = concat(columns, axis = 1)
# columns = ['t + 1']
# for i in range(1, (lags + 1)):
#     columns.append('t - ' + str(i))
# dataframe.columns = columns
# pyplot.figure(1)
# for i in range(1, (lags + 1)):
#     ax = pyplot.subplot(240 + i)
#     ax.set_title('t + 1 vs t - ' + str(i))
#     pyplot.scatter(x = dataframe['t + 1'].values, y = dataframe['t - '+ str(i)].values)

autocorrelation_plot(series)

pyplot.show()
