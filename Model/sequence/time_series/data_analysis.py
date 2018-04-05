#!/usr/bin/python
# -*- coding: utf-8 -*-


from pandas import Series
from pandas import DataFrame
from pandas import concat

# series = read_csv('daily-total-female-births-in-cal.csv', 
#                   header = 0, 
#                   parse_dates = [0], 
#                   index_col = 0, 
#                   squeeze = True)
# print(type(series))
# print(series.head())

# series = Series.from_csv('daily-total-female-births-in-cal.csv',  header = 0)
# print(series.describe())

series = Series.from_csv('daily-minimum-temperatures.csv',  header = 0)
dataFrame = DataFrame()
# dataFrame['month'] =  [series.index[i].month for i in range(len(series))]
# dataFrame['day'] =  [series.index[i].day for i in range(len(series))]
# dataFrame['temperature'] =  [series[i] for i in range(len(series))]
temps = DataFrame(series.values)
dataFrame = concat([temps.shift(3), temps.shift(2), temps.shift(1), temps], axis = 1)
dataFrame.columns = ['t-2', 't-1', 't', 't+1']
print(dataFrame.head(5))
