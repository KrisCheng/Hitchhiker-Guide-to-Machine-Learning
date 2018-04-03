#!/usr/bin/python
# -*- coding: utf-8 -*-


from pandas import Series
from pandas import read_csv

# series = read_csv('daily-total-female-births-in-cal.csv', 
#                   header = 0, 
#                   parse_dates = [0], 
#                   index_col = 0, 
#                   squeeze = True)
# print(type(series))
# print(series.head())

series = Series.from_csv('daily-total-female-births-in-cal.csv',  header = 0)
print(series.describe())