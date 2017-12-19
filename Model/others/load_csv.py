#load CSV 
import csv
import numpy as np
from pandas import read_csv
from pandas import set_option

# load using csv
# filename = 'pima-indians-diabetes.data.csv'
# raw_data = open(filename, 'r')
# reader = csv.reader(raw_data, delimiter = ',', quoting = csv.QUOTE_NONE)
# x = list(reader)
# data = np.array(x).astype('float')
# print(data.shape)

# load using pandas and some operations
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names = names)
set_option('display.width', 100)
set_option('precision', 3)
description = data.describe()
peek = data.head(20)
types = data.dtypes
class_counts = data.groupby('class').size()
correlations = data.corr(method = 'pearson')
skew = data.skew()
print(skew)
