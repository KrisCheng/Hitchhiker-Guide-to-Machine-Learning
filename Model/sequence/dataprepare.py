#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Desc: some data preprocessing practice, based on Python3.
Author: Kris Peng
Copyright (c) 2018 - Kris Peng <kris.dacpc@gmail.com>
'''

from pandas import Series
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import pad_sequences
from math import sqrt

# 1.normalization
print("1. Normalization Demo:")
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

series = Series(data)
print(series)

values = series.values
values = values.reshape((len(values), 1))

scaler = MinMaxScaler(feature_range = (0, 1))
scaler = scaler.fit(values)

print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))

normalized = scaler.transform(values)
print(normalized)

# inversed
inversed = scaler.inverse_transform(normalized)
print(inversed)

# 2.standardization
print("2. Standardization Demo:")
data = [1.0, 5.5, 9.0, 2.6, 8.8, 3.0, 4.1, 7.9, 6.3]
series = Series(data)
print(series)

values = series.values
values = values.reshape((len(values), 1))

scaler = StandardScaler()
scaler = scaler.fit(values)
print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))

standardized = scaler.transform(values)
print(standardized)

inversed = scaler.inverse_transform(standardized)
print(inversed)

# 3.pad sequence
print("3. Pad Sequence:")
sequences = [
    [1, 2, 3, 4],
       [1, 2, 3],
            [1]
]
padded = pad_sequences(sequences, padding = 'post')
print(padded)

# truncate
truncated = pad_sequences(sequences, maxlen = 2)
print(truncated)

truncated = pad_sequences(sequences, maxlen = 2, truncating = 'post')
print(truncated)

# 4. Sequence Prediction --> Supervised Learning
print("4. Sequence Prediction --> Supervised Learning:")
df = DataFrame()
df['t'] = [x for x in range(10)]
df['t+1'] = df['t'].shift(-1)
print(df)