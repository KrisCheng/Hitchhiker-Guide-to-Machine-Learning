#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Desc: the Vanilla LSTM model for echo sequence prediction problem, based on keras.
Author: Kris Peng
Copyright (c) 2018 - Kris Peng <kris.dacpc@gmail.com>
'''

from random import randint
from numpy import array
from numpy import argmax
from keras.models import Sequential
from keras.layers import LSTM, Dense

# generate sequence, a series of random number
def generate_sequence(length, n_features):
    return [randint(0, n_features - 1) for _ in range(length)]

# one hot encode sequence
def one_hot_encode(sequence, n_features):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_features)] 
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)

# one hot decode
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]

# generate sequence for lstm (Date Prepare)
def generate_example(length, n_features, out_index):
    sequence = generate_sequence(length, n_features)
    encoded = one_hot_encode(sequence, n_features)
    # reshape sequence to 3D
    X = encoded.reshape((1, length, n_features))
    y = encoded[out_index].reshape(1, n_features)
    return X, y

# define model
length = 5
n_features = 10 
out_index = 2
model = Sequential()
model.add(LSTM(25, input_shape = (length, n_features)))
model.add(Dense(n_features, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
print(model.summary())

# fit model
for i in range(10):
    X, y = generate_example(length, n_features, out_index)
    model.fit(X, y, epochs = 100, verbose = 2)

# evalute model
correct = 0
for i in range(100):
    X, y = generate_example(length, n_features, out_index)
    yhat = model.predict(X)
    if one_hot_decode(yhat) == one_hot_decode(y):
        correct += 1
print('Accuracy: %f %% ' % ((correct / 100) * 100.0))

# predicton on new data
X, y = generate_example(length, n_features, out_index)
yhat = model.predict(X)
print('Sequence: %s' % [one_hot_decode(x) for x in X])
print('Expected: %s' % one_hot_decode(y))
print('Predicted: %s' % one_hot_decode(yhat))