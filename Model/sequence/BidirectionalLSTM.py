#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Desc: the Bidirectional LSTM model for cumulative sum classification prediction problem, based on keras.
Author: Kris Peng
Copyright (c) 2018 - Kris Peng <kris.dacpc@gmail.com>
'''

from random import random
from numpy import array
from numpy import cumsum
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

def get_sequence(n_timesteps):
    X = array([random() for _ in range(n_timesteps)])
    metric = n_timesteps / 4
    y = array([0 if x < metric else 1 for x in cumsum(X)])
    return X, y

def get_sequences(n_sequences, n_timesteps):
    seqX, seqY = [], []
    for _ in range(n_sequences):
        X, y = get_sequence(n_timesteps)
        seqX.append(X)
        seqY.append(y)
    seqX = array(seqX).reshape([n_sequences, n_timesteps, 1])
    seqY = array(seqY).reshape([n_sequences, n_timesteps, 1])
    return seqX, seqY

n_timesteps = 10

# define LSTM
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences = True), input_shape = (n_timesteps, 1)))
model.add(TimeDistributed(Dense(1, activation = 'sigmoid')))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
print(model.summary())

# train LSTM
X, y = get_sequences(100000, n_timesteps)
model.fit(X, y, epochs = 1, batch_size = 10)

# evaluate LSTM
X, y = get_sequences(1000, n_timesteps)
loss, acc = model.evaluate(X, y, verbose = 0)
print("Loss : %s , Acc : %s %%" % (loss, acc * 100))

# # make predictions
for _ in range(10):
    X, y = get_sequences(1, 10)
    yhat = model.predict_classes(X, verbose = 0)
    print("Actual : %s " % (y.reshape(n_timesteps)))
    print("Predict: %s " % (yhat.reshape(n_timesteps)))
    print("Correct: %s" % (array_equal(y, yhat)))

