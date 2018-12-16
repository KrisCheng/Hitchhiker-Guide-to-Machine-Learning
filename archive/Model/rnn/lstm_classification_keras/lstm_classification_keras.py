#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Desc: a implementation of sequence classificaion on LSTM, based on keras
Ref: https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
Author: Kris Peng
Copyright (c) 2017 - Kris Peng <kris.dacpc@gmail.com>
'''

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

numpy.random.seed(7)

top_words = 5000
print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)

max_review_length = 500

X_train = sequence.pad_sequences(X_train, maxlen = max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen = max_review_length)

embedding_vector_length = 32

model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())

model.fit(X_train, y_train, nb_epoch = 3, batch_size = 64)

scores = model.evaluate(X_test, y_test, verbose = 0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

