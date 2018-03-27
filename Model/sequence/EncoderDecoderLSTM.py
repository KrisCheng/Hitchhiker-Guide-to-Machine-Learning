#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Desc: the Encoder-Decoder LSTM model for Sequence to Sequence Prediction Problem,
calculate teh sum output of two input numbers, based on keras.
Author: Kris Peng
Copyright (c) 2018 - Kris Peng <kris.dacpc@gmail.com>
'''

from random import seed
from random import randint
from numpy import array
from numpy import argmax
from math import ceil
from math import log10
from math import sqrt
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, RepeatVector

# 1.generate random sequence
def random_sum_pairs(n_examples, n_numbers, largest):
    X = []
    y = []
    for i in range(n_examples):
        integer_list = [randint(1, largest) for _ in range(n_numbers)]
        sum_list = sum(integer_list)
        X.append(integer_list)
        y.append(sum_list)
    return X, y

# 2.convert data to string
def to_string(X, y, n_numbers, largest):
    # the max length of the string
    max_length = n_numbers * ceil(log10(largest+1)) + n_numbers - 1
    Xstr = []
    for pattern in X:
        strp = '+'.join([str(n) for n in pattern])
        # fixed length
        strp = ''.join([" " for _ in range(max_length - len(strp))]) + strp
        Xstr.append(strp)
    max_length = ceil(log10(n_numbers * (largest + 1)))
    ystr = []
    for pattern in y:
        strp = str(pattern)
        strp = ''.join([" " for _ in range(max_length - len(strp))]) + strp
        ystr.append(strp)
    return Xstr, ystr

# 3.integer encode strings
def integer_encode(X, y, alphabet):
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    Xenc = list()
    for pattern in X:
        integer_encoded = [char_to_int[char] for char in pattern]
        Xenc.append(integer_encoded)
    yenc = list()
    for pattern in y:
        integer_encoded = [char_to_int[char] for char in pattern]
        yenc.append(integer_encoded)
    return Xenc, yenc

# 4.one hot eocede
def one_hot_encode(X, y, max_int):
    Xenc = []
    for seq in X:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        Xenc.append(pattern)
    yenc = []
    for seq in y:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        yenc.append(pattern)
    return Xenc, yenc

# test codes (for upper methods)
# seed(1)
# n_examples = 1
# n_numbers = 2
# largest = 10
# # generate pairs
# X, y = random_sum_pairs(n_examples, n_numbers, largest)
# print(X, y)

# # convert to strings
# X, y = to_string(X, y, n_numbers, largest)
# print(X, y)

# # integer encode
# alphabet=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ']
# X, y = integer_encode(X, y, alphabet)
# print(X, y)

# # one hot eocode
# X, y = one_hot_encode(X, y, len(alphabet))
# print(X, y)

# warp up those methods to form a complete encoder
def generate_data(n_samples, n_numbers, largest, alphabet):
    # 1.generate pairs
    X, y = random_sum_pairs(n_samples, n_numbers, largest)
    # 2. pairs(num) --> string
    X, y = to_string(X, y, n_numbers, largest)
    # 3.integer encode
    X, y = integer_encode(X, y, alphabet)
    # 4.one hot eocode
    X, y = one_hot_encode(X, y, len(alphabet))
    # reuturn as numpy arrays
    X, y = array(X), array(y)
    return X, y

# invert encoding (decode)
def invert(seq, alphabet):
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    strings = list()
    for pattern in seq:
        string = int_to_char[argmax(pattern)]
        strings.append(string)
    return ''.join(strings)

# parameters setting
# number of math terms
n_terms = 3

# largest value
largest = 10

# scope
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ']

# size of alphabet
n_chars = len(alphabet)

# length of input sequence
n_in_seq_length = n_terms * ceil(log10(largest + 1)) + n_terms - 1 

# length of output sequence
n_out_seq_length = ceil(log10(n_terms * (largest + 1)))

# define model
model = Sequential()
model.add(LSTM(75, input_shape = (n_in_seq_length, n_chars)))
model.add(RepeatVector(n_out_seq_length))
model.add(LSTM(50, return_sequences = True))
model.add(TimeDistributed(Dense(n_chars, activation = 'softmax')))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())

# fit model
X, y = generate_data(75000, n_terms, largest, alphabet)
model.fit(X, y, epochs = 1, batch_size = 32)

# evaluate model
X, y = generate_data(100, n_terms, largest, alphabet)
loss, acc = model.evaluate(X, y, verbose = 0)
print('Loss: %f, Accuracy: %f' % (loss, acc * 100))

# predict
for _ in range(100):
    # generate pairs
    X, y = generate_data(1, n_terms, largest, alphabet)
    # make prediction
    yhat = model.predict(X, verbose = 0)
    # decode input, expected and make prediction
    in_seq = invert(X[0], alphabet)
    out_seq = invert(y[0], alphabet)
    predicted = invert(yhat[0], alphabet)
    print('%s = %s (expeced %s)' % (in_seq, predicted, out_seq))