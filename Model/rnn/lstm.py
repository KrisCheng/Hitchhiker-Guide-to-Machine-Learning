#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Desc: a implementation of LSTM without frameworks, just use numpy
Author: Kris Peng
Copyright (c) 2017 - Kris Peng <kris.dacpc@gmail.com>
'''

import numpy as np

class RecurrentNeuralNetwork:
    
    def __init__ (self, xs, ys, rl, eo, lr):
        # input
        self.x = np.zeros(xs)
        self.xs = xs
        # output
        self.y = np.zeros(ys)
        self.ys = ys
        # weights
        self.w = np.random.random((ys, ys))
        # RMSProp
        self.G = np.zeros_like(self.w)
        # length of recurrent network
        self.rl = rl
        # learning rate
        self.lr = lr
        # array for storing input
        self.ia = np.zeros((rl + 1, xs))
        # array for storing cell states
        self.ca = np.zeros((rl + 1, ys))
        # array for storing output
        self.oa = np.zeros((rl + 1, ys))
        # array for storing hidden states
        self.ha = np.zeros((rl + 1, ys))
        # forget gate
        self.af = np.zeros((rl + 1, ys))
        # input gate
        self.ai = np.zeros((rl + 1, ys))
        # cell state
        self.ac = np.zeros((rl + 1, ys))
        # output gate
        self.ao = np.zeros((rl + 1, ys))
        # array of expected output values
        self.eo = np.vstack((np.zeros(eo.shape[0]), eo.T))
        # declare LSTM cell
        self.LSTM = LSTM(xs, ys, rl, lr)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forwardProp(self):
        for i in range(1, self.rl + 1):
            self.LSTM.x = np.hstack((self.ha[i - 1], self.x)) 
            cs, hs, f, inp, c, o = self.LSTM.forwardProp()

            maxI = np.argmax(self.x)
            self.x = np.zeros_like(self.x)
            self.x[maxI] = 1
            self.ia[i] = self.x
            # store cell state
            self.ca[i] = cs
            # store hidden state
            self.ha[i] = hs
            self.af[i] = f
            self.ai[i] = inp
            self.ac[i] = c
            self.ao[i] = o

            self.oa[i] = self.sigmoid(np.dot(self.w, hs))
            self.x = self.eo[i - 1]

        return self.oa

    def backProp(self):
        totalError = 0
        # cell state
        dfcs = np.zeros(self.ys)
        # hidden state
        dfhs = np.zeros(self.ys)

        tfu = np.zeros((self.ys, self.xs + self.ys))
        tiu = np.zeros((self.ys, self.xs + self.ys))
        tcu = np.zeros((self.ys, self.xs + self.ys))
        tou = np.zeros((self.ys, self.xs + self.ys))

        tu = np.zeros((self.ys, self.ys))

        for i in range(self.rl, -1, -1):
            error = self.oa[i] - self.eo[i]
            tu += np.dot(np.atleast_2d(error * self.dsigmoid(self.oa[i])), np.atleast_2d(self.ha[i]).T)
            error = np.dot(error, self.w)

            self.LSTM.x = np.hstack((self.ha[i - 1], self.ia[i]))

            self.LSTM.cs = self.ca[i]

            fu, iu, cu, ou, dfcs, dfhs = self.LSTM.backProp(error, self.ca[i - 1], self.af[i], self.ai[i], self.ac[i], self.ao[i], dfcs, dfhs)

            totalError += np.sum(error)

            tfu += fu
            tiu += iu
            tcu += cu
            tou += ou

        self.LSTM.update(tfu / self.rl, tiu / self.rl, tcu / self.rl, tou / self.rl)
        self.update(tu / self.rl)
        return totalError

    def update(self, u):
        self.G = 0.9 * self.G + 0.1 * u**2
        self.w -= self.lr / np.sqrt(self.G + 1e-8) * u
        return 

    def sample(self):
        for i in range(1, self.rl + 1):
            self.LSTM.x = np.hstack((self.ha[i - 1], self.x))
            cs, hs, f, inp, c, o = self.LSTM.forwardProp()

            maxI = np.argmax(self.x)

            self.x = np.zeros_like(self.x)
            self.x[maxI] = 1
            self.ia[i] = self.x

            self.ca[i] = cs

            self.ha[i] = hs
            self.af[i] = f
            self.ai[i] = inp
            self.ac[i] = c
            self.ao[i] = o

            self.oa[i] = self.sigmoid(np.dot(self.w, hs))

            maxI = np.argmax(self.oa[i])
            newX = np.zeros_like(self.x)
            newX[maxI] = 1
            self.x = newX
        return self.oa

class LSTM:

    def __init__(self, xs, ys, rl, lr):
        self.x = np.zeros(xs + ys)
        self.xs = xs + ys
        self.y = np.zeros(ys)
        self.ys = ys
        self.cs = np.zeros(ys)
        self.rl = rl
        self.lr = lr
        # forget gate
        self.f = np.random.random((ys, xs + ys))
        # input gate
        self.i = np.random.random((ys, xs + ys))
        # cell state
        self.c = np.random.random((ys, xs + ys))
        # output gate
        self.o = np.random.random((ys, xs + ys))

        self.Gf = np.zeros_like(self.f)
        self.Gi = np.zeros_like(self.i)
        self.Gc = np.zeros_like(self.c)
        self.Go = np.zeros_like(self.o)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tangent(self, x):
        return np.tanh(x)

    def dtangent(self, x):
        return 1 - np.tanh(x)**2

    def forwardProp(self):
        f = self.sigmoid(np.dot(self.f, self.x))
        self.cs *= f
        i = self.sigmoid(np.dot(self.i, self.x))
        c = self.tangent(np.dot(self.c, self.x))
        self.cs += i * c
        o = self.sigmoid(np.dot(self.o, self.x))
        self.y = o * self.tangent(self.cs)
        return self.cs, self.y, f, i, c, o

    def backProp(self, e, pcs, f, i , c, o, dfcs, dfhs):
        e = np.clip(e + dfhs, -6, 6)
        do = self.tangent(self.cs) * e
        ou = np.dot(np.atleast_2d(do * self.dtangent(o)).T, np.atleast_2d(self.x))
        dcs = np.clip(e * o * self.dtangent(self.cs) + dfcs, -6, 6)
        dc = dcs * i
        cu = np.dot(np.atleast_2d(dc * self.dtangent(c)).T, np.atleast_2d(self.x))
        di = dcs * c
        iu = np.dot(np.atleast_2d(di * self.dsigmoid(i)).T, np.atleast_2d(self.x))
        df = dcs * pcs
        fu = np.dot(np.atleast_2d(df * self.dsigmoid(f)).T, np.atleast_2d(self.x))
        dpcs = dcs * f
        dphs = np.dot(dc, self.c)[:self.ys] + np.dot(do, self.o)[:self.ys] + np.dot(di, self.i)[:self.ys] + np.dot(df, self.f)[:self.ys]
        return fu, iu, cu, ou, dpcs, dphs

    def update(self, fu, iu, cu, ou):
        self.Gf = 0.9 * self.Gf + 0.1 * fu**2
        self.Gi = 0.9 * self.Gi + 0.1 * iu**2
        self.Gc = 0.9 * self.Gc + 0.1 * cu**2
        self.Go = 0.9 * self.Go + 0.1 * ou**2

        self.f -= self.lr / np.sqrt(self.Gf + 1e-8) * fu
        self.i -= self.lr / np.sqrt(self.Gi + 1e-8) * iu
        self.c -= self.lr / np.sqrt(self.Gc + 1e-8) * cu
        self.o -= self.lr / np.sqrt(self.Go + 1e-8) * ou

def LoadText():
    with open("input.txt", "r") as text_file:
        data = text_file.read()
    text = list(data)
    outputSize = len(text)
    data = list(set(text))
    uniqueWords, dataSize = len(data), len(data)
    returnData = np.zeros((uniqueWords, dataSize))
    for i in range(0, dataSize):
        returnData[i][i] = 1
    returnData = np.append(returnData, np.atleast_2d(data), axis = 0)
    output = np.zeros((uniqueWords, outputSize))
    for i in range(0, outputSize):
        index = np.where(np.asarray(data) == text[i])
        output[:,i] = returnData[0:-1,index[0]].astype(float).ravel()
    return returnData, uniqueWords, output, outputSize, data

def ExportText(output, data):
    finalOutput = np.zeros_like(output)
    prob = np.zeros_like(output[0])
    outputText = ""
    print(len(data))
    print(output.shape[0])
    for i in range(0, output.shape[0]):
        for j in range(0, output.shape[1]):
            prob[j] = output[i][j] / np.sum(output[i])
        outputText += np.random.choice(data, p = prob)
    with open("output.txt", "w") as text_file:
        text_file.write(outputText)
    return 

# main process
print("Begin:")
iterations = 5000
learningRate = 0.001
returnData, numCategories, expecetedOutput, outputSize, data = LoadText()
print("Reading ...")
RNN = RecurrentNeuralNetwork(numCategories, numCategories, outputSize, expecetedOutput, learningRate)

for i in range(1, iterations):
    RNN.forwardProp()
    error = RNN.backProp()
    print("Error in Iteration ", i, ": ", error)
    if error > -100 and error < 100 or i % 100 == 0:
        seed = np.zeros_like(RNN.x)
        maxI = np.argmax(np.random.random(RNN.x.shape))
        seed[maxI] = 1
        RNN.x = seed
        output = RNN.sample()
        print(output)
        ExportText(output, data)
        print("Epoch Finished.")
    print("Completed...")