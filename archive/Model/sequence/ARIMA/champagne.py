#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Desc: the ARIMA model for champagne time series prediction.
Author: Kris Peng
Copyright (c) 2018 - Kris Peng <kris.dacpc@gmail.com>
'''

from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
from scipy.stats import boxcox
import warnings
import numpy

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

# monkey patch around bug in ARIMA class
def __getnewargs__(self):
	return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))

ARIMA.__getnewargs__ = __getnewargs__

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# Seasonal Line Plots
# series = Series.from_csv('dataset.csv')
# groups = series['1964':'1970'].groupby(TimeGrouper('A'))
# years = DataFrame()
# pyplot.figure()
# i = 1
# n_groups = len(groups)
# for name, group in groups:
# 	pyplot.subplot((n_groups*100) + 10 + i)
# 	i += 1
# 	pyplot.plot(group)
# pyplot.show()

# # Create a differenced series

# series = Series.from_csv('dataset.csv')
# X = series.values
# X = X.astype('float32')
# # difference data
# months_in_year = 12
# stationary = difference(X, months_in_year)
# stationary.index = series.index[months_in_year:]
# # check if stationary
# result = adfuller(stationary)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# print('Critical Values:')
# for key, value in result[4].items():
# 	print('\t%s: %.3f' % (key, value))
# # save
# stationary.to_csv('stationary.csv')
# # plot
# stationary.plot()
# pyplot.show()

# # ACF && PACF
# series = Series.from_csv('stationary.csv')
# pyplot.figure()
# pyplot.subplot(211)
# plot_acf(series, ax=pyplot.gca())
# pyplot.subplot(212)
# plot_pacf(series, ax=pyplot.gca())
# pyplot.show()

# # ARIMA model
# # load data
# series = Series.from_csv('dataset.csv')
# # prepare data
# X = series.values
# X = X.astype('float32')
# train_size = int(len(X) * 0.50)
# train, test = X[0:train_size], X[train_size:]
# # walk-forward validation
# history = [x for x in train]
# predictions = list()
# for i in range(len(test)):
# 	# difference data
# 	months_in_year = 12
# 	diff = difference(history, months_in_year)
# 	# predict
# 	model = ARIMA(diff, order=(1,1,1))
# 	model_fit = model.fit(trend='nc', disp=0)
# 	yhat = model_fit.forecast()[0]
# 	yhat = inverse_difference(history, yhat, months_in_year)
# 	predictions.append(yhat)
# 	# observation
# 	obs = test[i]
# 	history.append(obs)
# 	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# # report performance
# mse = mean_squared_error(test, predictions)
# rmse = sqrt(mse)
# print('RMSE: %.3f' % rmse)

# # evaluate an ARIMA model for a given order (p,d,q) and return RMSE
# def evaluate_arima_model(X, arima_order):
# 	# prepare training dataset
# 	X = X.astype('float32')
# 	train_size = int(len(X) * 0.50)
# 	train, test = X[0:train_size], X[train_size:]
# 	history = [x for x in train]
# 	# make predictions
# 	predictions = list()
# 	for t in range(len(test)):
# 		# difference data
# 		months_in_year = 12
# 		diff = difference(history, months_in_year)
# 		model = ARIMA(diff, order=arima_order)
# 		model_fit = model.fit(trend='nc', disp=0)
# 		yhat = model_fit.forecast()[0]
# 		yhat = inverse_difference(history, yhat, months_in_year)
# 		predictions.append(yhat)
# 		history.append(test[t])
# 	# calculate out of sample error
# 	mse = mean_squared_error(test, predictions)
# 	rmse = sqrt(mse)
# 	return rmse

# # evaluate combinations of p, d and q values for an ARIMA model
# def evaluate_models(dataset, p_values, d_values, q_values):
# 	dataset = dataset.astype('float32')
# 	best_score, best_cfg = float("inf"), None
# 	for p in p_values:
# 		for d in d_values:
# 			for q in q_values:
# 				order = (p,d,q)
# 				try:
# 					mse = evaluate_arima_model(dataset, order)
# 					if mse < best_score:
# 						best_score, best_cfg = mse, order
# 					print('ARIMA%s RMSE=%.3f' % (order,mse))
# 				except:
# 					continue
# 	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

# # load dataset
# series = Series.from_csv('dataset.csv')
# # evaluate parameters
# p_values = range(0, 7)
# d_values = range(0, 3)
# q_values = range(0, 7)
# warnings.filterwarnings("ignore")
# evaluate_models(series.values, p_values, d_values, q_values)

# # Fit the model
# # load data
# series = Series.from_csv('dataset.csv')
# # prepare data
# X = series.values
# X = X.astype('float32')
# # difference data
# months_in_year = 12
# diff = difference(X, months_in_year)
# # fit model
# model = ARIMA(diff, order=(0,0,1))
# model_fit = model.fit(trend='nc', disp=0)
# # bias constant, could be calculated from in-sample mean residual
# bias = 165.904728
# # save model
# model_fit.save('model.pkl')
# numpy.save('model_bias.npy', [bias])

# load and prepare datasets
dataset = Series.from_csv('dataset.csv')
X = dataset.values.astype('float32')
history = [x for x in X]
months_in_year = 12
validation = Series.from_csv('validation.csv')
y = validation.values.astype('float32')
# load model
model_fit = ARIMAResults.load('model.pkl')
bias = numpy.load('model_bias.npy')
# make first prediction
predictions = list()
yhat = float(model_fit.forecast()[0])
yhat = bias + inverse_difference(history, yhat, months_in_year)
predictions.append(yhat)
history.append(y[0])
print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))
# rolling forecasts
for i in range(1, len(y)):
	# difference data
	months_in_year = 12
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(0,0,1))
	model_fit = model.fit(trend='nc', disp=0)
	yhat = model_fit.forecast()[0]
	yhat = bias + inverse_difference(history, yhat, months_in_year)
	predictions.append(yhat)
	# observation
	obs = y[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
mse = mean_squared_error(y, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
pyplot.plot(y)
pyplot.plot(predictions, color='red')
pyplot.show()