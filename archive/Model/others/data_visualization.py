from matplotlib import pyplot
from pandas import read_csv
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
import numpy as np

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names = names)
# data.hist()
# data.plot(kind = 'box', subplots = True, layout = (3, 3), sharex = False, sharey = False)

# plot correlation matrix
# correlations = data.corr()
# fig = pyplot.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(correlations, vmin = -1, vmax = 1)
# fig.colorbar(cax)
# ticks = numpy.arange(0, 9, 1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(names)
# ax.set_yticklabels(names)

# scatter_matrix(data)
# pyplot.show()

# Rescale data
dataframe = read_csv(filename, names = names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
scaler = MinMaxScaler(feature_range = (0, 1))
rescaledX = scaler.fit_transform(X)
np.set_printoptions(precision = 3)
print(rescaledX[0:5, :])