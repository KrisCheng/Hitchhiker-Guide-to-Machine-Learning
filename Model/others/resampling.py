from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names = names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

# # train and test dataset
# test_size = 0.33
# seed = 7
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state = seed)
# model = LogisticRegression()
# model.fit(X_train, Y_train)
# result = model.score(X_test, Y_test)
# print("Accuracy: %.3f%%" % (result * 100.0))

# cross validation
num_folds = 10
seed = 7
kfold = KFold(n_splits = num_folds, random_state = seed)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv = kfold)
print("Accuracy: %.3f%%  (%.3f%%)" % (results.mean() * 100.0, results.std() * 100.0))