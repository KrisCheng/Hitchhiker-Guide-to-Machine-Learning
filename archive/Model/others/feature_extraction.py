from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names = names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

# # feature extrcation
# test = SelectKBest(score_func = chi2, k = 4)
# fit = test.fit(X, Y)
# # summarize scores
# set_printoptions(precision = 3)
# print(fit.scores_)
# features = fit.transform(X)
# # summarize selected features
# print(features[0:5, :])

# # feature extrcation
# model = LogisticRegression()
# rfe = RFE(model, 3)
# fit = rfe.fit(X, Y)
# print("Num Features: %d" % fit.n_features_)
# print("Selected Features: %s" % fit.support_)
# print("Feature Ranking: %s" % fit.ranking_)

# feature extraction(PCA)
# pca = PCA(n_components = 3)
# fit = pca.fit(X)
# # summarize components
# print("Explained Variance: %s" % fit.explained_variance_ratio_)
# print(fit.components_)

# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
