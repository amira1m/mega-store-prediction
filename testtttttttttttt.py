import pickle
# import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from pre_processing import *
from regressionModels import *
from sklearn import svm
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
import time
import joblib


data = pd.read_csv('megastore-tas-test-regression.csv')


# features
X = data.iloc[:, 0:19]
# target
Y = data['Profit']


# Feature processing
X = check_null(X)
X = split(X)
X = One_Hot_Encoder(X)
X = My_Binary_Encouder(X)
X = date_split(X)
# data scaling using z-score
X = data_scaling(X)
X = drop_cols(X)

X = X.fillna(0)
Y = Y.fillna(0)
# change dataframes into arrays
X = np.array(X)
Y = np.array(Y)



cls = pickle.load(open('multiple_regression', 'rb'))
Predi = cls.predict(X)

print('Co-efficient of multiple regression', cls.coef_)
print('Intercept of multiple regression model', cls.intercept_)
print('R2 Score', metrics.r2_score(Y, Predi))
print('Mean Square Error', metrics.mean_squared_error(np.asarray(Y), Predi))

print("slgsdgnsldgkjslkgslkgjsorigjoig")


poilynomia_features_model = joblib.load('poilynomial_features_model')
poly_model = joblib.load('polymodel')

polyfeat = poilynomia_features_model.transform(X)
predd = poly_model.predict(polyfeat)
print('Co-efficient of linear regression', poly_model.coef_)
print('Intercept of linear regression model', poly_model.intercept_)
print('R2 Score', metrics.r2_score(Y, predd))
print('Mean Square Error', metrics.mean_squared_error(Y, predd))


print("a33333333333333333333333333333333333")