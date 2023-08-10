from pre_processing import *
#from featureSelection import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import metrics
import time
import joblib
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression

# read data from csv file
data = pd.read_csv("megastore-regression-dataset.csv")
# features
X = data.iloc[:, 0:19]
# target
Y = data['Profit']



# check and drop null rows




###################################### SPLIT DATA INTO TRAIN-TEST #######################################
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 195, shuffle= True)
#print(X_train.columns)
#print(X_train['Discount'])
########################################### PRE_PROCESSING ###########################################

#X_train = check_null(X_train)

'''X_train=split(X_train)
X_train = One_Hot_Encoder(X_train)
X_train=My_Binary_Encouder(X_train)
X_train=date_split(X_train)
# data scaling using z-score
X_train = data_scaling(X_train)
X_train = drop_cols(X_train)


X_test = check_null(X_test)
X_test=split(X_test)
X_test = One_Hot_Encoder(X_test)
X_test=My_Binary_Encouder(X_test)
X_test=date_split(X_test)
# data scaling using z-score
X_test = data_scaling(X_test)
X_test = drop_cols(X_test)'''

#feature_selection(X_train,Y_train)

# # final features we should have
# X = ('Row ID', 'Order ID', 'Order Year', 'Ship Year', 'Ship Mode', 'Customer ID', 'Segment', 'City', 'State',
#      'Postal Code', 'Region', 'Product ID', 'CategoryTree', 'Sales', 'Quantity', 'Discount')


########################################### FEATURE SELECTION ###########################################
# from scipy.stats import chi2_contingency
# # Create a contingency table of the two columns
# contingency_table = pd.crosstab(X_train['Sales'], data['Profit'])
#
# # Perform the chi-squared test
# chi2, p_value, dof, expected = chi2_contingency(contingency_table)
#
# # Print the results
# print("Chi-squared statistic:", chi2)
# print("P-value:", p_value)
# # Create a contingency table of the two columns
# contingency_table = pd.crosstab(X_train['Quantity'], data['Profit'])
#
# # Perform the chi-squared test
# chi2, p_value, dof, expected = chi2_contingency(contingency_table)
#
# # Print the results
# print("Chi-squared statistic:", chi2)
# print("P-value:", p_value)



#print(X_train.columns)
# replace null in test data
X_test.fillna(0, inplace=True)
Y_test.fillna(0, inplace=True)


########################################### REGRESSION MODELS ###########################################
#print("0: Multiple regression model\n 1: Polynomial regression model")
#choice = int(input("Choose your model: "))


def multiple(X_train, Y_train, X_test, Y_test):
    cls = linear_model.LinearRegression()
    startTrain = time.time()
    cls.fit(X_train, Y_train)
    endTrain = time.time()
    start_test = time.time()
    prediction = cls.predict(X_test)
    end_test = time.time()
    print('Co-efficient of multiple regression', cls.coef_)
    print('Intercept of multiple regression model', cls.intercept_)
    print('R2 Score', metrics.r2_score(Y_test, prediction))
    print('Mean Square Error', metrics.mean_squared_error(np.asarray(Y_test), prediction))
    print("Actual time for training", endTrain - startTrain)
    print("Actual time for Testing", end_test - start_test)
    with open('multiple_regression', 'wb') as filename:
        pickle.dump(cls, filename)

def plynomial(X_train, Y_train, X_test, Y_test):
    poly_features = PolynomialFeatures(degree=2)
    # transform existing features to higher degree features
    X_train_poly = poly_features.fit_transform(X_train)

    # fit the transformed features to Linear Regression and calculate time for training
    poly_model = linear_model.LinearRegression()
    startTrain = time.time()
    poly_model.fit(X_train_poly, Y_train)
    endTrain = time.time()

    # predicting on training data-set and calculate time for testing
    y_train_predicted = poly_model.predict(X_train_poly)
    start_test = time.time()
    Y_pred = poly_model.predict(poly_features.transform(X_test))
    end_test = time.time()

    # predicting on test data-set
    prediction = poly_model.predict(poly_features.fit_transform(X_test))

    # print Co-efficient and statistics for polynomial regression
    print('Co-efficient of linear regression', poly_model.coef_)
    print('Intercept of linear regression model', poly_model.intercept_)
    print('R2 Score', metrics.r2_score(Y_test, prediction))
    print('Mean Square Error', metrics.mean_squared_error(Y_test, prediction))
    print("Actual time for training", endTrain - startTrain)
    print("Actual time for Testing", end_test - start_test)

    # test polynomial model on first sample
    true_pofit_value = np.asarray(Y_test)[0]
    predicted_profit_value = prediction[0]
    # print("The true profit value " + str(true_price_value))
    # print("The predicted profit  value " + str(predicted_price_value))

    joblib.dump(poly_model, 'polymodel')
    joblib.dump(poly_features, 'poilynomial_features_model')
# multiple regression model
#if choice == 0:
 #   multiple(X_train, Y_train, X_test, Y_test)

    # polynomial regression model
#elif choice == 1:
 #    plynomial(X_train, Y_train, X_test, Y_test)

