import pickle
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from pre_processing import *
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from feature_selection import *
import time
import matplotlib.pyplot as plt

# read data from csv file
data = pd.read_csv("megastore-classification-dataset.csv")

# features
X = data.iloc[:, 0:19]

# target
Y = data['ReturnCategory']
Y = Y.to_frame()

# split data into train_test data -> 80-20
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1, shuffle=True)

# replace null in test data
X_test.fillna(0, inplace=True)
Y_test.fillna(0, inplace=True)


# First, remove null values from the input data
X_train = check_null(X_train)
# Next, apply feature engineering and date splitting to the input data
X_train = feature_engineering(X_train)
X_train = date_split(X_train)
# Next, apply outlier detection to remove any outliers from the input data
#X_train = outlier_detection(X_train)
# Next, apply data scaling to the input data
X_train = data_scaling(X_train)
# Next, apply one-hot encoding, label encoding and binary encoding to the input data
X_train = one_hot_encoder(X_train)
X_train = label_encoder(X_train)
X_train = binary_encoder(X_train)
# Next, drop unnecessary columns from the input data
X_train = drop_cols(X_train)
# Finally, apply label encoding to the target variable
Y_train = label_encoder_target(Y_train)

#########################################################

X_test = feature_engineering(X_test)
X_test = date_split(X_test)
#X_test = outlier_detection(X_test)
X_test = data_scaling(X_test)
X_test = one_hot_encoder(X_test)
X_test = label_encoder(X_test)
X_test = binary_encoder(X_test)
X_test = drop_cols(X_test)
Y_test = label_encoder_target(Y_test)


# Apply ANOVA feature selection on train, test data
# Call the ANOVA feature selection function on the input data
selected_features = anova(X_train, Y_train)
X_train = selected_features
X_test = X_test[selected_features.columns]



####################################################################
def random_forest(X_train, Y_train, X_test, Y_test):
    rfc = RandomForestClassifier(random_state=42)

    # Define the hyperparameter gridto search over
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2']
    }

    # Create a grid search object and fit it to the training data
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1)
    startTrain = time.time()
    grid_search.fit(X_train, Y_train.values.ravel())

    # Print the best hyperparameters and corresponding accuracy
    print("Best parameters:", grid_search.best_params_)
    print("Best accuracy:", grid_search.best_score_)
    endTrain = time.time()
    # Test the Random Forest classifier on the testing data using the best hyperparametersh
    best_rfc = grid_search.best_estimator_
    start_test = time.time()
    Y_test_pred = best_rfc.predict(X_test)
    end_test = time.time()
    print('random tree test data accuracy: ', accuracy_score(Y_test, Y_test_pred))
    print("Actual time for training", endTrain - startTrain)
    print("Actual time for Testing", end_test - start_test)
    model1 = best_rfc
    with open('random_forest.pkl', 'wb') as filename:
        pickle.dump(model1, filename)


def logistic(X_train, Y_train, X_test, Y_test):
    clf = LogisticRegression(random_state=195, solver='newton-cg', penalty='none', max_iter=1000)
    startTrain = time.time()
    clf.fit(X_train, Y_train)

    y_train_pred = clf.predict(X_train)
    endTrain = time.time()

    start_test = time.time()
    pred = clf.predict(X_test)
    end_test = time.time()
    print(" the train accuracy of logistic model equals:", accuracy_score(Y_train, y_train_pred))
    acc = accuracy_score(Y_test, pred)
    print("the test accuracy of logistic model equals :", acc)
    print("Actual time for training", endTrain - startTrain)
    print("Actual time for Testing", end_test - start_test)
    model2 = clf
    with open('logistic_regressionModel.pkl', 'wb') as filename:
        pickle.dump(model2, filename)


def SVM_rbf(X_train, Y_train, X_test, Y_test):
    # SVM classification
    clf = svm.SVC(kernel='rbf', gamma=0.5, C=10)  # rbf Kernel
    startTrain = time.time()
    clf.fit(X_train, Y_train)
    y_train_pred = clf.predict(X_train)
    endTrain = time.time()
    print(" svm_rbf train data Accuracy:", accuracy_score(Y_train, y_train_pred))
    start_test= time.time()
    prediction = clf.predict(X_test)
    end_test = time.time()
    print("svm_rbf test data Accuracy:", metrics.accuracy_score(Y_test, prediction))
    print(classification_report(Y_test, prediction))
    print("Actual time for training", endTrain - startTrain)
    print("Actual time for Testing", end_test - start_test)
    model3 = clf
    with open('rbf_Model.pkl', 'wb') as filename:
        pickle.dump(model3, filename)


def SVM_poly(X_train, Y_train, X_test, Y_test):
    # SVM classification
    clf = svm.SVC(kernel='poly', degree=2, C=100)  # poly Kernel
    startTrain = time.time()
    clf.fit(X_train, Y_train)
    y_train_pred = clf.predict(X_train)
    endTrain = time.time()
    print(" svm_poly train data Accuracy:", accuracy_score(Y_train, y_train_pred))
    start_test = time.time()
    prediction = clf.predict(X_test)
    end_test = time.time()
    print(" svm_poly test data Accuracy:", metrics.accuracy_score(Y_test, prediction))
    print(classification_report(Y_test, prediction))
    print("Actual time for training", endTrain - startTrain)
    print("Actual time for Testing", end_test - start_test)
    model4 = clf
    with open('poly_Model.pkl', 'wb') as filename:
        pickle.dump(model4, filename)

# Call the functions to run the models and print their accuracy
random_forest(X_train, Y_train, X_test, Y_test)
logistic(X_train, Y_train, X_test, Y_test)
SVM_rbf(X_train, Y_train, X_test, Y_test)
SVM_poly(X_train, Y_train, X_test, Y_test)

# Plot the bar graph of accuracy scores for all models
