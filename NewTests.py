from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.datasets import make_classification
from main import selected_features
from main import *
from pre_processing import *
import pickle
from feature_selection import *

data = pd.read_csv("megastore-tas-test-classification.csv")
X1 = data.iloc[:, 0:19]

# target
Y1 = data['ReturnCategory']
Y1 = Y1.to_frame()

X1 = feature_engineering(X1)
X1 = date_split(X1)
# Next, apply outlier detection to remove any outliers from the input data
# X_train = outlier_detection(X_train)
# Next, apply data scaling to the input data
X1 = data_scaling(X1)
# Next, apply one-hot encoding, label encoding and binary encoding to the input data
X1 = one_hot_encoder(X1)
X1 = label_encoder(X1)
X1 = binary_encoder(X1)
# Next, drop unnecessary columns from the input data
X1 = drop_cols(X1)
# Finally, apply label encoding to the target variable
Y1 = label_encoder_target(Y1)

X1 = X1.fillna(0)
Y1 = Y1.fillna(0)

selected_feature = anova(X1, Y1)
X1 = X1[selected_features.columns]

print("#################################### A7la Random Forest Model ##################################")
print("\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")
with open('random_forest.pkl', 'rb') as filename:
    model1 = pickle.load(filename)

# logistic regression
y_pred1 = model1.predict(X1)
accuracy1 = accuracy_score(Y1, y_pred1)
precision1 = precision_score(Y1, y_pred1, average='micro')
recall1 = recall_score(Y1, y_pred1, average='micro')
f1 = f1_score(Y1, y_pred1, average='micro')

# print the results
print("the accuracy of the random forest model = ", "(", accuracy1, ")")
print("the precision of the random forest model = ", "(", precision1)
print("the recall of the random forest model = ", "(", recall1, ")")
print("the f1_score of the random forest model = ", "(", f1, ")")

print("#################################### A7la Logistic Regression Model #############################")
print("\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")
# random forest
with open('logistic_regressionModel.pkl', 'rb') as filename:
    model2 = pickle.load(filename)

y_pred2 = model2.predict(X1)
accuracy2 = accuracy_score(Y1, y_pred2)
precision2 = precision_score(Y1, y_pred2, average='micro')
recall2 = recall_score(Y1, y_pred2, average='micro')
f2 = f1_score(Y1, y_pred2,average='micro')

# print the results
print("the accuracy of the logisitic model = ", "(", accuracy2, ")")
print("the precision of the logistic model = ", "(", precision2, ")")
print("the recall of the logistic model = ", "(", recall2, ")")
print("the f1_score of the logistic model = ", "(", f2, ")")


print("#################################### A7la RBF SVM Model ###################################")
print("\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")
with open('rbf_Model.pkl', 'rb') as filename:
    model3 = pickle.load(filename)

y_pred3 = model3.predict(X1)
accuracy3 = accuracy_score(Y1, y_pred3)
precision3 = precision_score(Y1, y_pred3, average='micro')
recall3 = recall_score(Y1, y_pred3, average='micro')
f3 = f1_score(Y1, y_pred3,average='micro')

# print the results
print("the accuracy of the rbf_Model = ", "(", accuracy3, ")")
print("the precision of the rbf_Model = ", "(", precision3, ")")
print("the recall of the rbf_Model = ", "(", recall3, ")")
print("the f1_score of the rbf_Model = ", "(", f3, ")")

print("#################################### A7la Polynomial Model ##################################")
print("\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")
with open('poly_Model.pkl', 'rb') as filename:
    model4 = pickle.load(filename)
y_pred4 = model4.predict(X1)
accuracy4 = accuracy_score(Y1, y_pred4)
precision4 = precision_score(Y1, y_pred4, average='micro')
recall4 = recall_score(Y1, y_pred4, average='micro')
f4 = f1_score(Y1, y_pred4,average='micro')

# print the results
print("the accuracy of the poly_Model = ", "(", accuracy4, ")")
print("the precision of the poly_Model = ", "(", precision4, ")")
print("the recall of the poly_Model = ", "(", recall4, ")")
print("the f1_score of the poly_Model = ", "(", f4, ")")
print("#################################### 4okraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaannnnn ##############################")
print("\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")
