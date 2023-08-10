import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from category_encoders import OneHotEncoder
import category_encoders as ce
from scipy import stats
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
from sklearn.feature_selection import SelectKBest, f_regression
# read data
data = pd.read_csv("megastore-regression-dataset.csv")
# features
X = data.iloc[:, 0:18]
# target
Y = data['Profit']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 195, shuffle= True)
def remove_outliers(df, columns, threshold=3):

    new_df = df.copy()
    for col in columns:
        mean = np.mean(new_df[col])
        std_dev = np.std(new_df[col])
        z_scores = np.abs((new_df[col] - mean) / std_dev)
        new_df = new_df[z_scores <= threshold]
    return new_df



def date_split(X_train):
    date = pd.to_datetime(X_train['Order Date'])
    # X['Order Day'] = date.dt.day
    # X['Order Month'] = date.dt.month
    X_train['Order Year'] = date.dt.year

    date2 = pd.to_datetime(X_train['Ship Date'])

    #****************************************************8
    X_train['Ship Year'] = date2.dt.year

    X_train.drop('Order Date', axis=1, inplace=True)
    X_train.drop('Ship Date', axis=1, inplace=True)
    return X_train

# check and drop null rows
def check_null(X_train):
    # print(X.isnull().sum())
    X_train.dropna(axis=0, how='any', inplace=True)
    return X_train


# convert string columns to numeric data
#data scaling using z-score
def data_scaling(X_train):
    X_train['Row ID'] = zscore(X_train['Row ID'])
    X_train['Postal Code'] = zscore(X_train['Postal Code'])
    X_train['Quantity'] = zscore(X_train['Quantity'])
    X_train['Sales'] = zscore(X_train['Sales'])

    return X_train


# split Order ID
def split(X_train):
    X_train["Order_Country"] = X_train["Order ID"].str.extract('(\D+)', expand=False).str.strip()
    X_train["Year"]=X_train["Order ID"].str.extract('(\d+)')
    X_train["Order_Number"] = X_train["Order ID"].str.extract('(\d+)')

    X_train["Product_Type"] = X_train["Product ID"].str.extract('(\D+)', expand=False).str.strip()

    #*************************************************************

    X_train["Product_Ay7aga"] = X_train["Product ID"].str.extract('(\D+)', expand=False).str.strip()

    #******************************************************************************************
    X_train["Product_Number"] = X_train["Product ID"].str.extract('(\d+)')


########################################################################
    X_train["Customer_Ay7aga"] = X_train["Customer ID"].str.extract('(\D+)', expand=False).str.strip()

    #****************************************************************
    X_train["Customer_Number"] = X_train["Customer ID"].str.extract('(\d+)')


#####################################################################
    X_train["Main"] = X_train["CategoryTree"].str.extract('(\D+)', expand=False).str.strip()

    #********************************************************
    X_train["Sub"] = X_train["CategoryTree"].str.extract('(\D+)', expand=False).str.strip()

    #*************************************************************************
    X_train.drop('Customer ID', axis=1, inplace=True)
    X_train.drop('Order ID', axis=1, inplace=True)
    X_train.drop('Product ID', axis=1, inplace=True)
    X_train.drop('CategoryTree', axis=1, inplace=True)
    return X_train
def One_Hot_Encoder(X_train):

    #*************************************************************
    OneHotEncoder(cols=['Ship Mode',  'Region','Segment'
                        , 'Product_Type','Product_Ay7aga', 'Customer_Ay7aga','Main', 'Sub', 'State', 'City', 'Customer Name', 'Product Name']).fit(X_train).transform(X_train)
    X_train.drop('Ship Mode', axis=1, inplace=True)
    X_train.drop('Region', axis=1, inplace=True)

    return X_train

#Binary Encoder
def My_Binary_Encouder(X_train):
    X_train["Order_Countryyyyyyyyyy"] = np.where(X_train["Order_Country"].str.contains("US"), 1, 0)

    #***********************************************************
    X_train["Country_State"] = np.where(X_train["Country"].str.contains("United States"), 1, 0)

#************************************************************************
    X_train.drop('Order_Country', axis=1, inplace=True)
    X_train.drop('Country', axis=1, inplace=True)
    #X_train.drop('Country_State', axis=1, inplace=True)
    return X_train

def drop_cols(X_train):
    X_train.drop('Row ID', axis=1, inplace=True)
    X_train.drop('Postal Code', axis=1, inplace=True)
    X_train.drop('Country_State', axis=1, inplace=True)
    X_train.drop('Product_Type', axis=1, inplace=True)
    X_train.drop('Product_Ay7aga', axis=1, inplace=True)
    X_train.drop('Customer_Ay7aga', axis=1, inplace=True)
    X_train.drop('Main', axis=1, inplace=True)
    X_train.drop('Sub', axis=1, inplace=True)
    X_train.drop('City', axis=1, inplace=True)
    X_train.drop('State', axis=1, inplace=True)
    X_train.drop('Customer Name', axis=1, inplace=True)
    X_train.drop('Product Name', axis=1, inplace=True)
    X_train.drop('Order_Number', axis=1, inplace=True)
    X_train.drop('Year', axis=1, inplace=True)
    X_train.drop('Segment', axis=1, inplace=True)
    X_train.drop('Quantity', axis=1, inplace=True)
    #X_train.drop('Ship Year', axis=1, inplace=True)
    X_train.drop('Order_Countryyyyyyyyyy', axis=1, inplace=True)
    X_train.drop('Product_Number', axis=1, inplace=True)
    X_train.drop('Customer_Number', axis=1, inplace=True)
    X_train.drop('Ship Year', axis=1, inplace=True)

    return X_train
