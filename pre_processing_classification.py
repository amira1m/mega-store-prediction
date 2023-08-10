import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from category_encoders import OneHotEncoder
import category_encoders as ce
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# read data
data = pd.read_csv("megastore-classification-dataset.csv")

# features
X = data.iloc[:, 0:19]
# target
Y = data['ReturnCategory']
Y = Y.to_frame()

# Splitting the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=195, shuffle=True)


def check_null(features):
    features.dropna(axis=0, how='any', inplace=True)
    return features


def feature_engineering(features):
    # extract new features from 'Order ID' column
    features['Order_Country_engineered'] = features['Order ID'].str.extract('(\D+)', expand=False).str.strip()
    # features['Order_Year_engineered'] = features['Order ID'].str.extract('(\D+)')
    # features['Order_ID_engineered'] = features['Order ID'].str.extract('(\d+)')

    # extract new features from 'Customer ID' column
    # features['Customer_Name_engineered'] = features['Customer ID'].str.extract('(\D+)', expand=False).str.strip()
    # features['Customer_ID_engineered'] = features['Customer ID'].str.extract('(\d+)')

    # extract new features from 'Product ID' column
    features['Product_Category_engineered'] = features['Product ID'].str.extract('(\D+)', expand=False).str.strip()
    features['Product_Type_engineered'] = features['Product ID'].str.extract('(\D+)', expand=False).str.strip()
    # features['Product_ID_engineered'] = features['Product ID'].str.extract('(\d+)')

    # # extract new features from 'CategoryTree' column
    # features['Main_engineered'] = features['CategoryTree'].str.extract('(\D+)', expand=False).str.strip()
    # features['Sub_engineered'] = features['CategoryTree'].str.extract('(\D+)', expand=False).str.strip()

    features.drop(['Order ID', 'Customer ID', 'Product ID', 'CategoryTree'], axis=1, inplace=True)
    return features


def date_split(features):
    # extract Oredr Year from 'Order date' column
    date = pd.to_datetime(features['Order Date'])
    features['Order_Year_splitted'] = date.dt.year

    # extract Ship Year from 'Ship date' column
    date2 = pd.to_datetime(features['Ship Date'])
    features['Ship_Year_splitted'] = date2.dt.year

    # drop original columns
    features.drop(['Order Date', 'Ship Date'], axis=1, inplace=True)
    return features


def outlier_detection(features):
    # Select the columns we want to work with
    columns_to_check = ['Sales', 'Quantity', 'Discount']



    # Calculate the median and interquartile range (IQR) of each selected column
    column_medians = features[columns_to_check].median()
    column_iqrs = features[columns_to_check].quantile(0.75) - features[columns_to_check].quantile(0.25)

    # Calculate the lower and upper bounds for outliers for each column
    lower_bounds = column_medians - 1.5 * column_iqrs
    upper_bounds = column_medians + 1.5 * column_iqrs

    # Replace any values in the selected columns that fall outside the bounds with the column median
    for col in columns_to_check:
        features.loc[(features[col] < lower_bounds[col]) | (features[col] > upper_bounds[col]), col] = column_medians[col]

    return features


def data_scaling(features):
    # features['Row ID'] = zscore(features['Row ID'])
    # features['Postal Code'] = zscore(features['Postal Code'])
    features['Sales'] = zscore(features['Sales'])
    features['Quantity'] = zscore(features['Quantity'])
    # features['Discount'] = zscore(features['Discount'])
    return features


def one_hot_encoder(features):
    # spicify columns to apply one hot encoding
    encoder = OneHotEncoder(cols=['Customer Name', 'Segment', 'City', 'State', 'Region', 'Product_Category_engineered',
                                  'Product_Type_engineered', 'Product Name'])
    features = encoder.fit_transform(features)
    return features


# def one_hot_encoder(features):
#     # spicify columns to apply one hot encoding
#     cols_to_encode = ['Customer Name', 'Segment', 'City', 'State', 'Region', 'Product_Category_engineered',
#                       'Product_Type_engineered', 'Product Name']
#
#     # check that all columns to encode exist in the input DataFrame
#     missing_cols = set(cols_to_encode) - set(features.columns)
#     if missing_cols:
#         raise ValueError(f'Columns {missing_cols} do not exist in the input DataFrame')
#
#     # apply one-hot encoding to categorical features
#     encoder = OneHotEncoder(cols=cols_to_encode)
#     features = encoder.fit_transform(features)
#
#     return features


def label_encoder(features):
    lbl = LabelEncoder()
    # apply label encoding
    features['Ship Mode'] = lbl.fit_transform(features['Ship Mode'])
    return features


def binary_encoder(features):
    # apply binary encoding
    features['Order_Country_engineered_binEnc'] = np.where(features['Order_Country_engineered'].str.contains("US"), 1,
                                                           0)
    # drop original columns
    features.drop(['Order_Country_engineered'], axis=1, inplace=True)
    return features


def drop_cols(features):
    cols_to_drop = ['Product_Category_engineered', 'Product_Type_engineered', 'Customer Name', 'Country', 'City',
                    'State', 'Product Name']
    cols_to_drop = [col for col in cols_to_drop if col in features.columns]
    features.drop(cols_to_drop, axis=1, inplace=True)
    return features


def label_encoder_target(target):
    lbl = LabelEncoder()
    # apply label encoding
    target['ReturnCategory'] = lbl.fit_transform(target['ReturnCategory'])
    return target


def preprocessing(features, target):
    # First, remove null values from the input data
    #features = check_null(features)

    #Next, apply feature engineering and date splitting to the input data
    features = feature_engineering(features)
    features = date_split(features)

    # Next, apply outlier detection to remove any outliers from the input data
    features = outlier_detection(features)

    # Next, apply data scaling to the input data
    features = data_scaling(features)

    # Next, apply one-hot encoding, label encoding and binary encoding to the input data
    features = one_hot_encoder(features)
    features = label_encoder(features)
    features = binary_encoder(features)

    # Next, drop unnecessary columns from the input data
    features = drop_cols(features)

    # Finally, apply label encoding to the target variable
    features = label_encoder_target(target)

    return features


# def drop_unique_columns(features):
#     # Get the number of rows in the DataFrame
#     num_rows = features.shape[0]
#
#     # Get the number of unique values for each column
#     unique_counts = features.nunique()
#
#     # Identify columns where the number of unique values is equal to the number of rows
#     drop_cols = list(unique_counts[unique_counts == num_rows].index)
#
#     # Drop the identified columns from the DataFrame
#     features = features.drop(drop_cols, axis=1, inplace=True)
#
#     return features