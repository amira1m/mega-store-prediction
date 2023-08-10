import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from pre_processing import *
from scipy.stats import chi2_contingency


# Read data from csv file
data = pd.read_csv('megastore-classification-dataset.csv')

# features
X = data.iloc[:, 0:19]
# target
Y = data['ReturnCategory']

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=195, shuffle=True)


def correlation(X, Y):
    # Calculate correlation between features
    corr = data.corr()

    # Select highly correlated features (correlation > 0.02)
    top_features = corr.index[abs(corr['ReturnCategory']) > 0.02]

    # Plot correlation
    plt.subplots(figsize=(12, 8))
    top_corr = data[top_features].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()

    # Create a contingency table of the two columns
    contingency_table = pd.crosstab(data['Column_A'], data['Column_B'])

    # Perform the chi-squared test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    # Print the results
    print("Chi-squared statistic:", chi2)
    print("P-value:", p_value)


def anova(X, Y, k=5):
    # Perform ANOVA feature selection with available samples
    f_values, p_values = f_classif(X, Y)
    sorted_idx = np.argsort(f_values)[::-1]
    selected_features = X.iloc[:, sorted_idx[:k]]

    # Print the selected features
    print(selected_features.columns)
    return selected_features


