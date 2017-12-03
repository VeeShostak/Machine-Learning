# =============================================================================
# # Data Preprocessing
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# Numpy arrays and pandas dataframes use 2-d arrays, much like matlab and r. The 
# first argument typically represents a slice of rows, while the second 
# represents a slice of columns. So list[:,:3] would be all rows, first three columns.
# =============================================================================

# Importing the dataset
dataset = pd.read_csv('Data.csv')
# create arr of independent vars. (all rows from index 0-2)
X = dataset.iloc[:, :-1].values
# create arr of dependent vars
y = dataset.iloc[:, 3].values


# Taking care of missing data
# replace missing data by the mean of all values in the column that contains this missing data
from sklearn.preprocessing import Imputer
# missing vlaues have NaN, take means of columns axis = 0
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# fit for all rows for age and salary columns
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# Encoding categorical data
# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

# for all rows for first column (country), encode
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# since 1 >0 , ... model will think that one has higher value than the other ,
# we have no relational order (if we had size it would make sense, but not for countries)
# use dummy vars to represent country, to make sure learning machine models dont attribute an order into categorical values
onehotencoder = OneHotEncoder(categorical_features = [0]) # specify column
X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# ================


# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
# test size 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# =============================================================================
# Feature Scaling
# # ex. salary and age not on same scale, euclidean distance will be dominated by the salart
# 31000 96100000
# 21    441      in equation, 441 won'te exist as it will be dominated by 9610000
# even if ML algo not based on Euclidean dist, like dicision trees, it will make alg converge faster
# =============================================================================
# scale using standardisation or normalisation
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# for training set we have to fit, then transform
X_train = sc_X.fit_transform(X_train)
# sc object already fitted on the traingn set, just transoform
X_test = sc_X.transform(X_test)
# do we need to scale dumy vars? dependes , lose imterpretation of belonging to specific country
# we will scale sin e we have no imterpretation to make here


# no need fo future scaling of dependent var
# bu for regression for a dependent var with large scale of values, we would need to apply

#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)








