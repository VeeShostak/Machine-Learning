# Multiple Linear Regression


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Dummy Vars: (always omit 1)
# State is categorical represent as 0 1.
# AND our categorical variables are NOT comparable, cant be ranked
# and not relational
# 
# Since D2 = 1 - D1 then if you include both D1 and D2 you get:
# y = b0 + b1x1 + b2x2 + b3x3 + b4D1 + b5(1 - D1)
#    = b0 + b5 + b1x1 + b2x2 + b3x3 + (b4 - b5)D1
#    = b0* + b1x1 + b2x2 + b3x3 + b4* D1 with b0* = b0 + b5 and b4* = b4 - b5
# Therefore the information of the redundant dummy variable D2 is going into the constant b0.
# =============================================================================

# Import the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
# encode
X[:, 3] = labelencoder.fit_transform(X[:, 3])
# dummy vars for categorical State var
onehotencoder = OneHotEncoder(categorical_features = [3]) # specify column
X = onehotencoder.fit_transform(X).toarray()

# omit Dummy Variable Trap (lib for Linear Regression will take care of it, but can do manually)
X = X[:, 1:]

# Split  dataset into the training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
# calculates a single regression line containing all variables.
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

