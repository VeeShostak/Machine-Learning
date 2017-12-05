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


# =================================

# =============================================================================
# Backward Elimination: (remove indep vars that are above sig level)
# choose significance level (SL = 0.005)
# push all independent vars into model (All In) (fit full model with all possible predictors)
# Consider Predictor with highest P-value, if p > SL got to step 4, else go to finish
# remove the var with highest P value (remove the predictor)
# rebuild. fit model without the variable (then go to step 3)

# p-value is a probability that we did not compute our coefficient correctly (rough explanation)
# (the lower the p value of indep var the more sig your indep var will be with dependent var.)
# =============================================================================


# Build the optimal model using Backward Elimination
import statsmodels.formula.api as sm

# curr library thinks our eq is y = b1x1 + b2x2 + b3x3 + b4D1 + b5(1 - D1
# append x0b0     (we'll add a columns of 1's, x0 = 1)
# x = add matrix of features X to 1 column of 50 ones
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1) # axis = 1. columns

# ============================
# START Backward Elimination
# ============================

# optimal matrix of features. (predictors where p < SL)
X_opt = X[:, [0, 1, 2, 3, 4, 5]] # push all independent vars into model (All In)

# push all independent vars into model (All In) (fit full model with all possible predictors)
# ordinary least squares regressor
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Consider Predictor with highest P-value, if p > SL got to step 4, else go to finish
regressor_OLS.summary() # display each ind var p valu (and other infp)


#x2 has highest (95%) p value
# so remove x2
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

# ... complete Backward Elimination
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5]] # removed x1, 94%
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]] # removed x2, 60% NOTE: use R sqaured and adjusted R squared to know if to keep marketing spen
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()
X_opt = X[:, [0, 3]] # removed x4, 60%
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() # R&D Spent



# ============================
# END Backward Elimination
# ============================