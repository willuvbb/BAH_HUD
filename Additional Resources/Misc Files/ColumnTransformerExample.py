# Column Transformer Example

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer



features = dataset[['R&D Spend','Administration','Marketing Spend','State','Profit']]

# preprocess = make_column_transformer(
#     (OneHotEncoder(), ['State']),
#     (StandardScaler(), ['R&D Spend','Administration','Marketing Spend','Profit'])
#     )

preprocess = make_column_transformer(
    (OneHotEncoder(), ['State']),
    remainder='passthrough'
    )

fred = preprocess.fit_transform(features)

X = fred[:,:-1]
y = fred[:,-1]


# Avoid the "dummy variable trap"
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)).astype(int), values=X,axis=1)

X_opt = X[:,[0, 1, 2, 3, 4, 5]]
#ordinary least squares regressor object
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
# see from table that x2 (the second remaining dummy variable for state...
# has highest p-value, much greater than SL, so remove it and refit model
X_opt = X[:,[0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
# x1 not relevant (the other state one...) remove it
X_opt = X[:,[0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
#the 5th index has P-value 0.06, which is not bad... can remove it if you want to be super aligned to algorithm
X_opt = X[:,[0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()




# ## AUTOMATIC BACKWARD ELIMINATION
# import statsmodels.formula.api as sm
#
# def backwardElimination(x, sl):
#     numVars = len(x[0])
#     for i in range(0, numVars):
#         regressor_OLS = sm.OLS(y, x).fit()
#         maxVar = max(regressor_OLS.pvalues).astype(float)
#         if maxVar > sl:
#             for j in range(0, numVars - i):
#                 if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                     x = np.delete(x, j, 1)
#     regressor_OLS.summary()
#     return x
#
#
# SL = 0.05
# X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# X_Modeled = backwardElimination(X_opt, SL)
#
#
# # Backward Elimination with p - values and Adjusted R Squared:
# import statsmodels.formula.api as sm
#
#
# def backwardElimination(x, SL):
#     numVars = len(x[0])
#     temp = np.zeros((50, 6)).astype(int)
#     for i in range(0, numVars):
#         regressor_OLS = sm.OLS(y, x).fit()
#         maxVar = max(regressor_OLS.pvalues).astype(float)
#         adjR_before = regressor_OLS.rsquared_adj.astype(float)
#         if maxVar > SL:
#             for j in range(0, numVars - i):
#                 if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                     temp[:, j] = x[:, j]
#                     x = np.delete(x, j, 1)
#                     tmp_regressor = sm.OLS(y, x).fit()
#                     adjR_after = tmp_regressor.rsquared_adj.astype(float)
#                     if (adjR_before >= adjR_after):
#                         x_rollback = np.hstack((x, temp[:, [0, j]]))
#                         x_rollback = np.delete(x_rollback, j, 1)
#                         print(regressor_OLS.summary())
#                         return x_rollback
#                     else:
#                         continue
#     regressor_OLS.summary()
#     return x
#
#
# SL = 0.05
# X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# X_Modeled = backwardElimination(X_opt, SL)