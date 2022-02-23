'''
Assumptions:
	1. Data file is in csv format
	2. Last column is target column and remaining columns are featured columns
'''

import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
def model_SLR(data_file):
    lr_details = {}
    dataset = pd.read_csv(data_file)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    correlation = dataset.corr().iloc[:, -1]
    for i, column in enumerate(list(dataset.columns)[:-1]):
        X_train_lm , X_test_lm, y_train_lm, y_test_lm = train_test_split(pd.Series(X[:, i]), y, train_size=0.7, random_state=0)
        X_train_lm = X_train_lm.values.reshape(-1,1)
        X_test_lm = X_test_lm.values.reshape(-1,1)
        lm = LinearRegression()
        lm.fit(X_train_lm, y_train_lm)
        y_test_pred = lm.predict(X_test_lm)
        y_train_pred = lm.predict(X_train_lm)
        r2_train = r2_score(y_train_lm, y_train_pred)
        r2_test = r2_score(y_test_lm, y_test_pred)
        lr_details[column] = {'Correlation': round(correlation[column],3), 'Coefficient': round(lm.coef_[0], 3), 'Intercept': round(lm.intercept_, 3), 'R-Square_train': round(r2_train, 3), 'R-Square_test': round(r2_test, 3)}
    return lr_details
		
if __name__ = "__main__":
    model_SLR('Salary_Data.csv')
	'''
	Output:
	
	{'YearsExperience': {'Correlation': 0.978,
  'Coefficient': 9360.261,
  'Intercept': 26777.391,
  'R-Square_train': 0.942,
  'R-Square_test': 0.974}}
	'''
