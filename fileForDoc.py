
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import RandomizedSearchCV

import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from math import sqrt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
#from mlxtend.regressor import StackingRegressor

from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib


df = pd.read_excel('Data_Train.xlsx')

df.sort_values('Date_of_Journey')

#print(df)


df['year'] = pd.DatetimeIndex(df['Date_of_Journey']).year
df['month'] = pd.DatetimeIndex(df['Date_of_Journey']).month
df['Day'] = pd.DatetimeIndex(df['Date_of_Journey']).day

#print(df)


df['Airline'].replace(['Trujet', 'Vistara Premium economy'], 'Another')


#print(df)

def flight_dep_time(X):
    '''
    This function takes the flight Departure time
    and convert into appropriate format.
    '''
    if int(X[:2]) >= 0 and int(X[:2]) < 6:
        return 'mid_night'
    elif int(X[:2]) >= 6 and int(X[:2]) < 12:
        return 'morning'
    elif int(X[:2]) >= 12 and int(X[:2]) < 18:
        return 'afternoon'
    elif int(X[:2]) >= 18 and int(X[:2]) < 24:
        return 'evening'

df['flight_time'] = df['Dep_Time'].apply(flight_dep_time)

df[df['Total_Stops'].isnull()]
df.dropna(axis = 0)


def convert_into_stops(X):
    if X == '4 stops':
        return 4
    elif X == '3 stops':
        return 3
    elif X == '2 stops':
        return 2
    elif X == '1 stop':
        return 1
    elif X == 'non-stop':
        return 0


df['total_Stops'] = df['Total_Stops'].apply(convert_into_stops)
#print(df['total_Stops'])



def convert_into_seconds(X):
    '''
    This function takes the total time of flight from
    one city to another and converts it into the seconds.
    '''
    a = [int(s) for s in re.findall(r'-?\d+\.?\d*', X)]
    if len(a) == 2:
        hr = a[0] * 3600
        min = a[1] * 60
    else:
        hr = a[0] * 3600
        min = 0
    total = hr + min
    return total

df['Duration(sec)'] = df['Duration'].map(convert_into_seconds)

#print(df)

df.drop_duplicates()


df['Additional_Info'].unique()
df['Additional_Info'].replace('No Info', 'No info')
#print(df)

df.to_csv('C:/Users/Debargha Simlai/Documents/ineuron_CODE/DLCVNLP CODE/Python Prereq/Projects_DLCVNLP/Python project/Projects/Flight price prediction dataset EDA/Flight_Price/cleaned_data.csv', index = None)



df = pd.get_dummies(df, columns = ['Airline', 'Source', 'Destination', 'Additional_Info', 'flight_time'])

#print(df.head())
data= df.dropna(axis = 0)

dataframe= data.drop(['Date_of_Journey', 'Route', 'Dep_Time', 'Arrival_Time', 'Duration','Total_Stops'], axis = 1)
dataframe.to_csv('C:/Users/Debargha Simlai/Documents/ineuron_CODE/DLCVNLP CODE/Python Prereq/Projects_DLCVNLP/Python project/Projects/Flight price prediction dataset EDA/Flight_Price/final_data.csv', index = None)

y = dataframe['Price']
X = dataframe.drop('Price', axis = 1)

s = StandardScaler()
X = s.fit_transform(X)

# Splitting data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

'''
#baseline model

y_train_pred = np.ones(X_train.shape[0]) * y_train.mean()
y_test_pred = np.ones(X_test.shape[0]) * y_train.mean()


print("Train Results for Baseline Model:")
print(50 * '-')
print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-squared: ", r2_score(y_train.values, y_train_pred))


print("Test Results for Baseline Model:")
print(50 * '-')
print("Root mean squared error: ", sqrt(mse(y_test, y_test_pred)))
print("R-squared: ", r2_score(y_test, y_test_pred))

'''
'''
#KNN Regressor
k_range = list(range(1, 30))
params = dict(n_neighbors = k_range)
knn_regressor = GridSearchCV(KNeighborsRegressor(), params, cv = 10, scoring = 'neg_mean_squared_error')
knn_regressor.fit(X_train, y_train)

print(knn_regressor.best_estimator_)

y_train_pred =knn_regressor.predict(X_train) ##Predict train result
y_test_pred =knn_regressor.predict(X_test) ##Predict test result

print("Train Results for KNN Regressor Model:")
print(50 * '-')
print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-squared: ", r2_score(y_train.values, y_train_pred))

print("Test Results for KNN Regressor Model:")
print(50 * '-')
print("Root mean squared error: ", sqrt(mse(y_test, y_test_pred)))
print("R-squared: ", r2_score(y_test, y_test_pred))

'''


'''''
mode = XGBRegressor(reg_lambda = 0.001, n_estimators = 400, max_depth = 5, learning_rate = 0.05)
mode.fit(X_train, y_train)
prediction = mode.predict(X_test)

# open a file, where you ant to store the data
file = open('flight_rf.pkl', 'wb')

# dump information to that file
pickle.dump(mode, file)

model = open('flight_rf.pkl','rb')
#xg = pickle.load(model)

#y_prediction = xg.predict(X_test)


'''''


from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)

prediction = reg_rf.predict(X_test)

# open a file, where you ant to store the data
file = open('flight_rf.pkl', 'wb')

# dump information to that file
pickle.dump(reg_rf, file)

model = open('flight_rf.pkl','rb')
forest = pickle.load(model)






