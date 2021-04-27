import pandas as pd
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, f1_score, hinge_loss

# Path of the file to 
"""realEstate = pd.read_csv('../classification/realEstate.csv')
X = realEstate[['X2 house age']]
y = realEstate[['Y house price of unit area']]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
regressor = LinearRegression().fit(X_train, y_train)
plt.scatter(X_train, y_train)
plt.show
plt.scatter(X_train, y_train,  color='black')
plt.plot(X_train, y_train, color='blue', linewidth=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
regressor.fit(X_train,y_train)
prediction = regressor.predict(X_test)
mae = metrics.mean_absolute_error(y_test, prediction)
mse = metrics.mean_squared_error(y_test, prediction)
r2 = metrics.r2_score(y_test, prediction)

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))"""

"""studentData = pd.read_csv('../classification/student-mat.csv')
X = studentData[['absences']]
y = studentData[['health']]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
regressor = LinearRegression().fit(X_train, y_train)
plt.scatter(X_train, y_train)
plt.show
plt.scatter(X_train, y_train,  color='black')
plt.plot(X_train, y_train, color='blue', linewidth=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
regressor.fit(X_train,y_train)
prediction = regressor.predict(X_test)
r2 = metrics.r2_score(y_test, prediction)
maxError = metrics.max_error(y_test, prediction)
print("r2", r2)
print("max error", maxError) 
mae = metrics.mean_absolute_error(y_test, prediction)
mse = metrics.mean_squared_error(y_test, prediction)
r2 = metrics.r2_score(y_test, prediction)

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))"""

zooData = pd.read_csv('../classification/zoo.data')
X = zooData[['hair']]
y = zooData[['type']]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
regressor = LinearRegression().fit(X_train, y_train)
plt.scatter(X_train, y_train)
plt.show
plt.scatter(X_train, y_train,  color='black')
plt.plot(X_train, y_train, color='blue', linewidth=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
regressor.fit(X_train,y_train)
prediction = regressor.predict(X_test)
r2 = metrics.r2_score(y_test, prediction)
maxError = metrics.max_error(y_test, prediction)
print("r2", r2)
print("max error", maxError) 
mae = metrics.mean_absolute_error(y_test, prediction)
mse = metrics.mean_squared_error(y_test, prediction)
r2 = metrics.r2_score(y_test, prediction)

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))
