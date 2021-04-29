
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, f1_score, hinge_loss

realEstate = pd.read_csv("realEstate.csv")
zooData = pd.read_csv("zoo.data")
student_por = pd.read_csv("student-por.csv")
student_mat = pd.read_csv("student-mat.csv")


# X = realEstate[['X2 house age']]
# y = realEstate[['Y house price of unit area']]

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
# regressor = LinearRegression().fit(X_train, y_train)

# regressor.fit(X_train,y_train)
# prediction = regressor.predict(X_test)

# r2 = metrics.r2_score(y_test, prediction)
# meanAbsError = metrics.mean_absolute_error(y_test, prediction)
# meanSqrError = metrics.mean_squared_error(y_test, prediction)

# print("R2: ", r2)
# print("Mean Absolute Error: ", meanAbsError)
# print("Mean Square Error: ", meanSqrError)
#------------------

# X = zooData[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','tail','domestic','catsize']]
# y = zooData[['type']]

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
# regressor = LinearRegression().fit(X_train, y_train)

# regressor.fit(X_train,y_train)
# prediction = regressor.predict(X_test)

# r2 = metrics.r2_score(y_test, prediction)
# meanAbsError = metrics.mean_absolute_error(y_test, prediction)
# meanSqrError = metrics.mean_squared_error(y_test, prediction)

# print("R2: ", r2)
# print("Mean Absolute Error: ", meanAbsError)
# print("Mean Square Error: ", meanSqrError)

#--------------------

X = student_por[['Medu', 'Fedu', 'traveltime','studytime','failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']]
y = student_por[['G3']]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
regressor = LinearRegression().fit(X_train, y_train)

regressor.fit(X_train,y_train)
prediction = regressor.predict(X_test)

r2 = metrics.r2_score(y_test, prediction)
meanAbsError = metrics.mean_absolute_error(y_test, prediction)
meanSqrError = metrics.mean_squared_error(y_test, prediction)

print("R2: ", r2)
print("Mean Absolute Error: ", meanAbsError)
print("Mean Square Error: ", meanSqrError)

#------------

# X = student_mat[['Medu', 'Fedu', 'traveltime','studytime','failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']]
# y = student_mat[['G3']]

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
# regressor = LinearRegression().fit(X_train, y_train)

# regressor.fit(X_train,y_train)
# prediction = regressor.predict(X_test)

# r2 = metrics.r2_score(y_test, prediction)
# meanAbsError = metrics.mean_absolute_error(y_test, prediction)
# meanSqrError = metrics.mean_squared_error(y_test, prediction)

# print("R2: ", r2)
# print("Mean Absolute Error: ", meanAbsError)
# print("Mean Square Error: ", meanSqrError)
