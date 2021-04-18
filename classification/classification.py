import numpy as np
import pandas as pd 
from sklearn import neighbors, metrics, datasets, preprocessing, tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
import csv
import xlsxwriter

zooData = pd.read_csv("zoo.data")
student = pd.read_csv("student-por.csv")
realEstate = pd.read_csv("realEstate.csv")

#---------------------K Nearest Neighbors Classification
# zoo Data
# X = zooData[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','tail','domestic','catsize']]
# y = zooData[['type']]

# knn = neighbors.KNeighborsClassifier(n_neighbors = 25, weights = 'distance')
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# knn.fit(X_train,y_train)

# prediction = knn.predict(X_test)

# accuracy = metrics.accuracy_score(y_test, prediction)

# print("prediction: ", prediction)
# print("accuracy: ", accuracy)

#---------------------
# #student
X = student[['Medu', 'Fedu', 'traveltime','studytime','failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']]
y = student[['G3']]

knn = neighbors.KNeighborsClassifier(n_neighbors = 25, weights = 'distance')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

knn.fit(X_train,y_train)

prediction = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, prediction)

print("prediction: ", prediction)
print("accuracy: ", accuracy)

#---------------------
#realEstate
# X = realEstate[['X2 house age','X3 distance to the nearest MRT station']]
# y = realEstate[['Y house price of unit area']]

# knn = neighbors.KNeighborsClassifier(n_neighbors = 25, weights = 'distance')
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# knn.fit(X_train,y_train)

# prediction = knn.predict(X_test)
# accuracy = metrics.accuracy_score(y_test, prediction)

# print("prediction: ", prediction)
# print("accuracy: ", accuracy)