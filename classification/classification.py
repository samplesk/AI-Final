import numpy as np
import pandas as pd
from sklearn import neighbors, metrics, datasets, preprocessing, tree, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, f1_score, hinge_loss
from sklearn.tree import DecisionTreeClassifier
import csv
import xlsxwriter

zooData = pd.read_csv("zoo.data")
student_por = pd.read_csv("student-por.csv")
student_mat = pd.read_csv("student-mat.csv")
realEstate = pd.read_csv("realEstate.csv")

#---------------------K Nearest Neighbors Classification
#zoo Data
# X = zooData[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','tail','domestic','catsize']]
# y = zooData[['type']]

# knn = neighbors.KNeighborsClassifier(n_neighbors = 25, weights = 'distance')
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# knn.fit(X_train,y_train)
# prediction = knn.predict(X_test)
# pred_decision = knn.predict_proba(X_test)

# accuracy = metrics.accuracy_score(y_test, prediction)
# f1Score_macro = metrics.f1_score(y_test, prediction, average = 'macro')
# f1Score_micro = metrics.f1_score(y_test, prediction, average = 'micro')
# f1Score_weighted = metrics.f1_score(y_test, prediction, average = 'weighted')
# hingeLoss = metrics.hinge_loss(y_test, pred_decision)

# print("accuracy: ", accuracy)
# print("f1Score_macro: ", f1Score_macro)
# print("f1Score_micro: ", f1Score_micro)
# print("f1Score_weighted: ", f1Score_weighted)
# print("hingeLoss: ", hingeLoss)

#---------------------
###---student Portuguese 

# #X = student_por[['Medu', 'Fedu', 'traveltime','studytime','failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']]
# ### with G1 & G2, better predictions
X = student_por[['Medu', 'Fedu', 'traveltime','studytime','failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']]

y = student_por[['G3']]

knn = neighbors.KNeighborsClassifier(n_neighbors = 25, weights = 'distance')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

knn.fit(X_train,y_train)
prediction = knn.predict(X_test)
pred_decision = knn.predict_proba(X_test)

accuracy = metrics.accuracy_score(y_test, prediction)
f1Score_macro = metrics.f1_score(y_test, prediction, average = 'macro')
f1Score_micro = metrics.f1_score(y_test, prediction, average = 'micro')
f1Score_weighted = metrics.f1_score(y_test, prediction, average = 'weighted')
hingeLoss = metrics.hinge_loss(y_test, pred_decision)

print("accuracy: ", accuracy)
print("f1Score_macro: ", f1Score_macro)
print("f1Score_micro: ", f1Score_micro)
print("f1Score_weighted: ", f1Score_weighted)
print("hingeLoss: ", hingeLoss)

###----------------------math

# #X = student_mat[['Medu', 'Fedu', 'traveltime','studytime','failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']]

# # ### with G1 & G2, better predictions
# X = student_mat[['Medu', 'Fedu', 'traveltime','studytime','failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']]
# y = student_mat[['G3']]

# knn = neighbors.KNeighborsClassifier(n_neighbors = 25, weights = 'distance')
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# knn.fit(X_train,y_train)
# prediction = knn.predict(X_test)
# pred_decision = knn.predict_proba(X_test)

# accuracy = metrics.accuracy_score(y_test, prediction)
# f1Score_macro = metrics.f1_score(y_test, prediction, average = 'macro')
# f1Score_micro = metrics.f1_score(y_test, prediction, average = 'micro')
# f1Score_weighted = metrics.f1_score(y_test, prediction, average = 'weighted')
# hingeLoss = metrics.hinge_loss(y_test, pred_decision)

# print("accuracy: ", accuracy)
# print("f1Score_macro: ", f1Score_macro)
# print("f1Score_micro: ", f1Score_micro)
# print("f1Score_weighted: ", f1Score_weighted)
# print("hingeLoss: ", hingeLoss)

#---------------------realEstate

# X = realEstate[['X2 house age','X3 distance to the nearest MRT station']]
# y = realEstate[['Y house price of unit area']]

# knn = neighbors.KNeighborsClassifier(n_neighbors = 25, weights = 'distance')
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# knn.fit(X_train,y_train)

# prediction = knn.predict(X_test)
# accuracy = metrics.accuracy_score(y_test, prediction)

# print("prediction: ", prediction)
# print("accuracy: ", accuracy)