#cd..\AI-Final
#cd classification

import numpy as pydataPath 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd 
import csv
import xlsxwriter

X = [[101, 0], [18, 0]]
Y = [101, 18]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

digits = pd.read_csv("zoo.data", header=None)
#digits = pd.read_csv("student-por.csv", header=1)
#digits = pd.read_excel("Real estate valuation data set.xlsx", header=0)
print(digits)
