#import numpy as pydataPath 
#import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn import preprocessing
#from sklearn import datasets
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.pipeline import make_pipeline
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import mean_squared_error, r2_score
#from sklearn import tree
#from sklearn.tree import DecisionTreeClassifier

import pandas as pd 

import csv
digits = pd.read_csv("classification/zoo.data", header=None)
print(digits)
