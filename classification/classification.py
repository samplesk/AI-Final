import numpy as pydataPath 
import numpy as np
import pandas as pd 
#from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets
#from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


import csv
digits = pd.read_csv("data.csv", header=None)
print(digits)
