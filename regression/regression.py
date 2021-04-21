import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Path of the file to read
myData = pd.read_csv(r'C:\Users\victo\Downloads\RealEstDataSet.csv', header=None) 
#, usecols=[4,7]

x = myData.data
y = myData.target

print("x")
print(x)
print(x.shape)
print("y")
print(y)
print(y.shape)
#pull in more datasets and label them to use
#X = data.iloc[:,:-1].values  #independent variable array
#y = data.iloc[:,1].values  #dependent variable vector

#split the data set into 1/3 for test and 2/3 for training
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0) 

#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#regressor.fit(X_train,y_train)