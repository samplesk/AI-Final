import pandas as pd
import numpy as np
#from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Path of the file to read
myData = pd.read_csv(r'C:\Users\victo\Downloads\RealEstDataSet.csv', usecols=[3,6], header=1) 


#x = myData[3]
#y = myData[6]


#pull in more datasets and label them to use
X = myData.iloc[:,:-1].values  #independent variable array
y = myData.iloc[:,1].values  #dependent variable vector

#split the data set into 1/3 for test and 2/3 for training
# from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0) 

#from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)