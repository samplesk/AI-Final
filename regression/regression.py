import pandas as pd
import numpy as np
#from sklearn import datasets
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Path of the file to read
myData = pd.read_csv('../classification/realEstate.csv')

X = myData[['X2 house age']]
y = myData[['Y house price of unit area']]

#split the data set into 1/3 for test and 2/3 for training
# from sklearn.model_selection import train_test_split
### this is the test split I have tried doing,It should take my X and Y and split it into a training set of X_train and Y_train
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3) 

#from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
###this should just output a score, but doesn't
#linear_model.score(X_test, y_test)

###this is where I tried to just get a visual of it, it should take the 3rd column and y_train and plot it
plt.scatter(X_train, y_train)
### this should plot it
#plt.show
### I tried plotting it with just X-train and y_train
#plt.scatter(X_train, y_train,  color='black')
### tried plotting it
#plt.plot(X_train, y_train, color='blue', linewidth=3)

