import pandas as pd
import numpy as np
#from sklearn import datasets
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Path of the file to read
myData = pd.read_csv(r'C:\Users\victo\Downloads\RealEstDataSet.csv', header=1)
#this below doesn't work, I just thought maybe if I could rename the colums then my X=[['houseAge']] would work
#myData = myData.rename(columns=['number','X1','houseAge','X3','X4','X5','X6','housePrice'], inplace = False)
print(myData.shape)
#myData.shape  #414, 8
#myData.describe
###tried this below and it did not work, got it from an example, but it should just plot x and y I chose with 'o' style 
#myData.plot(X='X2 house age', y='Y house price of unit area', style='o')
#### doesn't work, I was trying to make sure that my names were right
#print(myData.feature_names)
#### doesn't work, don't know why
X = myData[['houseAge']]
#### below, I was thinking maybe I could just grab the 3rd column and it work.
#X = myData[3]
y = myData[['housePrice']]
###I tried to just print them after trying
#print(X)
#print(y)

#split the data set into 1/3 for test and 2/3 for training
# from sklearn.model_selection import train_test_split
### this is the test split I have tried doing,It should take my X and Y and split it into a training set of X_train and Y_train
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3, train_size=1) 

#from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
###this should just output a score, but doesn't
model.score(X_test, y_test)

###this is where I tried to just get a visual of it, it should take the 3rd column and y_train and plot it
#plt.scatter(X.T[3], y_train)
### this should plot it
#plt.show
### I tried plotting it with just X-train and y_train
#plt.scatter(X_train, y_train,  color='black')
### tried plotting it
#plt.plot(X_train, y_train, color='blue', linewidth=3)

