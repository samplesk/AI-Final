import pandas as pd
import numpy as np
#from sklearn import datasets
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Path of the file to read
myData = pd.read_csv('../classification/realEstate.csv') # Switched to using a relative file path, and did not drop the header names
# I'm not sure if this is the same file. At the very least, you seem to have changed the names of the columns. You should really
# be accessing a file inside of your repo, not somewhere else where the code can't find it.
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
X = myData[['X2 house age']] # The column name in the CSV I found was "X2 house age"
#### below, I was thinking maybe I could just grab the 3rd column and it work.
#X = myData[3]
y = myData[['Y house price of unit area']] # Another difference in the name of the column
###I tried to just print them after trying
#print(X)
#print(y)

#split the data set into 1/3 for test and 2/3 for training
# from sklearn.model_selection import train_test_split
### this is the test split I have tried doing,It should take my X and Y and split it into a training set of X_train and Y_train
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3) 

#from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


plt.scatter(X_train, y_train)
### this should plot it
#plt.show
### I tried plotting it with just X-train and y_train
#plt.scatter(X_train, y_train,  color='black')
### tried plotting it
#plt.plot(X_train, y_train, color='blue', linewidth=3)

