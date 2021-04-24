import pandas as pd
import numpy as np
#from sklearn import datasets
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Path of the file to read
myData = pd.read_csv(r'C:\Users\victo\Downloads\RealEstDataSet.csv', header=1)
myData = myData.rename(columns=['number','X1','houseAge','X3','X4','X5','X6','housePrice'], inplace = False)
#print(myData.shape)
#myData.shape  #414, 8
#myData.describe
#myData.plot(X='X2 house age', y='Y house price of unit area', style='o')
#print(myData.feature_names)
#pull in more datasets and label them to use
X = myData[['houseAge']]
#X = myData[3]
y = myData[['housePrice']]
#print(X)
#print(y)
#X = myData.iloc[:,:-1].values  #independent variable array
#y = myData.iloc[:,8].values  #dependent variable vector
#print(y)
#split the data set into 1/3 for test and 2/3 for training
# from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y.target,test_size=.2, train_size=1) 

#from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
model.score(X_test, y_test)

#regressor.fit(X_train,y_train)

#plt.scatter(X.T[3], y_train)
#plt.show
#plt.scatter(X_train, y_train,  color='black')
#plt.plot(X_train, y_train, color='blue', linewidth=3)

#plt.xticks(())
#plt.yticks(())

#plt.show()