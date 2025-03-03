# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for Gradient Design.
2. Upload the dataset and check any null value using .isnull() function.
3. Declare the default values for linear regression.
4. Calculate the loss usinng Mean Square Error.
5. Predict the value of y.
6. Plot the graph respect to hours and scores using scatter plot function.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 212224040109
RegisterNumber: HARSHINI.V
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/content/studentscores.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
m=lr.coef_
m[0]
b=lr.intercept_
b
```
## Output:
## df.head():
![Output](SS1.jpg)
## df.tail():
![Output](SS2.jpg)
## x values:
![Output](SS3.jpg)
## y values:
![Output](SS4.jpg)
## y_pred:
![Output](SS5.jpg)
## y_test:
![Output](SS6.jpg)
## Graph of training data:
![Output](SS7.jpg)
## Graph of test data:
![Output](SS8.jpg)
## Values of MSE, MAE, RMSE:
![Output](SS9.jpg)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
