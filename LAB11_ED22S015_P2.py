#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 16:39:59 2022

@author:ananthu2014
"""

''' Logistic regression. Using the data provided 
(Logistic_regression_ls.csv), plot the decision boundary 
(linear) using Optimization of the sigmoid function 
(as discussed in the class).'''

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def sigmoid(z):
     val=np.exp(-z)
     return 1/(1+val)

def training(X,y):
    print(X.shape)
    w=np.zeros((X.shape[1],1))
    print(w)
    b=0
    for i in range(10000):
        yhat=np.dot(X,w)+b
        h=sigmoid(yhat)
        loss=(-1/X.shape[0])*(np.sum(y*np.log10(h)+(1-y)*np.log10(1-h)))
        dw=np.dot(X.T,h-y)*(1/X.shape[0])
        db=np.sum(h-y)*(1/X.shape[0])
        w=w-(0.01*dw)
        b=b-(0.01*db)
        if i%1000==0:
            print(f"The loss after {i} iteration is {loss}")
    return w,b

def predict(X,w,b):
    yhat=sigmoid(np.dot(X,w)+b)
    yhat_class=yhat>.5
    return yhat_class

data=pd.read_csv(f"Logistic_regression_ls.csv")
df=np.array(data)
x=df[:,0:2]
y=df[:,-1]
y=y[:,np.newaxis]

w,b=training(x,y)
print(w)
print(b)
y_hat=predict(x,w,b)
accuracy=np.sum(y_hat==y)/x.shape[0]
print()

plt.scatter(x[:,0],x[:,1],s=50,c=y,marker='^')
c=-(b/w[1])
m=-(w[0]/w[1])
new_x=np.array([x[:,0].min(),x[:,1].max()])
y1=m*new_x+c
plt.plot(new_x,y1,'r')
