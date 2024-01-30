#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:33:15 2022

@author: ananthu2014
"""

'''(a ) Plot the sigmoid function. Print your interpretation on why 
this function is useful for a classification problem.

    (b) Plot the log functions in the cost function individually. 
    Print your interpretation of the log functions

     c) Using your own data for a single feature problem, and 
assuming linear regression problem, plot the cost function and 
the corresponding contours. Also, using cross entropy as the cost
 function, plot it as well as its contours. '''





import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def sigmoid(z):
    return (1/(1+math.exp(-z)))

z = np.linspace(-10,10,100)
#print(z)
out=np.zeros(len(z))
for i in range(len(z)):
    out[i] = sigmoid(z[i])

plt.plot(z,out)
plt.xlabel("z")
plt.ylabel("sigma(z)")
plt.title("Sigmoid Function")
plt.show()
         
'''In order to map predicted values to probabilities, we use 
the Sigmoid function. The function maps any real value into 
another value between 0 and 1. In machine learning, we use
 sigmoid to map predictions to probabilities. A sigmoid function 
 placed as the last layer of a machine learning model can serve
 to convert the model's output into a probability score, which
 can be easier to work with and interpret.'''
 
 
def compute_cost_1(x):
    return -1 * math.log(sigmoid(x))

def compute_cost_2(x):
    return -1 * math.log(1-sigmoid(x))



z = np.linspace(-10,10,100)
#print(z)
cost1=np.zeros(len(z))
for i in range(len(z)):
    cost1[i] = compute_cost_1(z[i])
 
plt.plot(out,cost1)
plt.xlabel("values")
plt.ylabel("-log(sigmoid)")
plt.title("Cross Entropy Loss Function - ( -log )")

plt.show()

z = np.linspace(-10,10,100)
#print(z)
cost2=np.zeros(len(z))
for i in range(len(z)):
    cost2[i] = compute_cost_2(z[i])
 
plt.plot(out,cost2)
plt.xlabel("values")
plt.ylabel("-log(1-sigmoid(z)")
plt.title("Cross )Entropy Loss Function - ( 1-log(sigmoid))")

plt.show()

''' For first cost function (-log(h_w(x))), if y  = 1, the cost function
will be 0 if the predicted value(h_w(x)) is 1. If the cost function is
penalized with large cost, then h_w(x) = 0

For the second cost function (-log(1-h_w(x))), if y = 0, the cost function
will be 0 if the predicted value(h_w(x)) is 0. If the cost function
 is penalized with large cost, then h_w(x) = 1 '''
 
 
# df=pd.read_csv("univariate_linear_regression.csv")
# df=np.array(df)
# x=df[:,0:df.shape[1]-1]
# y=df[:,-1]
def linear_regression_cost(x,y):
     x_data=np.ones((x.shape[0],x.shape[1]+1))
     x_data[:,1:]=x
     w=np.random.randint(1,10,size=x.shape[1]+1)
     gradw=np.zeros(len(w))
     loss_list=[]
     for i in range(5000):
         y_pred=x_data@w
         loss=(np.sum((y_pred-y)**2))/(2*x.shape[0])
         loss_list.append(loss)
         for j in range(len(gradw)):
             gradw[j]=np.dot((y_pred-y),x_data[:,j])/x.shape[0]
         w=w-(.001*gradw)
         
     return loss_list
x = np.array(np.random.randint(1,10,size=100))
x = x.reshape(100,1)
y = np.zeros(100)
y[50:] = 1

loss_var=linear_regression_cost(x,y)





plt.plot(loss_var)
plt.xlabel("No of iterations")
plt.ylabel("Loss value")
plt.title("Loss function vs Iteration")
plt.show()

def hypothesis(data,y,w0,w1):
    gen_data=np.ones((data.shape[0],data.shape[1]+1))
    gen_data[:,1:]=data
    w=np.array([w0,w1])
    step1=gen_data@np.array([w0,w1])
    step2=np.sum((y-step1)**2)/(2*data.shape[0])
    return step2
    

def hypothesis_cross_entropy(data,y,w0,w1):
    gen_data=np.ones((data.shape[0],data.shape[1]+1))
    gen_data[:,1:]=data
    w=np.array([w0,w1])
    step1=gen_data@np.array([w0,w1])
    step2=np.sum(-y*np.log(step1) - (1-y)*np.log(1-step1))/(2*data.shape[0])
    return step2
    



w0=np.linspace(-20,20,100)
w1=np.linspace(-10,10,100)
z=np.zeros((100,100))
W0,W1=np.meshgrid(w0,w1)
for i in range(len(w1)):
    for j in range(len(w0)):
        z[i,j]=hypothesis(x,y,w0[j],w1[i])
        



fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111,projection='3d')
ax.set_xlabel('W0',size=20)
ax.set_ylabel('W1',size=20)
ax.set_zlabel('J(W0,W2)',size=20)
ax.set_title('Surface_plot of loss function',size=40,color='b')
ax.plot_surface(W0,W1,z)

plt.show()

plt.figure(figsize=(10,10))
C=plt.contour(W0,W1,z,200)
plt.clabel(C,fontsize=10)


plt.show()

# Cross entropy cost function

w0=np.linspace(-20,20,1000)
w1=np.linspace(-10,10,1000)
z=np.zeros((1000,1000))
W0,W1=np.meshgrid(w0,w1)
for i in range(len(w1)):
    for j in range(len(w0)):
        z[i,j]=hypothesis_cross_entropy(x,y,w0[j],w1[i])
        



fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111,projection='3d')
ax.set_xlabel('W0',size=20)
ax.set_ylabel('W1',size=20)
ax.set_zlabel('J(W0,W2)',size=20)
ax.set_title('Surface_plot of loss function',size=40,color='b')
ax.plot_surface(W0,W1,z)

plt.show()

plt.figure(figsize=(10,10))
C=plt.contour(W0,W1,z,300)
plt.clabel(C,fontsize=10)


plt.show()

