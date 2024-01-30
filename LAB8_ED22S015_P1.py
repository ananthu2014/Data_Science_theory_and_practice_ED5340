# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:10:38 2022

@author: anantHU2014
"""

#1)

import matplotlib.pyplot as plt
import numpy as np


x = np.arange(5.0, 15.0, 0.01)
y = np.arange(5.0, 15.0, 0.01)
X,Y=np.meshgrid(x,y)
Z = (X-10) **2 + (Y-10) ** 2
fig = plt.figure() #FIGURE PLOTTING
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z,color='yellow') #3D SURFACE PLOTTING

ax.set_xlabel('w1 values')
ax.set_ylabel('w2 values')
ax.set_zlabel('J values')
ax.set_title('3D DEMONSTRATION OF A MULTIVARIABLE FUNCTION')

plt.show()


cp  = plt.contour(x, y, Z)
plt.clabel(cp, fontsize=8)
ax.set_xlabel('w1 VALUES')
ax.set_ylabel('w2 VALUES')
ax.set_zlabel('J VALUES')
ax.set_title('3D DEMONSTRATION OF A MULTIVARIABLE FUNCTION')
plt.show()




