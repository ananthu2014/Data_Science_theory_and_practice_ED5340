# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:29:06 2022

@author: ananthu2014
"""

#a)
def j(w):
    if w != 0:
        return (w**2+(54/w))

a = 1
b = 10
n = 10
x = (b-a)/n
print(j(2))
print('Initial value=', a)
print('Final value=', b)
print('No.of intermediate steps=', n)
print('Stepsize=', x)

for i in range(0, n+1):
    if j(a+i*x) >= j(a+(i+1)*x) and j(a+(i+1)*x) <= j(a+(i+2)*x):
        print(f"min between{a + (i*x)} and {a+(i+2)*x}")
        x1= a + (i*x)
        x2 = a + (i+2)*x
        break
    elif a+(i+2)*x <= b:
        continue
    else:
        print('there is no minimum between given intervals')
print(x1)
print(x2)

#b)
a=x1
b=x2
n=10**(-3)
wm=(a+b)/2
w1= a+((b-a)/4)
w2=b-((b-a)/4)  
while abs(b-a) < n:
  if j(w1) < j(wm):
    b=wm
    wm=w1
  elif j(w2)<j(wm):
      a=wm
      wm=w2
  else:
      a=w1
      b=w2
print('The value obtained using interval halving is:',wm)      
    
#c)
def j1(w):
    return 2*w-(54/w**2)

def j2(w):
    return 2 + (108/w**3)

x1=wm
for k in range(1,10):
    temp=x1-(j1(x1)/j2(x1))
    if (j1(temp) < 10**-6):
        break
    else:
        x1=temp
    
print('The value after newton raphson method is:',temp)    
    
    


   














