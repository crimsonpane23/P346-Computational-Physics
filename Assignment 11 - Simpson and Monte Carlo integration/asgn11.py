'''
Name - Aryan Shrivastava
Roll no. - 2311041
Assignment - 11: Monte Carlo and Simpson Integration
'''

from mylibrary1 import *
import numpy as np
import math
import matplotlib.pyplot as plt

def f1(x):
    return (np.sin(x))**2 

def f2(x):
    return 1/x

def f3(x):
    return x*np.cos(x)

print("-------------Question 1 - integral 1 ------------")
print("Simpson result = ", Simpson_integral(f2,1,2,20))             #N = 20
print("Midpoint result = ",Midpoint_int(f2,1,2,289))                 #N = 289

print("")

print("--------------Question 1 - integral 2 ------------")
print("Simpson result = ",Simpson_integral(f3,0,(math.pi)/2,22))   #N = 22
print("Midpoint result = ",Midpoint_int(f3,0,(math.pi)/2,610))       #N = 610

print("")

print("--------------Question - 2 --------------")
print("Monte Carlo result = ",Monte_Carlo_integral(f1,-1,1,70000))    #N = 70000


#------------------------Output-------------------------#
'''
-------------Question 1 - integral 1 ------------
Simpson result =  0.6931473746651161
Midpoint result =  0.6931468064035272

--------------Question 1 - integral 2 ------------
Simpson result =  0.570796987316687
Midpoint result =  0.5707970370864707

--------------Question - 2 --------------
Monte Carlo result =  0.5450239255093544
'''
    
    














