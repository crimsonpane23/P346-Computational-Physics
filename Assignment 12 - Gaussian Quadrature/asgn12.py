'''
Name - Aryan Shrivastava
Roll no. - 2311041
Assignment 12 - Gaussian Quadrature
'''

#Solved upto accuracy of 9 decimal places in both questions

from mylibrary1 import *
import numpy as np

def f1(x):
    return x**2/(1+x**4) 

def f2(x):
    return np.sqrt(1+x**4)

#Question 1
print('Gaussian Quadrature on Ist Integral = ',GaussianQuad(f1,-1,1,14))

#Question 2
print('Gaussian Quadrature on IInd Integral = ',GaussianQuad(f2,0,1,8))
print('Simpsons method on IInd Integral = ',Simpson_integral(f2,0,1,24))

    




















