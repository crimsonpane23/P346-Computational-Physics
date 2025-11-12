'''
Name - Aryan Shrivastava
Roll no. - 2311041
Assignment - 16: Data Fitting and Interpolation
'''

from mylibrary1 import *
import math
import numpy as np

X = [2,3,5,8,12]
Y = [10,15,25,40,60]
print('Interpolated y(6.7) =',interpolate_lagrange(X,Y,6.7)) 

X = [2.5,3.5,5.0,6.0,7.5,10.0,12.5,15.0,17.5,20.5]
Y = [13.0,11.0,8.5,8.2,7.0,6.2,5.2,4.8,4.6,4.3]
X1 = np.log(np.array(X))
Y1 = np.log(np.array(Y))
X2 = np.array(X)
Y2 = np.log(np.array(Y))

#Question 2
a1, sigma_a1, a2, sigma_a2, r_2 = leastsquareslinearfit(X1,Y1)
print("For Power Law Fit:")
print("a (log a) =", a1, "+-", sigma_a1)
print("b =", a2, "+-", sigma_a2)
print("Goodness of fit (r^2) =", r_2)

b1, sigma_b1, b2, sigma_b2, r_2 = leastsquareslinearfit(X2,Y2)
print("\nFor Exponential Fit:")
print("a (log a) =", b1, "+-", sigma_b1)
print("b =", b2, "+-", sigma_b2)
print("Goodness of fit (r^2) =", r_2)

print('Power law gives a better Goodness of fit as compared to Exponential fit.')
#Output
'''
Interpolated y(6.7) = 33.49999999999999
For Power Law Fit:
a (log a) = 3.0467272510281007 +- 1.0413062848072583
b = -0.53740930145056 +- 0.472348893379857
Goodness of fit (r^2) = 0.7750435352872259

For Exponential Fit:
a (log a) = 2.5025003706646873 +- 0.6253965263241253
b = -0.05845553447818332 +- 0.05395561278850259
Goodness of fit (r^2) = 0.5762426888065756
Power law gives a better Goodness of fit as compared to Exponential fit.
'''
















