from mylibrary1 import *
import numpy as np

#Q1
def f(x):
    return (x - 5)*np.exp(x) + 5

def df(x):
    return (x-4)*np.exp(x)

print(Newton_Raphson(f,df,6))

#Q2
M = [[0,0,0,2],
     [0,0,3,0],
     [0,4,0,0],
     [5,0,0,0]]

print(GJInverse(M))
print(is_invertible(M))

#Q3
M2 = [[3,-7,-2,2],
      [-3,5,1,0],
      [6,-4,0,-5],
      [-9,5,-5,12]]
b2 = [[-9],
      [5],
      [7],
      [11]]

print(LUSolve(M2,b2))

def f2(x):
    return 4*np.exp(-x)*np.sin(x) - 1

print(Bisection(f2,0,1))
print(Regula_falsi(f2,0,1))

#Q: real gas
def f3(V):
    R = 0.0821
    T = 300
    a = 6.254
    b = 0.05422
    P = 5.95
    return (P + a/(V**2))*(V - b) - R*T

def df3(V):
    R = 0.0821
    T = 300
    a = 6.254
    b = 0.05422
    P = 5.95
    return P + a/(V**2) - (2*a*(V - b))/(V**3)

print(Newton_Raphson(f3, df3, 1))
