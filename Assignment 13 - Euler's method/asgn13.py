'''
Name - Aryan Shrivastava
Roll no. - 2311041
Assignment - 13: Euler's method and Predictor Corrector
'''

from mylibrary1 import *
import math
import matplotlib.pyplot as plt
import numpy as np

def f1(y,x):
    return y - x**2

def f2(y,x):
    return (x+y)**2

def y1(x):
    return x**2 + 2*x + 2 - 2*math.exp(x)

def y2(x):
    return math.tan(x+math.pi/4) - x

#For 1st Function
P, Q = forward_Euler(f1, 0, 2, 0.1, 0, 0)
S = np.linspace(0, 2, 100)

plt.plot(Q, P, label="Euler's Method Approximation")
plt.plot(S, [y1(x) for x in S], label="Analytic Solution", linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Euler's Method vs Analytic Solution for dy/dx = y - x^2")
plt.legend()
plt.grid()
plt.show()

#For 2nd Function
P1, Q1 = forward_Euler(f2, 0, math.pi/5, 0.1, 0, 1)
S1 = np.linspace(0, math.pi/5, 100)

plt.plot(Q1, P1, label="Euler's Method Approximation")
plt.plot(S1,[y2(x) for x in S1], label="Analytic Solution", linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Euler's Method vs Analytic Solution for dy/dx = (x+y)^2")
plt.legend()
plt.grid()
plt.show()



#For 1st Function
P2, Q2 = ODE_Predictor_corrector(f1, 0, 2, 0.1, 0, 0)
S = np.linspace(0, 2, 100)

plt.plot(Q2, P2, label="Predictor-Corrector solution  Approximation")
plt.plot(S, [y1(x) for x in S], label="Analytic Solution", linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Predictor-Corrector solution vs Analytic Solution for dy/dx = y - x^2")
plt.legend()
plt.grid()
plt.show()

#For 2nd Function
P3, Q3 = ODE_Predictor_corrector(f2, 0, math.pi/5, 0.1, 0, 1)
S1 = np.linspace(0, math.pi/5, 100)

plt.plot(Q3, P3, label="Predictor-Corrector Approximation")
plt.plot(S1,[y2(x) for x in S1], label="Analytic Solution", linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Predictor-Corrector solution vs Analytic Solution for dy/dx = (x+y)^2")
plt.legend()
plt.grid()
plt.show()

#Comparing Euler's method and Predictor-Corrector
#1st Function
plt.plot(Q, P, label="Euler's Method Approximation")
plt.plot(Q2, P2, label="Predictor-Corrector solution  Approximation")
plt.plot(S, [y1(x) for x in S], label="Analytic Solution", linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Comparing methods for dy/dx = y - x^2")
plt.legend()
plt.grid()
plt.show()

#2nd Function
plt.plot(Q1, P1, label="Euler's Method Approximation")
plt.plot(Q3, P3, label="Predictor-Corrector Approximation")
plt.plot(S1, [y2(x) for x in S1], label="Analytic Solution", linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Comparing methods for dy/dx = y - x^2")
plt.legend()
plt.grid()
plt.show()




