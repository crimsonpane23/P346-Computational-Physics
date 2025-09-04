'''
Name - Aryan Shrivastava
Roll no. - 2311041
Assignment 7 - Root Finding methods: Fixed Point and Newton-Raphson method
'''

from mylibrary1 import *
import math

#-------------------------------Question 1-----------------------#
def f1(x):
    return 3*x + math.sin(x) - math.exp(x)

def df1(x):
    return 3 + math.cos(x) - math.exp(x)

#Choosing initial guess = 0 to converge towards the root in the given interval for Newton-Raphson method
print('Root approximated using Newton-Raphson method and the number of iterations taken are:',Newton_Raphson(f1, df1, 0))
print('Root approximated using Regula-Falsi method and the number of iterations taken are:',Regula_falsi(f1, -1.5, 1.5 ))
print('Root approximated using Bisection method and the number of iterations taken are:',Bisection(f1, -1.5, 1.5))
print('Overall Newton-Raphson has the highest rate of convergence')
#Here, Bisection method and Regula falsi give same convergence rate because in the bracket [a,b], a is never updated in this case, only b gets closer and closer to the root and convergence is achieved when f(c) == 0 (Rounded off by float dtype)

#--------------------------------------Question 2-------------------------------#

def f2(x):
    return x**2 - 2*x - 3

def g2(x):
    return (2 + 3/x)

def g2_new(x):
    return 3/(x-2)

#There are two roots (-1) and (+3), so I chose different functions g(x) and initial guesses to converge towards those roots
print('Root converged when g2 = (2 + 3/x) and initial guess = 2:',Fixed_point_root_find(f2, g2, 2))
print('Root converged when g2 = 3/(x-2) and initial guess = 0:',Fixed_point_root_find(f2, g2_new, 0))
    
#-------------------------------------------Output----------------------------------#
'''
Root approximated using Newton-Raphson method and the number of iterations taken are: (0.36042168047601975, 3)
Root approximated using Regula-Falsi method and the number of iterations taken are: (0.36042170296032444, 23)
Root approximated using Bisection method and the number of iterations taken are: (0.36042165756225586, 23)
Overall Newton-Raphson has the highest rate of convergence
Root converged when g2 = (2 + 3/x) and initial guess = 2: 2.999999721233142
Root converged when g2 = 3/(x-2) and initial guess = 0: -0.999999721233142
'''













