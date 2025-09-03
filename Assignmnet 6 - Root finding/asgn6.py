'''
Name - Aryan Shrivastava
Roll no. - 2311041
Assignment 6 - Bisection and Regula Falsi
'''

from mylibrary1 import *
import math

'''
I have choosen epsilon = delta = 10^(-6) and beta = 0.5 by default.
But they are taken as an input from the user in the defined function, so they can  be modified for later use as required.
'''

#--------------Question 1--------------#

def f1(x):
    return (math.log(x/2) - math.sin(5*x/2))

print('Approximated Root and function value at the approximated root of given function f1 by Bisection method is:',Bisection(f1, 1.5, 3))
print('')
print('Approximated Root and function value at the approximated root of given function f1 by Regula Falsi method is:',Regula_falsi(f1, 1.5, 3))

#--------------Question 2---------------#

def f2(x):
    return (-x - math.cos(x))

print('')
print('Brackets adjusted for the function f2 and given interval [2,4] to obtain the root inside the brackets are:',bracket_adjust(f2, 2, 4))


#--------Output---------------#
'''
Approximated Root and function value at the approximated root of given function f1 by Bisection method is: (2.623140335083008, 7.154571934897547e-10)

Approximated Root and function value at the approximated root of given function f1 by Regula Falsi method is: (2.6231403354363083, 2.7755575615628914e-16)

Brackets adjusted for the function f2 and given interval [2,4] to obtain the root inside the brackets are: (-2.75, 0.625)
'''

    











