'''
Name - Aryan Shrivastava
Roll no. - 2311041
Assignment 9 -Laguerre's method
'''
#Caution - Here n is not the degree of the polynomial
#List L = [a0, a1, a2, ...., an] represents the polynomial a0 + a1*x + a2*x^2 + ... + an*x^n
#So, degree of polynomial = len(L) - 1
#For example, L = [6,1,-7,-1,1] represents the polynomial 6 + x - 7x^2 - x^3 + x^4


#For all polynomials, initial guess x0 = 1, epsilon = 10**(-6)
#I tried using Laguerre_solve function to find all roots but it gave error after finding 2 roots
#So, I called Laguerre and deflation functions separately to find 2 roots of each polynomial
#The method doesnt converge after finding 2 roots for each polynomial
#Error was that the square root term becomes complex sometimes 
#Or the values just oscillate between a few values and dont converge

from mylibrary1 import *
import math

p1 = [6,1,-7,-1,1]
p2 = [4,0,-5,0,1]
p3 = [-4.5,13.5,0.5,-19.5,0,2]

itr = 0
def Laguerre(p, x0, epsilon = 10**(-6)):
    if polynomial_eval(p,x0) == 0:
        return x0
    global itr
    n = len(p)
    deg = poly_deg(p)
    dp = poly_derivative(p)
    d2p = poly_derivative(dp)
    G = polynomial_eval(dp,x0)/polynomial_eval(p,x0)
    H = G**2 - polynomial_eval(d2p,x0)/polynomial_eval(p,x0)
    '''
    if (deg-1)*(deg*H - G**2) < 0:
        print(x0, 'negative root')
        return x0
    '''
    denom1 = G + math.sqrt((deg-1)*(deg*H - G**2))
    denom2 = G - math.sqrt((deg-1)*(deg*H - G**2))
    if abs(denom1) >= abs(denom2):
        denom = denom1
    if abs(denom1) < abs(denom2):
        denom = denom2
    
    a = deg/denom
    x = x0 - a

    if abs(x - x0) < 10**(-6): 
        if abs(polynomial_eval(p, x)) < epsilon:
            return x

    itr += 1
    return Laguerre(p, x, epsilon)

def deflation(p,x):
    n = len(p)
    p1 = [] #reversing the polynomial list to fit slides format
    for i in range(n-1, -1, -1):
        p1.append(p[i])
    
    p2 = []
    for i in range(n):
        if i==0:
            p2.append(p1[i])
        else:
            p2.append(p1[i] + p2[i-1]*x)

    p3 = [] #reversing back to original format and removing the last term
    for i in range(n-2, -1, -1):
        p3.append(p2[i])

    return p3

def Laguerre_solve(p, x0, epsilon=10**(-6)):
    n = len(p)
    roots = []
    for i in range(n):
        x1 = Laguerre(p, x0-1, epsilon)
        roots.append(x1)
        p = deflation(p, x1)
        print(p, x1)
        if len(p) == 1:
            break
    return roots

x1 = Laguerre(p1, 1)
q1 = deflation(p1, x1)
print('Root 1 of p1 =', x1)
print('Deflated p1 =', q1)
x2 = Laguerre(q1, 1)
print('Root 2 of p1 =', x2)
q2 = deflation(q1, x2)
print('Deflated p1 =', q2)
#Further the method doesnt converge and give error

print('Question 2')
x3 = Laguerre(p2, 1)
q3 = deflation(p2, x3)
print('Root 1 of p2 =', x3)
print('Deflated p2 =', q3)
x4 = Laguerre(q3, 1)
print('Root 2 of p2 =', x4)
q4 = deflation(q3, x4)
print('Deflated p2 =', q4)
#Further the method doesnt converge and give error

print('Question 3')
x5 = Laguerre(p3, 1)
q5 = deflation(p3, x5)
print('Root 1 of p3 =', x5)
print('Deflated p3 =', q5)
#Further the method doesnt converge and give error


#Output:
'''
Root 1 of p1 = 1
Deflated p1 = [-6, -7, 0, 1]
Root 2 of p1 = 2.999999952546814
Deflated p1 = [1.9999997152808877, 2.999999952546814, 1]
Question 2
Root 1 of p2 = 1
Deflated p2 = [-4, -4, 1, 1]
Root 2 of p2 = 1.9999999462293525
Deflated p2 = [1.9999997311467652, 2.9999999462293525, 1]
Question 3
Root 1 of p3 = 0.5000001989620881
Deflated p3 = [8.99999641868176, -9.000003581317467, -18.999999602075743, 1.0000003979241763, 2]
''' 







