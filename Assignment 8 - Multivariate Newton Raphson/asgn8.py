'''
Name: Aryan Shrivastava
Roll no. - 2311041
Assignment 8: Multivariate Root finding
'''

from mylibrary1 import *
import math

def f1(x1,x2,x3):
    return x1**2 + x2 -37 

def f2(x1,x2,x3):
    return x1 - x2**2 - 5

def f3(x1,x2,x3):
    return x1 + x2 + x3 - 3

def g1(x1, x2, x3):
    return math.sqrt(37 - x2)

def g2(x1,x2,x3):
    return math.sqrt(x1 - 5)

def g3(x1,x2,x3):
    return 3 - x1 - x2

def F(x1,x2,x3):
    return [[f1(x1,x2,x3)], [f2(x1,x2,x3)], [f3(x1,x2,x3)]]

def J(x1,x2,x3):
    return [[2*x1, 1, 0], [1, -2*x2, 0], [1, 1, 1]]

f = [f1, f2, f3]
g = [g1, g2, g3]

def l2norm(x):
    n = len(x)
    sum = 0
    for i in range(n):
        sum += x[i]**2
    return math.sqrt(sum)

itr = 0
def Multi_fixed_point_root_find(f, g, x0, epsilon = 10**(-6)):
    global itr
    n = len(f)
    x1,x2,x3 = x0[0], x0[1], x0[2]
    y = []
    for i in range(n):
        y.append(g[i](x1,x2,x3))

    diff = []
    for i in range(n):
        diff.append(x0[i] - y[i])
    
    if l2norm(diff)/l2norm(y) < epsilon:
        return x0
    
    x =[]
    for i in range(n):
        x.append(g[i](x1,x2,x3))
    itr += 1
    return Multi_fixed_point_root_find(f, g, x)

print(Multi_fixed_point_root_find(f,g, [5,5,5]))
print(itr)


itr1 = 0
def Multi_Newt_Raphson(F, J, x0, epsilon=10**(-6)):
    global itr1
    n = 3
    x1,x2,x3 = x0[0], x0[1], x0[2]
    Jinv = GJInverse(J(x1,x2,x3))
    term2 = (matrix_multiply(Jinv,F(x1,x2,x3)))
    x = []
    for i in range(n):
        x.append(x0[i] - term2[i][0])

    diff = []
    for i in range(n):
        diff.append(x0[i] - x[i])
    
    if l2norm(diff)/l2norm(x) < epsilon:
        return x

    itr1+=1
    return Multi_Newt_Raphson(F,J, x)

    
print(Multi_Newt_Raphson(F,J,[5,5,5]))
print(itr1)
print('Newton Raphson converges faster than Fixed point iteration method')
#There was a problem in my GJInverse code earlier whihc i have fixed, it was giving transpose of the inverse matrix of A

'''
[6.000000244759765, 0.9999994315904682, -3.999995926064018]
10
[5.9999999999995834, 1.0000000000050626, -4.000000000004646]
5
Newton Raphson converges faster than Fixed point iteration method
'''