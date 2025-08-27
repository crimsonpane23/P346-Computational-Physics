'''
Name - Aryan Shrivastava
Roll no. - 2311041
Assignment 4 - Cholesky and Jacobi
'''

from mylibrary1 import *
import numpy as np
import math

A = read_matrix(r'Assignmnet 4 - Cholesky and Jacobi\matrixA.text')
#print(A)

b = read_matrix(r'Assignmnet 4 - Cholesky and Jacobi\vectorb.text')
#print(b)

print(Jacobi(A,b))
print(Cholesky_solve(A,b))

'''
The sequence of solutions has converged
Number of iterations for convergence: 45
[[0.0], [0.9999997019767761], [0.9999998509883881], [0.9999998509883881]]
[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
[[0.967544818005012], [0.9774985876498932], [0.5324470642591002], [0.6198750760709587]]

'''


