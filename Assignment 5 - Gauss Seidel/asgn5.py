'''
Name - Aryan Shrivastava
Roll no. - 2311041
Assignment - 5
(Gauss - Seidel)
'''

from mylibrary1 import *
import math

#--------------------------Question 1----------------------------------#

Q1 = read_matrix(r'Assignment 5 - Gauss Seidel\matrixQ1.text')
b1 = read_matrix(r'Assignment 5 - Gauss Seidel\vectorbQ1.text')

print('The matrix Q1 is symmetric:',Symm_matrix_check(Q1))

if Symm_matrix_check(Q1) == True:
    print('Solution from Cholesky:',Cholesky_solve(Q1,b1))
else:
    print('Q1 matrix is not symmetric, hence Cholesky decomposition cannot be used')

print('Gauss-Seidel:',Gauss_Seidel(Q1,b1))      #Precision - 10**(-6)

#---------------------------Question 2---------------------------------#

Q2 = read_matrix(r'Assignment 5 - Gauss Seidel\MatrixQ2.text')
b2 = read_matrix(r'Assignment 5 - Gauss Seidel\vectorbQ2.text')

#Making the matrix diagonally dominant manually
Q2[2], Q2[4] = Q2[4], Q2[2]
b2[2], b2[4] = b2[4], b2[2]

Q2[0], Q2[3] = Q2[3], Q2[0]
b2[0], b2[3] = b2[3], b2[0]

print('Is the matrix diagonally dominant after manual interchanges:',Diagonal_dominance_check(Q2))
print('Jacobi:', Jacobi(Q2,b2))
print('Gauss-Seidel:',Gauss_Seidel(Q2,b2))

#-------------------------------Output------------------------------#
'''
The matrix Q1 is symmetric: True
Solution from Cholesky: [[1.0], [0.9999999999999999], [1.0], [1.0], [1.0], [1.0]]
Number of iterations for convergence for Gauss-Seidel method: 15
Gauss-Seidel: [[0.9999997530614102], [0.9999997892247294], [0.9999999100460266], [0.9999998509593769], [0.9999998727858708], [0.9999999457079743]]
Is the matrix diagonally dominant after manual interchanges: True
The sequence of solutions has converged
Number of iterations for convergence in Jacobi method: 62
Jacobi: [[2.9791652963501267], [2.2155997156801988], [0.21128418509279026], [0.15231709056098097], [5.715033754695062]]
Number of iterations for convergence for Gauss-Seidel method: 11
Gauss-Seidel: [[2.979165086347139], [2.215599676186742], [0.21128402698819157], [0.15231700827754802], [5.715033568811629]]
'''














