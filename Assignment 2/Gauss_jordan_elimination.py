'''
Name - Aryan Shrivastava
Roll no. - 2311041
Assignment 2: Gauss Jordan Elimination
'''

#Importing my library
from mylibrary1 import *

########################################################
#Question 1:
########################################################

A = read_matrix(r'Assignment 2\matrix_A.text')
b = read_matrix(r'Assignment 2\vector_b.text')
 
print('The solution to the given system of linear equations is:',GJElimination(A,b))


###########################################################
#Question 2: 
###########################################################

A1 = read_matrix(r'Assignment 2\matrix_A1.text')
b1 = read_matrix(r'Assignment 2\vector_b1.text')

print('The solution to the given system of linear equations is:',GJElimination(A1,b1))

#Output
'''
The solution to the given system of linear equations is: [-2.0, -2.0, 1.0]
The solution to the given system of linear equations is: [-1.7618170439978602, 0.8962280338740123, 4.051931404116157, -1.617130802539542, 2.041913538501913, 0.15183248715593547]
'''


