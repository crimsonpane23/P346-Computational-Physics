'''
Name - Aryan Shrivastava
Roll no. - 2311041
Assignment 2 - LU Decomposition
'''

###################################
#Question - 1
###################################


from mylibrary1 import *

A = read_matrix(r'Assignment 2 - LU\Matrix_A.text')

print(LUDecomposition(A),'is the LU Decomposition of A')

###################################
#Question - 2
###################################

A1 = read_matrix(r'Assignment 2 - LU\Matrixq2.text')

b1 = read_matrix(r'Assignment 2 - LU\vectorq2.text')

def LUSolve(A, b):
    n = len(A)
    D = LUDecomposition(A)
    
    #Making the L and U matrices again
    L = []
    U = []
    for i in range(n):
        row = []
        row1 = []
        for j in range(n):
            if i == j:
                row.append(1)
                row1.append(D[i][j])
            if i < j:
                row.append(0)
                row1.append(D[i][j])
            if i > j:
                row.append(D[i][j])
                row1.append(0)
        L.append(row)
        U.append(row1)
    
    #Forward substitution
    y = []
    for i in range(n):
        sum3 = 0
        for j in range(i):
            sum3 += L[i][j]*y[j]
        y.append(b[i][0] - sum3)
    
    #Backward Substitution
    X = []
    for i in range(n):
        X.append([0])
    
    count = n-1
    X[n-1][0] = y[n-1]/U[n-1][n-1]

    while count > 0:
        count -= 1
        sum4 = 0
        for j in range(count+1,n):
            sum4 += U[count][j]*X[j][0]
        X[count][0] = (y[count] - sum4)/U[count][count]
    
    return X

print(LUSolve(A1,b1))

#Output
#-----------------------------------------#
'''
[[1.0, 2.0, 4.0], [3.0, 2.0, 2.0], [2.0, 1.0, 3.0]] is the LU Decomposition of A
[[-1.6382182862226529], [0.897650531764325], [4.011929313761171], [-1.5796837720894497], [1.990436329815197], [0.16747543370132217]]
'''
        








