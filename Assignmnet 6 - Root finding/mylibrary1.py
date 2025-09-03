# Name - Aryan Shrivastava
# Roll no. - 2311041
# File - Library file 

import numpy as np  
import math
import matplotlib.pyplot as plt

class MyComplex():

    def __init__(self, real, imag=0.0):
        self.r = real
        self.i = imag

    def display_complex(self):
        print(self.r,", ", self.i,"i", sep="")

    def add_complex(self, c1, c2):
        self.r = c1.r + c2.r
        self.i = c1.i + c2.i 
        return MyComplex(self)

    def sub_complex(self, c1, c2):
        self.r = c1.r - c2.r
        self.i = c1.i - c2.i 
        return MyComplex(self)

    def mul_complex(self, c1, c2):
        self.r = c1.r*c2.r - c1.i*c2.i 
        self.i = c1.i*c2.r + c1.r*c2.i 
        return MyComplex(self)

    def mod_complex(self):
        return np.sqrt(self.r**2 + self.i**2)


class MyMatrix:

    @staticmethod
    def shape(M):
        rows = len(M)
        cols = len(M[0])
        return (rows, cols)

    @staticmethod
    def matrix_multiply(A, B):
        a_rows, a_cols = MyMatrix.shape(A)
        b_rows, b_cols = MyMatrix.shape(B)
        if a_cols != b_rows:
            raise ValueError("Given matrices cannot be multiplied: incompatible dimensions")

        # initialize a zero matrix 
        result = [[0.0 for _ in range(b_cols)] for _ in range(a_rows)]
        for i in range(a_rows):
            for k in range(a_cols):
                aik = A[i][k]
                for j in range(b_cols):
                    result[i][j] += aik * B[k][j]  #Add each desired element zero matrix
        return result

    @staticmethod
    def read_matrix(filename):
        matrix = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = [float(x) for x in line.split()]
                matrix.append(row)
        return matrix


class MySeq:

    @staticmethod
    def make_GP(num, start, cr):
        L = []
        val = start
        while len(L) < num:
            L.append(val)
            val = val * cr
        return L

    @staticmethod
    def make_AP(num, start, cd):
        L = []
        val = start
        while len(L) < num:
            L.append(val)
            val = val + cd
        return L

    @staticmethod
    def make_HP(num, start, cd):
        L = []
        val = start
        while len(L) < num:
            L.append(1.0 / val)
            val = val + cd
        return L
    
    @staticmethod
    def sum_list(L):
        s = 0.0
        for x in L:
            s += x
        return s
    
    @staticmethod
    def RandomLCG(num, a=1103515245, c=12345, m=32768):
        x = 0.11
        L = []
        for i in range(num):
            x = (a*x + c)%m
            L.append(x)

        return L


class MyNum:

    @staticmethod
    def fact(self, n):
        if n == 0 or n == 1:
            return 1
        else:
            return n * self.fact(n - 1)

    @staticmethod
    def is_odd(self, n):
        return (n % 2) != 0
        

class MyVector:

    @staticmethod
    def dot_product(A, B):
        d = 0
        for i in range(len(A)):
            d += A[i][0] * B[i][0]
        return d


def matrix_multiply(A, B):
        a_rows, a_cols = MyMatrix.shape(A)
        b_rows, b_cols = MyMatrix.shape(B)
        if a_cols != b_rows:
            raise ValueError("Given matrices cannot be multiplied: incompatible dimensions")

        # initialize a zero matrix 
        result = [[0.0 for _ in range(b_cols)] for _ in range(a_rows)]
        for i in range(a_rows):
            for k in range(a_cols):
                aik = A[i][k]
                for j in range(b_cols):
                    result[i][j] += aik * B[k][j]  #Add each desired element zero matrix
        return result

def RandomLCG(num, a=1103515245, c=12345, m=32768):
    x = 0.11
    L = []
    for i in range(num):
        x = (a*x + c)%m
        L.append(x)

    return L

def coplot(L, k):
    L0 = []
    for i in range(len(L)):
        if (i+k) > len(L)-1:
            break
        L0.append(L[i+k-1])
    
    for i in range(k):
        L.remove(L[len(L)-1-i])
    
    plt.scatter(L, L0, marker='o', color='r')
    plt.xlabel('x_i')
    plt.ylabel(f'x_i+{k}')
    plt.grid()
    plt.title(f'x_i vs x_i+{k} plot')
    plt.show()

def correlation_test(k, L):
    term1 = 0
    for i in range(len(L)):
        if (i+k) > len(L)-1:
            break

        term1 += L[i]*L[i+k]
    term1 = term1/len(L)

    term2 = 0
    for i in range(len(L)):
        term2 += L[i]
    term2 = term2/len(L)
    term2 = term2*term2

    return term1 - term2

#To read a matrix (outside the class)
def read_matrix(filename):
        matrix = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = [float(x) for x in line.split()]
                matrix.append(row)
        return matrix


def GJElimination(A, b):
    #Step 1: Make the Augumented matrix [A|b]:
    n = len(A)
    aug = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(A[i][j])
        row.append(b[i][0])
        aug.append(row)

    #Step 2: Elimination process for every column
    for i in range(n):

        #Step 3: Find the pivot and place it appropriately
        K = []
        for j in range(n):
            k = aug[j][i]
            K.append(k)

        max_row = K[0]
        max_row_index = 0
        for a in range(n):
            if K[a] >= max_row:
                max_row = K[a]
                max_row_index = a
            
        aug[i], aug[max_row_index] = aug[max_row_index], aug[i]

        #Step 4:Normalise the pivot point
        factor = aug[i][i]
        if factor == 0:
            print('The matrix is singular and the system Ax = b cannot be solved using Guass Jordan elimination method')
            exit()
        for j in range(n+1):
            aug[i][j] = aug[i][j]/factor

        #Step 5: Perform the elimination
        for j in range(n):
            if j!=i:
                catch = aug[j][i]
                for k in range(n+1):
                    aug[j][k] = aug[j][k] - catch*aug[i][k]

    #Step 6: Extract the final output from last column
    out = []
    for j in range(n):
        out.append(aug[j][n])
    return(out)
            
def GJInverse(A):
    #Step 1: Make an identity matrix
    I = []
    n = len(A)
    for j in range(n):
        row = []
        for i in range(n):
            if i == j:
                row.append(1)
            else:
                row.append(0)
        I.append(row)
    
    #Step 2: Apply GJElimination for A with all columns of I one by one and merge them to form A inverse
    A_inv = []
    Final = []
    for i in range(n):
        b0 = []
        for j in range(n):
            b0.append([I[i][j]])
        v = GJElimination(A, b0)
        Final.append(v)
    
    return Final

def LUDecomposition(A):
    #The decomposition technique uses Doolittle calculations
    n = len(A)
    for j in range(n):
        for i in range(n):
            if i <= j:
                sum1 = 0
                for k in range(i):
                    sum1 += A[i][k]*A[k][j]
                A[i][j] = (A[i][j] - sum1)
            if i > j:
                sum2 = 0
                for k in range( j):
                    sum2 += A[i][k]*A[k][j]
                A[i][j] = (A[i][j] - sum2)/A[j][j]

    return A

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


def Cholesky_decompose(A, b):
    n = len(A)
    L = [[0.0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                sum = 0
                for k in range(i):
                    sum += L[i][k]**2
                L[i][i] += math.sqrt(A[i][i] - sum)

            if i < j:
                sum1 = 0
                for k in range(i):
                    sum1 += L[i][k]*L[k][j]
                L[i][j] += (A[i][j] - sum1)/L[i][i]
                L[j][i] += (A[i][j] - sum1)/L[i][i]

    return L

def Cholesky_solve(A,b):
    L = Cholesky_decompose(A,b)
    n = len(A)
    L_upper = [[0.0 for i in range(n)] for j in range(n)]
    L_lower = [[0.0 for i in range(n)] for j in range(n)]

    for i in range(n):
        for j in range(n):
            if i <= j:
                L_upper[i][j] += L[i][j]
            if i >= j:
                L_lower[i][j] += L[i][j]
    
    #Forward substitution
    y = []
    for i in range(n):
        sum3 = 0
        for j in range(i):
            sum3 += L_lower[i][j]*y[j]
        y.append((b[i][0] - sum3)/L[i][i])
    
    #Backward Substitution
    X = []
    for i in range(n):
        X.append([0])
    
    count = n-1
    X[n-1][0] = y[n-1]/L_upper[n-1][n-1]

    while count > 0:
        count -= 1
        sum4 = 0
        for j in range(count+1,n):
            sum4 += L_upper[count][j]*X[j][0]
        X[count][0] = (y[count] - sum4)/L_upper[count][count]
        
    return X

def Jacobi(A, b):
    '''Solve the convergence problem'''
    #Iterative method for solving Ax = b
    #Here, A must be Hermitian (symmetric for real matrices) and det(A) > 0

    n = len(A)
    #Initial guess 
    x0 = [[0] for i in range(n)]
    X = []              #Stores all the solutions across the iterations
    X.append(x0)

    for k in range(1000):
        x_knext = [[0] for i in range(n)]
        for i in range(n):
            sum = 0
            for j in range(n):
                if i != j:
                    sum += A[i][j]*X[k][j][0]

            x_knext[i][0] += (b[i][0] - sum)/A[i][i]
        X.append(x_knext)

        #Test convergence and break the loop
        epsilon = 10**(-6)
        
        #Taking l2-norm
        norm = 0
        if k!=0:
            for i in range(n):
                norm += (X[k][i][0] - X[k-1][i][0])**2

            norm = math.sqrt(norm) 
            if norm < epsilon:
                print('The sequence of solutions has converged')
                break

    r = len(X)
    print('Number of iterations for convergence in Jacobi method:',r)
    sol = X[r-1]

    return sol

def Diagonal_dominance_check(A):
    n = len(A)
    for i in range(n):
        sum_row = 0
        for j in range(n):
            if i != j:
                sum_row += abs(A[i][j])
        if abs(A[i][i]) < sum_row:
            return False
    return True

def Symm_matrix_check(A):
    n = len(A)
    for i in range(n):
        for j in range(n):
            if A[i][j] != A[j][i]:
                return False
            
    return True

def Gauss_Seidel(A,b):
    n = len(A)
    epsilon = 10**(-6)
    X = [[0] for i in range(n)]
    
    for k in range(1000):
        norm = 0
        for i in range(n):
            sum1 = 0
            sum2 = 0
            for j in range(n):
                if i > j:
                    sum1 += A[i][j]*X[j][0]
                if i < j:
                    sum2 += A[i][j]*X[j][0]

            norm += (X[i][0] - ((b[i][0] - sum1 - sum2)/A[i][i]))**2
            X[i][0] = (b[i][0] - sum1 - sum2)/A[i][i]
        
        norm = math.sqrt(norm)
        if norm < epsilon:
            print('Number of iterations for convergence for Gauss-Seidel method:',k)
            break

    return X

def Make_diagonally_dominant(A,b):
    #Not completely working yet
    n = len(A)
    for i in range(n):
        max_row = i
        for j in range(i+1, n):
            if abs(A[j][i]) > abs(A[max_row][i]):
                max_row = j
        if max_row != i:
            A[i], A[max_row] = A[max_row], A[i]
            b[i], b[max_row] = b[max_row], b[i]
    return A, b

def bracket_adjust(f, a, b, beta = 0.5):
    if f(a)*f(b) < 0:
        return a,b
    
    if f(a)*f(b) > 0:
        if abs(f(a)) < abs(f(b)):
            a = a - beta*(b-a)
        if abs(f(a)) > abs(f(b)):
            b = b - beta*(b-a)
    
    return bracket_adjust(f, a, b)

def Bisection(f, a, b, epsilon = 10**(-6), delta = 10**(-6)):
    a, b = bracket_adjust(f, a, b)

    if f(a)*f(b) < 0:
        if abs(b-a) < epsilon:
            if f(a) < delta and f(b) < delta:
                if abs(f(a)) >= abs(f(b)):
                    return b, f(b)
                else:
                    return a , f(a)
                
    
            
    c = (a + b)/2 

    if f(c)*f(a) < 0:
        b = c
    if f(c)*f(b) < 0:
        a = c
    
    return Bisection(f, a, b)

def Regula_falsi(f, a, b, epsilon = 10**(-6), delta = 10**(-6)):
    a, b = bracket_adjust(f, a, b)

    if f(a)*f(b) < 0:
        if abs(b-a) < epsilon:
            if f(a) < delta and f(b) < delta:
                if abs(f(a)) >= abs(f(b)):
                    return b, f(b)
                else:
                    return a , f(a)
            
    c = b - ((b-a)*f(b))/(f(b) - f(a))

    if f(c)*f(a) < 0:
        b = c
    if f(c)*f(b) < 0:
        a = c

    return Regula_falsi(f, a, b)