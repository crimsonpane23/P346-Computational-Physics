# Name - Aryan Shrivastava
# Roll no. - 2311041
# File - Library file 

import numpy as np  
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