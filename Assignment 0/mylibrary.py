# Name - Aryan Shrivastava
# Roll no. - 2311041
# File - Library file

import numpy as np

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
    
class MyMatrix():

    def __init__(self):
        pass

    def matrix_multiply(self, B):

        if len(self[0]) != len(B):
            raise ValueError("Given matrices cannot be multiplied")
    
        result = [[0 for k in range(len(B[0]))] for k in range(len(self))]  #Start with a zero matrix

        for i in range(len(self)):
            for j in range(len(B[0])):     #Go to each element of the initialized zero matrix
                for k in range(len(B)):
                    result[i][j] += self[i][k] * B[k][j]    #Add to get desired element
        return result

    def read_matrix(self,filename):  #To read a txt file 
        with open(filename,'r') as f:
            matrix = []
            for line in f:
                #Split the line into numbers and convert into int/float
                row = [float(num) for num in line.strip().split()]
                matrix.append(row)
            return matrix
        
    
class MySeq():
    
    def __init__(self):
        pass

    #Makes a list of numbers in the GP
    def make_GP(self,num, start, cr): 
        L = []
        var = start
        while len(L) < num:
            L.append(var)
            var = var*cr
        return L
    
    #Makes a list of numbers in the AP
    def make_AP(self,num, start, cd):
        L = []
        var = start
        while len(L) < num:
            L.append(var)
            var += cd 
        return L
    
    #Makes a list of numbers in the HP
    def make_HP(self,num, start, cd):
        L = []
        var = start
        while len(L) < num:
            L.append((var)**(-1))
            var += cd 
        return L
    
    def sum_list(self, L):
        sum = 0
        for i in L:
            sum += i
        return sum

class MyNum():

    def __init__(self):
        pass


    def fact(self, n):  
        if n == 0:
            return 1
        else:
            return n*self.fact(n-1)
        
    #Function to check if a given number is odd
    def is_odd(n):  
        if n%2 != 0:
            return True
        else:
            return False
        
class Myvector():

    def __init__(self):
        pass

    def dot_product(self,B): 
        d = 0
        for i in range(len(self)):
            d += self[i][0]*B[i][0]
        return d
    
    
    
    
