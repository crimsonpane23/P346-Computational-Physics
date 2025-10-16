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


def make_GP(num, start, cr):
        L = []
        val = start
        while len(L) < num:
            L.append(val)
            val = val * cr
        return L

    
def make_AP(num, start, cd):
        L = []
        val = start
        while len(L) < num:
            L.append(val)
            val = val + cd
        return L

  
def make_HP(num, start, cd):
        L = []
        val = start
        while len(L) < num:
            L.append(1.0 / val)
            val = val + cd
        return L
    
    
def sum_list(L):
        s = 0.0
        for x in L:
            s += x
        return s
    
def fact(n):
        if n == 0 or n == 1:
            return 1
        else:
            return n *fact(n - 1)


def is_odd(n):
        return (n % 2) != 0
        


def dot_product(A, B):
    d = 0
    for i in range(len(A)):
        d += A[i][0] * B[i][0]
    return d


def matrix_multiply(A, B):
        a, p = len(A), len(A[0])
        p1, b = len(B), len(B[0])
        if p != p1:
            raise print("Given matrices cannot be multiplied: incompatible dimensions")

        # initialize a zero matrix 
        result = [[0.0 for _ in range(b)] for _ in range(a)]
        for i in range(a):
            for k in range(p):
                aik = A[i][k]
                for j in range(b):
                    result[i][j] += aik * B[k][j]  #Add each desired element zero matrix
        return result

def RandomLCG(num, a=1103515245, c=12345, m=32768):
    x = 0.11
    L = []
    for i in range(num):
        x = (a*x + c)%m
        L.append(x)

    return L

def RandomLCG_seed(x, num, a=1103515245, c=12345, m=32768):
    L = []
    for i in range(num):
        x = (a*x + c)%m
        L.append(x)

    return L

#Pseudo Random Number Generator
def pRNG(seed, c, num):
    x = seed
    L = []
    itr = 0 
    while itr < num:
        x = c*x*(1 - x)
        itr += 1
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

#kth order correlation test, where L is the sequence
def correlation_test(k, L):
    term1 = 0
    for i in range(len(L)):
        if (i+k) > len(L)-1:
            break

        term1 += L[i]*L[i+k]
    term1 = term1/len(L)

    term2 = 0
    for i in range(k,len(L)):
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

def determinant(matrix):
    # Recursive function to find the determinant of a matrix
    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    det = 0
    for c in range(n):
        # Minor: matrix excluding first row and current column
        minor = [row[:c] + row[c+1:] for row in matrix[1:]]
        det += ((-1)**c) * matrix[0][c] * determinant(minor)
    return det

def is_invertible(matrix):
    """Check if the given square matrix is invertible (determinant ≠ 0)."""
    # Must be square
    if len(matrix) != len(matrix[0]):
        return False

    det = determinant(matrix)
    if det == 0:
        return False
    return True


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

        max_row = K[i]
        max_row_index = i
        for a in range(i,n):
            if abs(K[a]) > abs(max_row):
                max_row = K[a]
                max_row_index = a
            
        aug[i], aug[max_row_index] = aug[max_row_index], aug[i]

        #Step 4:Normalise the pivot point
        factor = aug[i][i]
        if factor == 0:
            print('The matrix is singular and the system Ax = b cannot be solved using Guass Jordan elimination method')
            return None 
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
    n = len(A)
    I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    # Initialize the inverse matrix with zeros
    A_inv = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        b0 = [[I[j][i]] for j in range(n)]  
        v = GJElimination(A, b0)            

        # Place solution vector v as the i-th column of the inverse
        for j in range(n):
            A_inv[j][i] = v[j]

    return A_inv

def LUDecomposition(A):
    #The decomposition technique uses Doolittle calculations
    #Be careful since it overwrites the original matrix A
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
    #A = orginal matrix
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


itr_bisection = 0
def Bisection(f, a, b, epsilon = 10**(-6), delta = 10**(-6)):
    global itr_bisection 
    if f(a)*f(b) < 0:
        if abs(b-a) < epsilon:
            if f(a) < delta and f(b) < delta:
                if abs(f(a)) >= abs(f(b)):
                    return b, itr_bisection
                else:
                    return a, itr_bisection
                
    c = (a + b)/2 

    if f(c)*f(a) < 0:
        b = c
    if f(c)*f(b) < 0:
        a = c
    
    itr_bisection += 1
    return Bisection(f, a, b)

itr_regula_falsi = 0 
def Regula_falsi(f, a, b, epsilon = 10**(-6), delta = 10**(-6)):
    global itr_regula_falsi
    
    c = b - ((b-a)*f(b))/(f(b) - f(a))

    if f(c)*f(a) < 0:
        bef = b
        b = c
    if f(c)*f(b) < 0:
        bef = a
        a = c
    
    if f(c) == 0:
        print('here2')
        return c, itr_regula_falsi
    
    if abs(c-bef) < epsilon:
        if f(a) < delta and f(b) < delta:
            if abs(f(a)) >= abs(f(b)):
                return b, itr_regula_falsi
            else:
                return a, itr_regula_falsi
    
    itr_regula_falsi += 1
    return Regula_falsi(f, a, b)



itr_Newton_raphson = 0
def Newton_Raphson(f, df, x0 = 0, epsilon = 10**(-6)):
    global itr_Newton_raphson
    if abs(f(x0)/df(x0)) < epsilon:
        return x0, itr_Newton_raphson

    x = x0 - f(x0)/df(x0)
    itr_Newton_raphson += 1
    return Newton_Raphson(f, df, x)

def Fixed_point_root_find(f, g, x0 = 2.5, epsilon = 10**(-6)):
    if abs(x0 - g(x0)) < epsilon:
        return x0

    x = g(x0)
    return Fixed_point_root_find(f, g, x)

######################################## DONT USE THE CODE BELOW, (OLD NOTATION) #######################################################
def poly_derivative(p):
    n = len(p)
    dp = []
    for i in range(n):
        dp.append(p[i]*i)
    return dp

def poly_deg(p):
    n = len(p)
    deg = 0
    for i in range(n-1,0,-1):
        if p[i] != 0:
            deg = i
            break
    return deg


def polynomial_eval(L,x):
    n = len(L)
    eval = 0
    for i in range(n):
        eval += L[i]*(x**i)
    return eval

####################################################### UPDATED CODE FOR NEW NOTATION #####################################################
def poly_eval(p, x):
    """Evaluate polynomial p(x) = a0*x^n + a1*x^(n-1) + ... + an."""
    result = 0
    n = len(p)
    for i in range(n):
        result += p[i] * (x ** (n - i - 1))
    return result

def poly_derivative(p):
    """Return first derivative coefficients."""
    n = len(p)
    dp = []
    for i in range(n - 1):
        dp.append(p[i] * (n - i - 1))
    return dp

def poly_second_derivative(p):
    """Return second derivative coefficients."""
    n = len(p)
    d2p = []
    for i in range(n - 2):
        d2p.append(p[i] * (n - i - 1) * (n - i - 2))
    return d2p

def laguerre_root(p, x0=1, epsilon=1e-6, max_iter=1000):
    """Find one real root of polynomial p using Laguerre's method (loop version)."""
    n = len(p) - 1  # degree

    x = x0
    for _ in range(max_iter):
        f = poly_eval(p, x)
        if abs(f) < epsilon:
            return x

        f1 = poly_eval(poly_derivative(p), x)
        f2 = poly_eval(poly_second_derivative(p), x)

        G = f1 / f
        H = G * G - f2 / f
        D_term = (n - 1) * (n * H - G * G)

        if D_term < 0:
            D_term = 0  # clamp negative sqrt

        D = math.sqrt(D_term)

        denom1 = G + D
        denom2 = G - D
        denom = denom1 if abs(denom1) > abs(denom2) else denom2
        if denom == 0:
            break

        a = n / denom
        x_new = x - a

        if abs(x_new - x) < epsilon:
            return x_new

        x = x_new

    return x  # return best estimate if max_iter reached

def deflate(p, root):
    """Perform synthetic division of p(x) by (x - root)."""
    n = len(p)
    q = [p[0]]
    for i in range(1, n):
        q.append(p[i] + root * q[-1])
    q.pop()  # remove remainder
    return q

def laguerre_all_roots(p, x0=1, epsilon=1e-6):
    """Find all real roots of polynomial using Laguerre's method with deflation."""
    roots = []
    poly = p[:]
    while len(poly) > 1:
        root = laguerre_root(poly, x0, epsilon)
        roots.append(round(root, 6))
        poly = deflate(poly, root)
    return roots


def Midpoint_int(f,a,b,N):
    h = (b-a)/N
    L = []     #List of midpoints
    for i in range(N):
        x = (a + i*h + a + (i+1)*h)/2
        L.append(x)

    #Sum function evaluation on these points
    m = 0
    for i in range(len(L)):
        m += f(L[i])*h
    
    return m

def Trapezoidal_int(f,a,b,N):
    h = (b-a)/N
    L = []     #List of endpoints of each interval (basically Partition set)
    for i in range(N):
        x = a + i*h
        L.append(x)
    L.append(b)

    #Summing areas of each trapezoid
    t = 0
    for i in range(N):
        t += (f(L[i]) + f(L[i+1]))*h/2 
    return t


def Monte_Carlo_integral(f, a, b, N):
    L1 = RandomLCG_seed(0.1,N)
    #Normalising the random numbers between 0 and 1
    m=32768
    for i in range(N):
        L1[i] = L1[i]/m
    
    #Shifting random numbers to the domain [a,b]
    L2 = []
    for i in range(N):
        L2.append(a + (b-a)*L1[i])
    
    #Finding Fn
    F = 0
    F_list = []
    Itr_list = []
    for i in range(N):
        F += f(L2[i])
        if i%1000 == 0:
            F_list.append((b-a)*F/(i+1))
            Itr_list.append(i)
    F = (b-a)*F/N
    
    plt.plot(Itr_list, F_list)
    plt.xlabel("Iterations")
    plt.ylabel("Integral estimated")
    plt.title("Integral estimated vs iterations")
    plt.show()
    return F


def Simpson_integral(f, a, b, N):
    h = (b-a)/N
    L = []     #Make the partition
    for i in range(N):
        x = a + i*h
        L.append(x)
    L.append(b)
    
    simp = 0
    for i in range(len(L)):
        if i==0:
            simp += f(L[i])
        if i==N:
            simp += f(L[i])
        if i%2 != 0:
            simp += 4*f(L[i])
        if i%2 == 0 and i!=N and i!=0:
            simp += 2*f(L[i])
    simp = simp*h/3

    return simp

def GaussianQuad(f,a,b,N):
    x_np, w_np = np.polynomial.legendre.leggauss(N)
    quad1 = 0
    for i in range(N):
        quad1 += w_np[i]*f(((b-a)/2)*x_np[i] + (b+a)/2)
    quad1 = quad1*(b-a)/2

    return quad1