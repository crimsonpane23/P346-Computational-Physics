'''
Name - Aryan Shrivastava
Roll no. - 2311041
Endsem
'''

from mylibrary1 import *
import matplotlib.pyplot as plt
import math
import numpy as np


#-------------------------Question 1--------------------#

N = 10000
L1 = RandomLCG(5000)
#Normalising the random numbers between 0 and 1
m=32768
for i in range(5000):
    L1[i] = L1[i]/m

#We need to find equilibrium by plotting number of particles in both sides (they must converge to 2500)
#I tried to use for loops to iterate over all particles(represented by random numbers and determine where they should go by their value) but the loop was too long to compute, idk the problem. 
#So i tried to simpllify it but it has logical error.

Time = []
no_l = []
no_r = []
left = 5000
right = 0
for t in range(N):
    Time.append(t)
    no_l.append(left)
    no_r.append(right)
    r = RandomLCG(1)
    r = r[0]/m
    if r>0.5 and left>0 and right<5000 and right<left:
        left -= 1
        right += 1
    if r<=0.5 and right>0 and left<5000 and left<right:
        left += 1
        right -= 1


plt.scatter(Time, no_l, color='r', label='no of points in the left half')
plt.scatter(Time, no_r, color='b', label='no of points in the right half')
plt.xlabel('Time')
plt.ylabel('Number of particles')
plt.grid(True)
plt.legend()
plt.title('Number of particles in each side variation with time')
plt.show()


#--------------------------Question 2---------------------------#

A = read_matrix(r'Endsem\matrixA.text')
b = read_matrix(r'Endsem/vectorb.text')

x = Gauss_Seidel(A, b)
print(x)

#Output
'''
Number of iterations for convergence for Gauss-Seidel method: 15
[[0.9999997530614102], [0.9999997892247294], [0.9999999100460266], [0.9999998509593769], [0.9999998727858708], [0.9999999457079743]]
'''

#-------------------------Question 3----------------------------#

def F(x):
    return 2.5 - x*math.exp(x)

def DF(x):
    return -math.exp(x) - x*math.exp(x)

root, itr_Newton_raphson = Newton_Raphson(F,DF,0)
print('The spring can be strteched this many units of length far',root)

#output
'''
The spring can be strteched this many units of length far 0.9585863570793399
'''

#-----------------------------Question 4-----------------------#
#Using Gaussian quadrature

def lamb(x):
    return x**2

com = GaussianQuad(lamb,0,2,5)/2
print('The position of centre of mass is',com)

#Output
'''
The position of centre of mass is 1.3333333333333333
'''

#-----------------Question 5-------------------------#

def f(v, y):
    if abs(v)<10**(-2.6):
        print('Max height reached with air resistance is approximately:', y)
    if v!=0:
        return -0.02 -10/v

ly, lv= ODE_rk4(f, 0, 10, 0.001, 0, 5)

t = []
for i in range(len(ly)):
    t.append((lv[i],ly[i]))

plt.scatter(ly, lv)
plt.xlabel('height')
plt.ylabel('velocity')
plt.title('Variation of velocity with height')
plt.grid(True)
plt.show()

#Without air resistance max height = 5m 
#Output
'''
Max height reached with air resistance is approximately: 4.935999999999983
'''


#----------------Question 6----------------#

def g(x):
    return 20*abs(np.sin(math.pi*x)) 

V, X, T = PDE_HeatEqn_Solve(g,0,4,0,2,0.1,0.0008)
V0 = V[0]
V1 = V[10]
V2 = V[20]
V3 = V[50]
V4 = V[100]
V5 = V[200]
V6 = V[500]
V7 = V[1000]
plt.plot(X,V0, label="PDE at time_step=0")
plt.plot(X,V1, label="PDE at time_step=10")
plt.plot(X,V2, label="PDE at time_step=20")
plt.plot(X,V3, label="PDE at time_step=50")
plt.plot(X,V4, label="PDE at time_step=100")
plt.plot(X,V5, label="PDE at time_step=200")
plt.plot(X,V6, label="PDE at time_step=500")
plt.plot(X,V7, label="PDE at time_step=1000")
plt.xlabel('Position (x)')
plt.ylabel('Temperature (T)')
plt.title("PDE solution evolution with time")
plt.legend()
plt.grid()
plt.show()


#----------------------------Question 7-------------#

data = read_matrix(r'Endsem\esem4fit.txt')

x_data = []
y_data = []
for i in range(len(data)):
    x_data.append(data[i][0])
    y_data.append(data[i][1])

coeff = LeastsquaresPolynomialFit(x_data, y_data, 4)
print('The coefficients for th fitted polynomial are:', coeff)

def polynfit(x):
    return coeff[0] + coeff[1]*x + coeff[2]*(x**2) + coeff[3]*(x**3) + coeff[4]*(x**4)


s = np.linspace(-2,2, 1000)
plt.scatter(x_data, y_data, label='data points')
plt.plot(s, [polynfit(x) for x in s], label='fitted curve', color='r')
plt.grid(True)
plt.ylabel('y')
plt.legend()
plt.xlabel('x')
plt.show()

#Output
'''
The coefficients for th fitted polynomial are: [0.254629507211548, -1.1937592138092277, -0.45725541238296813, -0.8025653910658186, 0.013239427477396298]
'''








