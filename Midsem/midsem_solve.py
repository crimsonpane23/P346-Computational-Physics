'''
Name - Aryan Shrivastava
Roll no. - 2311041

Mid-semester examination

'''

from mylibrary1 import *
import matplotlib.pyplot as plt
import math

#--------------------------------------Question 1-----------------------------------------#
print('--------Question 1---------')
'''
LOGIC 
Equation of ellipse: (x/a)^2 + (y/b)^2 = 1
a : semi major axis = 2
b : semi minor axis = 1

Inside ellipse condition in 1st quadrant: (x/2)^2 + y^2 < 1

Define ratio = (no. of points inside ellipse/total points in rectangle [0,2]x[0,2]) 

Area of ellipse = 4*area of ellipse in 1st quadrant

area of ellipse in first quadrant = (area of rectangle [0,2]x[0,2])*ratio
                                  = 4*ratio

Area of ellipse = 4*4*ratio
                = 16*ratio 

In the plot, steady convergence was observed after around N=2000, so we average over values of area for N>=2000 in teh calculation
'''
#Generating and normalising random numbers
X = RandomLCG_seed(1,5000) 
Y = RandomLCG_seed(9,5000)

X_norm = []
Y_norm = []

#Normalising to the range [0,2]
for i in range(len(X)):
    X_norm.append(X[i]*2/32768)
    Y_norm.append(Y[i]*2/32768)

#Finding points inside the first quadrant of ellipse and inside the square of [0,2]x[0,2]
count = 0
total = len(X_norm)
L = []              #For area vs number of random points plot
T = []
In = []             #For area visualization plot
In2 = []
Out = []
Out2 = []
S = []              #To calculate area


for i in range(len(X_norm)):
    if ((X_norm[i]**2)/4) + (Y_norm[i])**2 < 1:
        In.append(X_norm[i])
        In2.append(Y_norm[i])
        count += 1
    else:
        Out.append(X_norm[i])
        Out2.append(Y_norm[i])
    
    if i%10 == 0 and i != 0:
        T.append(i)
        L.append(16*(count/i))
    if i!=0: 
        S.append(16*(count/i))
    

#Taking average over the last 3000 values(saturation region in the graph)

calc_ellipse = S[-3000:]
sum = 0
for i in calc_ellipse:
    sum += i 

avg = sum/len(calc_ellipse)
print('Area of the ellipse calculated:',avg)

#Ellipse area visualisation
plt.scatter(In,In2, color='red')
plt.scatter(Out,Out2,color = 'blue')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Random numbers distribution (Red - inside ellipse) (Blue - Outside ellipse)')
plt.show()

#Ellipse area vs number of random points plot (Good convergence observed)
plt.plot(T,L)
plt.title('Calculated value of area of ellipse vs number of random numbers')
plt.ylabel('Calculated  value of area of ellipse')
plt.xlabel('Number of random numbers')
plt.grid(True)
plt.show()


# --------------------------------------------Question 2-------------------------------------------#
print('')
print('--------Question 2---------')
def f(x):
    return (x-5)*math.exp(x) + 5

def df(x):
    return math.exp(x) + (x-5)*math.exp(x)

x, itr = Newton_Raphson(f,df, 6)
print('Root of given equation: ', x,itr)

h = 6.626*10**(-34)
c = 3*10**8
k = 1.381*10**(-23)
b = h*c/(k*x)
print('Wein constant calculated:', b)


#Question 3
print('')
print('--------Question 3---------')
M = [[0.2, -5, 3, 0.4, 0],
     [-0.5 ,1 ,7 ,-2 ,0.3],
     [0.6, 2 , -4, 3, 0.1],
     [3, 0.8, 2, -0.4, 3],
     [0.5, 3, 2, 0.4, 1]]

x1 = LUSolve(M,[[1],[0],[0],[0],[0]])
x2 = LUSolve(M,[[0],[1],[0],[0],[0]])
x3 = LUSolve(M,[[0],[0],[1],[0],[0]]) 
x4 = LUSolve(M,[[0],[0],[0],[1],[0]])
x5 = LUSolve(M,[[0],[0],[0],[0],[1]])  

Inv = [[x1[0],x2[0],x3[0],x4[0],x5[0]],
       [x1[1],x2[1],x3[1],x4[1],x5[1]],
       [x1[2],x2[2],x3[2],x4[2],x5[2]],
       [x1[3],x2[3],x3[3],x4[3],x5[3]],
       [x1[4],x2[4],x3[4],x4[4],x5[4]]]

print(Inv)


#Question 4
print('')
print('--------Question 4---------')
A = read_matrix(r'Midsem\matrixA.text')
b = read_matrix(r'Midsem\vectorB.text')

print('Solution using Gauss Seidel:',Gauss_Seidel(A,b))  #Epsilon = 10**(-6) by default in the function


#-----------------------------------Output--------------------------------------------#
'''
--------Question 1---------
Area of the ellipse calculated: 6.225425225201637

--------Question 2---------
Root of given equation:  4.965114942713781 5
Wein constant calculated: 0.0028990099156204225

--------Question 3---------
LU Decomposed matrix of M (rounded of to 3 decimal places): [[0.2, -5.0, 3.0, 0.4, 0.0], [-2.5, -11.5, 14.5, -1.0, 0.3], [3.0, -1.478, 8.435, 0.322, 0.543], [15.0, -6.591, 6.233, -14.997, 1.59], [2.5, -1.348, 1.665, 0.166, 0.236]]

--------Question 4---------
Number of iterations for convergence for Gauss-Seidel method: 12
Solution using Gauss Seidel: [[1.4999998297596437], [-0.4999999999999992], [1.9999999999999998], [-2.4999999148640373], [1.0000000000000004], [-0.9999999999957907]]
'''






