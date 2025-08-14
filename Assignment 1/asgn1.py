'''
Name - Aryan Shrivastava
Roll no. - 2311041
Assignment - 1
Question 3 and 4
'''

#Importing the libraries
from mylibrary1 import *
import matplotlib.pyplot as plt
import numpy as np

#Generating and normalising random numbers
X = RandomLCG(5000, 5, 1103515245, 12345, 32768) 
Y = RandomLCG(5000, 9, 1103515245, 12345, 32768)

X_norm = []
Y_norm = []

for i in range(len(X)):
    X_norm.append(X[i]/32768)
    Y_norm.append(Y[i]/32768)

#Finding points inside the circle and inside the square
count = 0
total = len(X_norm)
L = []
T = []


for i in range(len(X_norm)):
    if (X_norm[i]**2 + Y_norm[i]**2) <= 1:
        count += 1
    
    if i%10 == 0 and i != 0:
        T.append(i)
        L.append(4*(count/i))
    

plt.plot(T,L)
plt.title('Calculated value of pi vs number of random numbers')
plt.ylabel('Calculated  value of pi')
plt.xlabel('Number of random numbers')
plt.grid(True)
plt.show()

#Taking average over the last 3000 values(saturation region in the graph)
Calc_pi = L[-3000:]
sum = 0
for i in Calc_pi:
    sum += i 

avg = sum/len(Calc_pi)
print('The average value of pi over the last 3000 values',avg)

#Output
'''
The average value of pi over the last 3000 values 3.1303186844670843
'''


###############################################################

#Question 4

###############################################################

R = RandomLCG(10000, 10, 1103515245, 12345, 32768)
R_norm = []
for i in range(len(R)):
    R_norm.append(R[i]/32768)

#Transformation 
R_exp = []
for i in R_norm:
    R_exp.append(-np.log(i))


plt.hist(R_exp, bins=50, color='skyblue', edgecolor='black')
plt.title('Histogram plot for exponential distribution')
plt.show()

#Output plot is attached


