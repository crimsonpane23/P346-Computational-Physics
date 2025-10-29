'''
Name - Aryan Shrivastava
Roll no. - 2311041
Assignment - 15: Shooting method and PDE
'''

from mylibrary1 import *

#Question 1
def f_x(v,x,t):
    return v

def f_v(v,x,t):
    a = 0.01
    x_a = 20
    return -a*(x_a - x) 

x_b = 200

x0 = 40
v0 = 10 #guess 
h = 0.1 #step size
t0 = 0

X,V,T = ODE_coupled_2(f_x,f_v,x0,v0,t0,h,0,10)
print(X[-1])

x1 = 40
v1 = 20 #guess 
h = 0.1 #step size
t0 = 0

X1,V1,T1 = ODE_coupled_2(f_x,f_v,x1,v1,t0,h,0,10)
print(X1[-1])



v0_guess = Lagrange_interpolation(v1,v0,X1[-1],X[-1],x_b)
print(v0_guess)

X2,V2,T2 = ODE_coupled_2(f_x,f_v,x1,v0_guess,t0,0.1,0,10)
print(X2[-1])

#It came out to be almost 200..

#plotting the results

plt.scatter(T,X, label="For lower guess")
plt.scatter(T1,X1, label="For Higher guess")
plt.scatter(T2,X2, label="Result of 1st interpolation")
plt.xlabel('Position (x)')
plt.ylabel('Temperature (T)')
plt.title("RK4 Approximation for BVP")
plt.legend()
plt.grid()
plt.show()

for i in range(len(X2)):
    if X2[i] < 100 and X2[i+1] > 100:
        x_left = X2[i]
        x_right = X2[i+1]
        t_left = T2[i]
        t_right = T2[i+1]

print('x_right:', x_right)
print('x_left:', x_left)
print('t_right:', t_right)
print('t_left:', t_left)

def straight_line_interpolate(y_high,y_low,x_high,x_low,y):
    x = x_low + (y - y_low)*(x_high - x_low)/(y_high - y_low)
    return x

print('straight line interpolated x for T=100:', straight_line_interpolate(x_right,x_left,t_right,t_left,100))

#Question 2


def g(x):
    if abs(x-1)<10**(-6):
        return 300
    if x != 1:
        return 0
    

V, X, T = PDE_HeatEqn_Solve(g,0,2,0,2,0.1,0.002)
V0 = V[0]
V1 = V[50]
V2 = V[200]
V3 = V[-1]
V4 = V[5]

plt.plot(X,V0, label="PDE at t=0")
plt.plot(X,V4, label="PDE at t=0.01")
plt.plot(X,V1, label="PDE at t=0.1")
plt.plot(X,V2, label="PDE at t=1")
plt.plot(X,V3, label="PDE at t=2")
plt.xlabel('Position (x)')
plt.ylabel('Temperature (T)')
plt.title("PDE solution evolution with time")
plt.legend()
plt.grid()
plt.show()

'''
168.25886880474943
285.6817351061639
12.703147367717442
199.99999999999991
x_right: 101.14233074467893
x_left: 99.6533916533454
t_right: 4.5
t_left: 4.4
straight line interpolated x for T=100: 4.423278880155143
'''
















