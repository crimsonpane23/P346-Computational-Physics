'''
Name - Aryan Shrivastava
Roll No. - 2311041
Assignment 14 - Runge Kutta 4
'''

from mylibrary1 import *
import numpy as np
import matplotlib.pyplot as plt
import math

def ODE_rk4(f, x0, y0, h, a, b):
    x = x0
    y = y0
    L_x = [x0]
    L_y = [y0]
    n = int(((b-a)/h))

    for i in range(n):
        k1 = h * f(y,x)
        k2 = h * f(y + k1/2, x + h/2)
        k3 = h * f(y + k2/2,x + h/2)
        k4 = h * f(y + k3, x + h)
        
        y += (k1 + 2*k2 + 2*k3 + k4) / 6
        x += h
        
        L_x.append(x)
        L_y.append(y)
    
    return L_x, L_y

def ODE_plot_data(X,Y):
    n = len(X)
    P = []
    for i in range(n):
        P.append((X[i], Y[i]))
    return P

def f1(y,x):
    return (x+y)**2

def y1(x):
    return math.tan(x + math.pi/4) - x

X, Y = ODE_rk4(f1, 0, 1, 0.1, 0, math.pi/5)
S = np.linspace(0, math.pi/5, 100)
print(ODE_plot_data(X,Y))

plt.plot(X,Y, label="RK4 solution  Approximation with step size h = 0.1")
plt.plot(S, [y1(x) for x in S], label="Analytic Solution", linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.title("RK4 solution vs Analytic Solution for dy/dx = (x+y)^2")
plt.legend()
plt.grid()
plt.show()

X, Y = ODE_rk4(f1, 0, 1, 0.25, 0, math.pi/5)
S = np.linspace(0, math.pi/5, 100)
print(ODE_plot_data(X,Y))

plt.plot(X,Y, label="RK4 solution  Approximation with step size h = 0.25")
plt.plot(S, [y1(x) for x in S], label="Analytic Solution", linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.title("RK4 solution vs Analytic Solution for dy/dx = (x+y)^2")
plt.legend()
plt.grid()
plt.show()

X, Y = ODE_rk4(f1, 0, 1, 0.45, 0, math.pi/5)
S = np.linspace(0, math.pi/5, 100)
print(ODE_plot_data(X,Y))

plt.plot(X,Y, label="RK4 solution  Approximation with step size h = 0.45")
plt.plot(S, [y1(x) for x in S], label="Analytic Solution", linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.title("RK4 solution vs Analytic Solution for dy/dx = (x+y)^2")
plt.legend()
plt.grid()
plt.show()

X1, Y1 = ODE_rk4(f1, 0, 1, 0.1, 0, math.pi/5)
X2, Y2 = ODE_rk4(f1, 0, 1, 0.25, 0, math.pi/5)
X3, Y3 = ODE_rk4(f1, 0, 1, 0.45, 0, math.pi/5)
S = np.linspace(0, math.pi/5, 100)
print(ODE_plot_data(X,Y))


plt.plot(X1,Y1, label="RK4 solution  Approximation with step size h = 0.1")
plt.plot(X2,Y2, label="RK4 solution  Approximation with step size h = 0.25")
plt.plot(X3,Y3, label="RK4 solution  Approximation with step size h = 0.45")
plt.plot(S, [y1(x) for x in S], label="Analytic Solution", linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.title("RK4 solution vs Analytic Solution for dy/dx = (x+y)^2")
plt.legend()
plt.grid()
plt.show()

#OUTPUT
'''
[(0, 1), (0.1, 1.1230489138367843), (0.2, 1.308496167191276), (0.30000000000000004, 1.5957541602334353), (0.4, 2.0648996869578835), (0.5, 2.907820425151802), (0.6, 4.7278968165905155)]
[(0, 1), (0.25, 1.43555804125693), (0.5, 2.8972272051287176)]
[(0, 1), (0.45, 2.3890595350788653)]
'''

#Question 2

#Ignore- -was trying n-coupled ODE
'''
def ODE_Coupled(f, x0, y0, h, a, b):
    #Here f, y, y0 are vectors (numpy arrays)
    #This is for a n-coupled first order ODE
    x = x0
    y = np.array(y0, dtype='float64')
    L_x = [x0]
    L_y = [y0]
    n = int(((b-a)/h))

    for i in range(n):
        k1 = np.array(h * f(y,x), dtype='float64')
        k2 = np.array(h * f(y + k1/2, x + h/2), dtype='float64')
        k3 = np.array(h * f(y + k2/2,x + h/2), dtype='float64')
        k4 = np.array(h * f(y + k3, x + h), dtype='float64')
        y += (k1 + 2*k2 + 2*k3 + k4) / 6
        x += h
        
        L_x.append(x)
        L_y.append(y)
    
    return L_x, L_y

def f_vec(y_vec, x):
    k = 1.0
    m = 1.0
    mu = 0.15   
    w = 1.0
    return np.array([-y_vec[1],-mu*y_vec[1] -(w**2)*x])
    
X,Y = ODE_Coupled(f_vec, 0, [1,0], 0.1, 0, 40)
print(np.shape(Y))

S_1 = np.linspace(0, math.pi/5, 100)

plt.plot(X,Y[0], label="RK4 solution for velocity  Approximation")
plt.plot(X,Y[1], label="RK4 solution for position Approximation")
plt.plot(S_1, [y1(x) for x in S_1], label="Analytic Solution", linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.title("RK4 solution vs Analytic Solution for SHO")
plt.legend()
plt.grid()
plt.show()
'''

def ODE_coupled_2(f_x,f_v,x0,v0,t0,h,a,b):
    v = v0
    x = x0
    x = x0
    t = t0
    L_x = [x0]
    L_v = [v0]
    L_t = [t0]
    n = int(((b-a)/h))

    for i in range(n):
        k1x = h*f_x(v,x,t)
        k1v = h*f_v(v,x,t)

        k2x = h*f_x(v + k1v/2, x+k1x/2, t+h/2)
        k2v = h*f_v(v + k1v/2, x+k1x/2, t+h/2)

        k3x = h*f_x(v + k2v/2, x+k2x/2, t+h/2)
        k3v = h*f_v(v + k2v/2, x+k2x/2, t+h/2)

        k4x = h*f_x(v + k3v/2, x+k3x/2, t+h/2)
        k4v = h*f_v(v + k3v/2, x+k3x/2, t+h/2)

        x += (k1x + 2*k2x + 2*k3x + k4x)/6
        v += (k1v + 2*k2v + 2*k3v + k4v)/6
        t += h
        L_x.append(x)
        L_v.append(v)
        L_t.append(t)

    return L_x, L_v, L_t

def f_x(v,x,t):
    return v

def f_v(v,x,t):
    k = 1.0
    m = 1.0
    mu = 0.15
    w = 1
    return -mu*v - (w**2)*x

X,V, T = ODE_coupled_2(f_x,f_v,1,0,0,0.1,0,40)

E = []
k = 1.0
m = 1.0
mu = 0.15
w = 1
for i in range(len(X)):
    e = 0.5*k*X[i]**2 + 0.5*m*V[i]**2
    E.append(e)


plt.plot(T,X, label="Postion vs Time")
plt.xlabel('Time (t)')
plt.ylabel('position (x)')
plt.title("RK4 Approximation for SHO")
plt.legend()
plt.grid()
plt.show()

plt.plot(T,V, label="Velocity vs Time")
plt.xlabel('Time (t)')
plt.ylabel('Velocity (v)')
plt.title("RK4 Approximation for SHO")
plt.legend()
plt.grid()
plt.show()

plt.plot(T,V, label="Velocity vs Time")
plt.plot(T,X, label="Position vs Time")
plt.xlabel('Time (t)')
plt.ylabel('y-axis')
plt.title("RK4 Approximation for SHO")
plt.legend()
plt.grid()
plt.show()

plt.plot(X,V, label="Velocity vs Postion")
plt.xlabel('Position (x)')
plt.ylabel('Velocity (v)')
plt.title("Phase space of SHO using RK4")
plt.legend()
plt.grid()
plt.show()

plt.plot(T,E, label="Energy vs Time")
plt.xlabel('Time (t)')
plt.ylabel('Energy (E)')
plt.title("RK4 Approximation for SHO")
plt.legend()
plt.grid()
plt.show()


plt.plot(X,E, label="Energy vs Position")
plt.xlabel('Position (x)')
plt.ylabel('Energy (E)')
plt.title("RK4 Approximation for SHO")
plt.legend()
plt.grid()
plt.show()

plt.plot(V,E, label="Energy vs Velocity")
plt.xlabel('Velocity (v)')
plt.ylabel('Energy (E)')
plt.title("RK4 Approximation for SHO")
plt.legend()
plt.grid()
plt.show()




