'''
Name - Aryan Shrivastava
Roll no. - 2311041
Assignment - 10: Numerical Integration using Midpoint and Trapezoidal method
'''

from mylibrary1 import *
import numpy as np

def f1(x):
    return 1/x 

def f2(x):
    return x*np.cos(x)

def f3(x):
    return x*np.arctan(x)

#Table for outputs 

print("The output table is follows:" )
print("--------------------Table for function 1------------------------")
print("N______________Midpoint__________________Trapezoidal")
print("4______________",Midpoint_int(f1, 1,2,4),"___________",Trapezoidal_int(f1,1,2,4))
print("8______________",Midpoint_int(f1, 1,2,8),"___________",Trapezoidal_int(f1,1,2,8))
print("15______________",Midpoint_int(f1, 1,2,15),"___________",Trapezoidal_int(f1,1,2,15))
print("20______________",Midpoint_int(f1, 1,2,20),"___________",Trapezoidal_int(f1,1,2,20))

print("")

print("--------------------Table for function 2------------------------")
print("N______________Midpoint__________________Trapezoidal")
print("4______________",Midpoint_int(f2,0,(math.pi)/2, 4 ),"___________",Trapezoidal_int(f2,0,(math.pi)/2, 4))
print("8______________",Midpoint_int(f2,0,(math.pi)/2, 8 ),"___________",Trapezoidal_int(f2,0,(math.pi)/2, 8))
print("15______________",Midpoint_int(f2,0,(math.pi)/2, 15 ),"___________",Trapezoidal_int(f2,0,(math.pi)/2, 15))
print("20______________",Midpoint_int(f2,0,(math.pi)/2, 20 ),"___________",Trapezoidal_int(f2,0,(math.pi)/2, 20))

print("")

print("--------------------Table for function 3------------------------")
print("N______________Midpoint__________________Trapezoidal")
print("4______________",Midpoint_int(f3, 0, 1, 4),"___________",Trapezoidal_int(f3, 0,1,4))
print("8______________",Midpoint_int(f3, 0, 1, 8),"___________",Trapezoidal_int(f3, 0,1,8))
print("15______________",Midpoint_int(f3, 0, 1, 15),"___________",Trapezoidal_int(f3, 0,1,15))
print("20______________",Midpoint_int(f3, 0, 1, 20),"___________",Trapezoidal_int(f3, 0,1,20))


