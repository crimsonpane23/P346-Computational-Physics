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

#_____________________Output_____________________________#
'''
The output table is follows:
--------------------Table for function 1------------------------
N______________Midpoint__________________Trapezoidal
4______________ 0.6912198912198912 ___________ 0.6970238095238095
8______________ 0.6926605540432034 ___________ 0.6941218503718504
15______________ 0.6930084263712957 ___________ 0.6934248043580644
20______________ 0.6930690982255869 ___________ 0.6933033817926942

--------------------Table for function 2------------------------
N______________Midpoint__________________Trapezoidal
4______________ 0.5874479167573121 ___________ 0.5376071275673586
8______________ 0.5749342733821311 ___________ 0.5625275221623353
15______________ 0.5719716590967575 ___________ 0.5684462350385162
20______________ 0.5714572867152204 ___________ 0.569474588169518

--------------------Table for function 3------------------------
N______________Midpoint__________________Trapezoidal
4______________ 0.2820460493571144 ___________ 0.2920983458939516
8______________ 0.2845610193056679 ___________ 0.28707219762553304
15______________ 0.28516010270349235 ___________ 0.2858742642174127
20______________ 0.28526426016144524 ___________ 0.285665963360493
'''
