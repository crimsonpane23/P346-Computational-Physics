#Name - Aryan Shrivastava
#Roll no. - 2311041
#Assignment 0: Question 4

from mylibrary import *

#Given Complex numbers
c1 = MyComplex(1.3,-2.2)
c2 = MyComplex(-0.8,1.7)

#Adding them
print("The sum of given complex numbers is:")
c3 = MyComplex(0,0)
c3.add_complex(c1,c2)
c3.display_complex()

#Subtracting them
print("The difference of given complex numbers is:")
c4 = MyComplex(0,0)
c4.sub_complex(c1,c2)
c4.display_complex()

#Product 
print("The product of given complex numbers is:")
c5 = MyComplex(0,0)
c5.mul_complex(c1,c2)
c5.display_complex()

#Modulus
print(f"Modulus of c1: {c1.mod_complex():.3f}")
print(f"Modulus of c2: {c2.mod_complex():.3f}")

'''
Output:
The sum of given complex numbers is:
0.5, -0.5000000000000002i
The difference of given complex numbers is:
2.1, -3.9000000000000004i
The product of given complex numbers is:
2.7, 3.97i
Modulus of c1: 2.555
Modulus of c2: 1.879
'''