#Antonio Diaz UIN 327003625 TAMU 2022
#Numerical Simulations 430 
#Hmwk 1 Case 1 FDM Solutions 

# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt

# Bar Parameters ----------------------------------------------------------------------------------------------------------------------------------
k = .5              # thermal conductivity of material
R = .1              # cross section radius
Ac = np.pi * R**2   # cross sectional area
L = 1               # length

# Case Parameters -----------------------------------------------------------------------------------------------------------------------------
Tb = 0              # T(0), base temperature
T_l = 100            # T(L), tip temperature
Ta = 0              # ambient temperature

# Processing & Output ----------------------------------------------------------------------------------------------------------------------------
x = np.linspace(0, L)
qdot = []
i = 0
for a in [0.25, 0.5, 1, 2, 4, 8]:
    i += 1
    h = a**2 * k * R / 2
    print("h val")
    print(h)
    for case in [1, 2]:
        if case == 1:
            print("Beg case 1")
            C = (T_l - Ta - (Tb-Ta)*np.cosh(a*L))/np.sinh(a*L)
            D = 0
            print("End case 1")
        elif case == 2:
            print("Beg case 2")
            C = h/(k*a)*(T_l/(h/(k*a)*np.sinh(a*L)+np.cosh(a*L))-Ta)
            D = T_l/(h/(k*a)*np.sinh(a*L)+np.cosh(a*L))-Ta
            print("End case 2")
        T = C*np.sinh(a*x) + D*np.cosh(a*x) + Ta
        print("x val")
        print(x)
        print("Temp")
        print(T)
        #print("xval {}".format(x[12]))
        #print("temp {}".format(T[12]))
        plt.subplot(2, 3, i)
        plt.plot(x, T, label="Case %i" % case)
        plt.xlabel("Position (x)")
        plt.ylabel("Temperature (T)")
        plt.title("[alpha={:.2f}]".format(a))
        plt.legend()
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.grid(True)
    qdot.append(-k*Ac*a*(C*np.sinh(a*L) + D*np.cosh(a*L)))
#print(qdot)
plt.suptitle("Temperature vs Position, T(x)")
plt.show()

#code for initial finite difference model test, delta x = 1/(2**2) cm
a = np.matrix([[-3,1,-0],[1,-3,1],[0,1,-3]])
a_matrix_inverse = np.linalg.inv(a)
resultant_matrix = np.matrix([[0],[0],[-100]])
temp_solutions = np.dot(a_matrix_inverse,resultant_matrix)
pos_along_rod_list_cm = np.linspace(0, 1, num=2**2+1)
#print(pos_along_rod_list_cm)
#print(temp_solutions)
temp_solutions_list = [0]

for i in np.nditer(temp_solutions):
    temp_solutions_list.append(i.tolist())

temp_solutions_list.append(100)

for i in range(0, len(temp_solutions_list)):
    print(u'{:.5f} cm : {:.5f} \xb0C'.format(pos_along_rod_list_cm[i], temp_solutions_list[i]))
