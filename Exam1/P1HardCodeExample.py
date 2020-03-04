#Hardcoding FEM p = 1 and analyzing outputs
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

# input n to function later
n = 1
# creating 2**n - 1 rows / columns
zero_array = np.zeros((2**n, 2**n))
stiffness_matrix = zero_array
print(stiffness_matrix)
# alpha_sum const open to change
alpha = 4
delta_x = 1.0/(2**n)

stiffness_matrix[0][0] = 1/delta_x  + alpha**2*delta_x/3
stiffness_matrix[0][1] = -1/delta_x  + alpha**2*delta_x/6
stiffness_matrix[1][0] = -1/delta_x  + alpha**2*delta_x/6
stiffness_matrix[1][1] = 1/delta_x  + alpha**2*delta_x/3
print(stiffness_matrix)

resultant_matrix = np.zeros((2**n, 1))
resultant_matrix[0][0] = 50
resultant_matrix[1][0] = 100

print(resultant_matrix)

a_matrix_inverse = np.linalg.inv(stiffness_matrix)
solution_matrix = np.dot(a_matrix_inverse, resultant_matrix)
print(solution_matrix)

