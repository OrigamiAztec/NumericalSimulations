# -*- coding: utf-8 -*
import sys
import numpy as np
import matplotlib.pyplot as plt
"""
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
"""

#attempt to create more general method for smaller delta x
def case_1_produce_FDE_matrix(n):
    # creating 2**n - 1 rows / columns
    zero_array = np.zeros((2**n-1, 2**n-1))
    FDE_matrix = zero_array
    #print(zero_array)
    # alpha_sum const open to change
    alpha = 4
    delta_x = 1.0/(2**n)
    alpha_sum_const = 2+alpha**2*delta_x**2
    case_1_row_triple = [1, -alpha_sum_const, 1]

    FDE_matrix[0][0:2] = case_1_row_triple[1:3]

    for row_num in range(1, 2**n-2):
        FDE_matrix[row_num][row_num-1:row_num+2] = case_1_row_triple

    FDE_matrix[2**n-2][2**n-3:2**n-1] = case_1_row_triple[0:2]

    print(FDE_matrix)
    return(FDE_matrix)

#creating resultant matrix
n=3
resultant_matrix = np.zeros((2**n-1,1))
resultant_matrix[2**n-2,0] = -100
#print(resultant_matrix)

a_matrix_inverse = np.linalg.inv(case_1_produce_FDE_matrix(n))

solution_matrix = np.dot(a_matrix_inverse, resultant_matrix)
pos_along_rod_list_cm = np.linspace(0, 1, num=2**n+1)
temp_solutions_list = [0]

for i in np.nditer(solution_matrix):
    temp_solutions_list.append(i.tolist())

temp_solutions_list.append(100)

def analytical_sol(n):
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
    x = np.linspace(0, 1, 2**n+1)   
    a=4
    h = a**2 * k * R / 2
    C = (T_l - Ta - (Tb-Ta)*np.cosh(a*L))/np.sinh(a*L)
    D = 0 
    T = C*np.sinh(a*x) + D*np.cosh(a*x) + Ta
    analytical_sol_list = []
    for i in np.nditer(T):
        analytical_sol_list.append(i.tolist())
    return analytical_sol_list

for i in range(0, len(temp_solutions_list)):
    print(u'{:.5f} cm : T_(FDM) = {:.5f} \xb0C, T_analytical = {:.5f} \xb0C'.format(pos_along_rod_list_cm[i], temp_solutions_list[i], analytical_sol(n)[i]))

print(analytical_sol(n))