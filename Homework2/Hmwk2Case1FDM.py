#Antonio Diaz UIN 327003625 TAMU 2022
#Numerical Simulations 430 
#Hmwk 1 graphing and plotting exat solutions

# -*- coding: utf-8 -*
import sys
import numpy as np
import matplotlib.pyplot as plt

#attempt to create more general method for smaller delta x
def case_1_produce_FDE_matrix(n):
    # creating 2**n - 1 rows / columns
    zero_array = np.zeros((2**n-1, 2**n-1))
    FDE_matrix = zero_array
    #print(zero_array)
    # alpha_sum const open to change
    alpha = 4
    delta_x = 1.0/(2**n)
    # second order k
    # k = 2+alpha**2*delta_x**2
    # fourth order k
    # k = 2+alpha**2*delta_x**2*(1 + delta_x**2/12*alpha**2)
    # sixth order k
    #k = 2+alpha**2*delta_x**2*(1 + delta_x**2/12*alpha**2+delta_x**4/360*alpha**4)
    # eighth order k
    k = 2+alpha**2*delta_x**2*(1 + delta_x**2/12*alpha**2+delta_x**4/360*alpha**4+delta_x**6/20160*alpha**6)
    # tenth order k
    #k = 2+alpha**2*delta_x**2*(1 + delta_x**2/12*alpha**2+delta_x**4/360*alpha**4+delta_x**6/2160*alpha**6+delta_x**8/1814400*alpha**8)
    print("K value used in matrix")
    print(k)
    case_1_row_triple = [-1, k, -1]

    FDE_matrix[0][0:2] = case_1_row_triple[1:3]

    for row_num in range(1, 2**n-2):
        FDE_matrix[row_num][row_num-1:row_num+2] = case_1_row_triple

    FDE_matrix[2**n-2][2**n-3:2**n-1] = case_1_row_triple[0:2]

    #print(FDE_matrix)
    return(FDE_matrix)


def analytical_sol_case1(n):
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

#creating resultant matrix

log_error_list = []
log_deltax_list = []

for n in range(6, 12):
    print('results for n = {:.2f}'.format(n))
    resultant_matrix = np.zeros((2**n-1,1))
    resultant_matrix[2**n-2,0] = 100
    #print(resultant_matrix)

    a_matrix_inverse = np.linalg.inv(case_1_produce_FDE_matrix(n))
    #print("FDE matrix:")
    #print(case_1_produce_FDE_matrix(n))
    solution_matrix = np.dot(a_matrix_inverse, resultant_matrix)
    pos_along_rod_list_cm = np.linspace(0, 1, num=2**n+1)
    temp_solutions_list = [0]

    for i in np.nditer(solution_matrix):
        temp_solutions_list.append(i.tolist())

    temp_solutions_list.append(100)

    print("Temp solutions")
    print(temp_solutions_list)

    halfway = len(temp_solutions_list)/2
    print('Delta X = 1/2^({:.2f})'.format(n))
    print(u'{:.12f} cm : T_FDM = {:.12f} \xb0C,  T_analytical = {:.12f} \xb0C, error = {:.12f}'.format(pos_along_rod_list_cm[halfway], temp_solutions_list[halfway], analytical_sol_case1(n)[(2**n+1)/2],  (temp_solutions_list[halfway]-analytical_sol_case1(n)[(2**n+1)/2])/analytical_sol_case1(n)[(2**n+1)/2]*100))

    log_deltax_list.append(1.0/2**n)
    print(log_deltax_list)
    log_error_list.append((-temp_solutions_list[halfway]+analytical_sol_case1(n)[(2**n+1)/2])/analytical_sol_case1(n)[(2**n+1)/2]*100)
    print(log_error_list)

log_deltax_list = np.log(log_deltax_list)
log_error_list = np.log(np.abs(log_error_list))

print("log(DeltaX):")
print(log_deltax_list)
print("log(Error):")
print(log_error_list)
plt.title("10th order FDM Convergence of Temperature at .5 cm")
plt.plot(log_deltax_list,log_error_list)
plt.xlabel('log DeltaX')
plt.ylabel('log Error')

plt.show()

"""
print("Case 1, 10th order FDM Results, DeltaX = 1/2**4:")
for i in range(0, len(temp_solutions_list)):
    percenterror = (temp_solutions_list[i]-analytical_sol_case1(n)[i])/(analytical_sol_case1(n)[i]+.000000001)
    print(u'{:.7f} cm : T_(FDM) = {:.7f} \xb0C,  T_analytical = {:.7f} \xb0C,  Error = {:.7f}'.format(pos_along_rod_list_cm[i], temp_solutions_list[i], analytical_sol_case1(n)[i], percenterror*100))

plt.title("10th order FDM Temp Solutions vs Analytical, DeltaX = 1/2**4")
plt.plot(np.linspace(0, 1, 2**n+1), analytical_sol_case1(n), 'r', np.linspace(0, 1, 2**n+1), temp_solutions_list, 'b')
plt.xlabel('Position Along Rod (cm)')
plt.ylabel('Temperature (C)')

plt.show()
"""