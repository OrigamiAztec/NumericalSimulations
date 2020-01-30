#Antonio Diaz UIN 327003625 TAMU 2022
#Numerical Simulations 430 
#Hmwk 1 Case 2 FDM Solutions with error, beta, and total heat loss
from __future__ import division

# -*- coding: utf-8 -*
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import sin, pi
import matplotlib.pyplot as plt

from richardsons import richardsons

#attempt to create more general method for smaller delta x
# input n divisions (2, 3, 4, ...), output left side FDE matrix
def case_2_produce_FDE_matrix(n):
    # creating 2**n rows / columns of 0 
    zero_array = np.zeros((2**n, 2**n))

    #redefining left side of matrix equation to inital zero array
    FDE_matrix = zero_array
   
    #variables for heat rod, alpha, deltaX, thermal conductivity, radius, h, cross section area
    alpha = 4
    delta_x = 1.0/(2**n)
    k=.5
    R = .1
    h = alpha**2*k*R/2
    A_cross_section = np.pi * R**2  

    #case 2 of heat transfer across rod with normal coefficient
    k_normal = 2+alpha**2*delta_x**2
    case_2_row_triple = [-1, k_normal, -1]

    #case 2 of heat transfer across rod with new coefficient
    k_prime = 2*delta_x*h/k+2+alpha**2*delta_x**2
   
    #setting first row of second case to new matrix pair
    FDE_matrix[0][0:2] = [k_prime, -2]

    #redefining each row in PDE_Matrix after first row to matrix to new matrix
    for row_num in range(1, 2**n-1):
        FDE_matrix[row_num][row_num-1:row_num+2] = case_2_row_triple

    #redefining last row of matrix to right two matrix values
    FDE_matrix[2**n-1][2**n-2:2**n] = case_2_row_triple[0:2]

    #print(FDE_matrix)
    # function returns matrix. e.g [ [-K',1, 0, 0], [-1,K, -1, 0], [0,-1,K,-1], [0, 0, -1, K] ]
    return(FDE_matrix)

#case 2 exact solution, input number of divisions for deltaX, output exact Temp solution
def analytical_sol_case2(n):
    # Bar Parameters ----------------------------------------------------------------------------------------------------------------------------------
    k = .5              # thermal conductivity of material
    R = .1              # cross section radius
    Ac = np.pi * R**2   # cross sectional area
    L = 1               # length

    # Case Parameters -----------------------------------------------------------------------------------------------------------------------------
    Tb = 0              # T(0), base temperature
    T_l = 100            # T(L), tip temperature
    Ta = 0              # ambient temperature

    # more variable definitions ----------------------------------------------------------------------------------------------------------------------------
    x = np.linspace(0, 1, 2**n+1)   
    a=4
    h = a**2 * k * R / 2
    C = h/(k*a)*(T_l/(h/(k*a)*np.sinh(a*L)+np.cosh(a*L))-Ta)
    D = T_l/(h/(k*a)*np.sinh(a*L)+np.cosh(a*L))-Ta
    T = C*np.sinh(a*x) + D*np.cosh(a*x) + Ta
    #adding all values to a list to output
    analytical_sol_list = []
    for i in np.nditer(T):
        analytical_sol_list.append(i.tolist())
    return analytical_sol_list


#creating resultant matrix, ex  [ 0 ] 
                               #[ 0 ]
                               #[ 0 ]
                               #[ 0 ]
                               #[ 100 ]
Q_approx_list = []

#output q_dot total, deltaX, T_approx, T_exact, percent error, beta value results for division of 2, 3, 4, 5, 6, 7,8 - (2^n outputs)
#for n in range (2, 9):
flux_array = []
total_beta_list = []
total_error_list = []
for n in range(2, 11):
    print('n = {:d}'.format(n))
    #n=2
    alpha = 4
    delta_x = 1.0/(2**n)
    k=.5
    R = .1
    h = alpha**2*k*R/2
    A_cross_section = np.pi * R**2   # cross sectional area
  

    resultant_matrix = np.zeros((2**n,1))
    resultant_matrix[2**n-1,0] = 100
    #print(resultant_matrix)

    a_matrix_inverse = np.linalg.inv(case_2_produce_FDE_matrix(n))

    solution_matrix = np.dot(a_matrix_inverse, resultant_matrix)
    pos_along_rod_list_cm = np.linspace(0, 1, num=2**n+1)
    temp_solutions_list = []

    for i in np.nditer(solution_matrix):
        temp_solutions_list.append(i.tolist())

    temp_solutions_list.append(100)

    #for i in range(0, len(temp_solutions_list)):
        #print(u'{:.5f} cm : {:.5f} \xb0C, T_analytical = {:.5f} \xb0C, error% = {:.5f}'.format(pos_along_rod_list_cm[i], temp_solutions_list[i], analytical_sol_case2(n)[i], (temp_solutions_list[i] - analytical_sol_case2(n)[i])/analytical_sol_case2(n)[i]*100))
    T_n = temp_solutions_list[len(temp_solutions_list)-1]
    T_n_min1 = temp_solutions_list[len(temp_solutions_list)-2]
    q_dot_env_approx = (-k*A_cross_section*(-T_n_min1/delta_x+T_n/delta_x + alpha**2*T_n*delta_x**2/(2*delta_x)))
    # not about this value, it is a value taken from Bradshaw report. This value is -6.283185307179588 from q_total variable in Hmwk1 python code, however the FDM values also seem to converge to -6.280376
    q_dot_env_exact = -6.280376
    Q_approx_list.append(temp_solutions_list[0])
    beta = np.absolute(np.log((analytical_sol_case2(2)[0] - Q_approx_list[len(Q_approx_list)-1])  / (analytical_sol_case2(2)[0] - Q_approx_list[len(Q_approx_list)-2])) / np.log(2))
    T_approx = temp_solutions_list[0]
    T_exact = analytical_sol_case2(2)[0]
    percent_error = (-analytical_sol_case2(2)[0] + temp_solutions_list[0])/analytical_sol_case2(2)[0]*100
    
    def simpson(a, b, n, input_array):
        sum = 0
        inc = (b - a) / n
        #print(inc)
        for k in range(n + 1):
            x = a + (k * inc)
            summand = input_array[k]
            if (k != 0) and (k != n):
                summand *= (2 + (2 * (k % 2)))
            sum += summand
        return ((b - a) / (3 * n)) * sum

    # Examples of use
    #print(simpson(0, 1, len(test_array)-1, test_array))a
    perimeter = 2*pi*R
    test_array = temp_solutions_list
    lateral_flux_val = h*perimeter*simpson(0, 1, len(test_array)-1, test_array)
    #print(lateral_flux_val)
    flux_array.append(lateral_flux_val)
    if n > 2:
        print('DeltaX : {:.6f}, Q_extrapolated : {:.6f}, Beta: {:.6f}'.format(delta_x, richardsons(flux_array)[0], richardsons(flux_array)[1]))
        total_beta_list.append(richardsons(flux_array)[1])
    
    print('DeltaX : {:.6f}, HeatFluxLateral : {:.6f}'.format(delta_x, lateral_flux_val))
    #print(type(lateral_flux_val))

    print('DeltaX : {:.6f}, T_approx(0) : {:.6f}, T_exact(0) : {:.6f},Percent Error : {:.6f}, q_dot_total_approx : {:.6f}, q_dot_tota_exact : {:.6f} , beta : {:.6f}'.format(delta_x,T_approx, T_exact, percent_error, q_dot_env_approx, q_dot_env_exact, beta))
    #print(np.linspace(0, 1, 2**n+1))
    #print(temp_solutions_list)
    total_error_list.append(percent_error)
    
    
    
delta_x_array = [1/(2**1), 1/(2**2), 1/(2**3), 1/(2**4), 1/(2**5), 1/(2**6), 1/(2**7), 1/(2**8)]
#print(total_error_list[1:9])
#print(total_beta_list[1:9])
plt.title("DeltaX vs Beta convergence for case 2")
plt.plot(delta_x_array[1:9], np.log(total_beta_list[1:9]))
plt.ylabel('Beta')
plt.xlabel('DeltaX')
plt.show()

