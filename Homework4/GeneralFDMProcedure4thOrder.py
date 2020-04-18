import time

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix

log_error_list = []
log_relative_error_list = []
log_deltax_list = []
temp_extrapolated_array = []
percent_extrapolated_error = []
heat_flux_error_array = []
MMM_error_array = []

for n in range(1, 7):
    start_time = time.time()
    K = 1
    delta_x = (1/2)**n
    divisions=2**n

    def analyticalHeatFunction(x, y):
        return 100 * np.sinh(K*np.pi*y)*np.sin(np.pi*x) / np.sinh(K*np.pi)
    true_val = analyticalHeatFunction(.5, .5)
    #print("True:")
    #print(true_val)

    #A = (-2+4/12)*(K**2+1)
    A_const = 5/3 * (K**2 + 1)

    #B = K**2-2/12*(K**2+1)
    B_const = 1/6 * (K**2 + 1) - K**2

    #C = 1 - 2/12 * (K**2 + 1)
    C_const = 1/6 * (K**2 + 1) - 1

    #D = 1/12 * (K**2+1)
    D_const = -1/12 * (K**2 + 1)



    def assembledMatrix(n):
        ac = np.zeros([n-1, n-1])
        ai = np.zeros([n-1, n-1])
        az = np.zeros([n-1, n-1])
        ARow = np.zeros([n+1, n+1])
        for row in range(0, n-1):
            for col in range(0, n-1):  
                if row == col:
                    ac[row, col] = A_const
                    ai[row, col] = C_const
                elif abs(row - col) == 1:
                    ac[row, col] = B_const
                    ai[row, col] = D_const
        for i in range(0, n-1):
            for j in range(0, n-1):
                if (j == 0) and (i == j):
                    ARow = ac
                elif (j == 0) and (np.abs(i - j) == 1):
                    ARow = ai
                elif (j == 0) and (abs(i - j) > 1):
                    ARow = az
                elif i == j:
                    ARow =np.concatenate((ARow,ac))
                elif np.abs(i - j) == 1:
                    ARow = np.concatenate((ARow,ai))
                elif np.abs(i - j) > 1:
                    ARow = np.concatenate((ARow, az))
            if i == 0:
                A = ARow
            else:
                A = np.concatenate((A, ARow), axis = 1)
        return A

    num_of_unknowns = (2**n+1)**2-2**(2+n)
    resultant_matrix = np.zeros([num_of_unknowns, 1])
    for num in range(0, 2**n-1):
        resultant_matrix[num] = -C_const * 100 * np.sin(np.pi * (num+1) * delta_x) - D_const * 100 * np.sin(np.pi * (num) * delta_x) - D_const * 100 * np.sin(np.pi * (num+2) * delta_x)
    
    temp_outputs = np.linalg.solve(assembledMatrix(2**n), resultant_matrix)
    
    Z = np.zeros([(2**n+1), (2**n+1)])

    for num in range(1, len(Z)-1):
        #setting last row to boundary conditions
        Z[len(Z)-1][num] = 100*np.sin(np.pi*num*delta_x)

    counter = 1
    for num in range(1, 2**n):
        for col in range(1, 2**n):
            Z[num][col] = temp_outputs[len(temp_outputs)-counter][0]
            counter += 1

    #print("node outputs:")

    
    # Plotting the surface.
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.title(r'4th order FDM Output T(x, y)$\degree$ C, $\Delta$X = ' + str(delta_x))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel(r'T(x, y)$\degree$C')

    X = np.linspace(0, 1.0, 2**n + 1, endpoint = True)
    Y = np.linspace(0, 1.0, 2**n + 1, endpoint = True)
    X, Y = np.meshgrid(X, Y)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.inferno,linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(0, 100)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.show()


    approximate = temp_outputs[int((num_of_unknowns-1)/2)][0]
    #print("FDM Approx:")
    #print(approximate)
    log_error_list.append(true_val - approximate)
    
    log_deltax_list.append(-1*np.log(1.0/2**n))
    # print(log_deltax_list)
    log_relative_error_list.append(-1*np.log(np.abs(true_val - approximate)/approximate))

    if n > 1:
        #print("erro(Delx):")
        error_deltax = approximate - true_val

        #print("erro(Delx/2)")
        error_deltax_over_2 = -1*log_error_list[len(log_error_list)-2]

        Beta = -1/(np.log(2))*(np.log(error_deltax) - np.log(error_deltax_over_2))
        print("DeltaX:", delta_x)
        #print("Beta value:")
        #print(Beta)

        temp_derivative_approx_array = []

        # approximation of temperature gradients using forward difference.
        for col in range(0, len(Z)):
            # setting each temperature gradient calculation to be done by column. 
            col_checked = Z[:,col]
            # minus one taken into account for starting index of 0 in Python.
            max_index = len(col_checked)-1
            # forward difference fourth order = (-2*T_{DeltaX - 3} + 9*T_{DeltaX - 2} - 18 * T_{DeltaX - 1} + 11 * T_{DeltaX}) / {6*DeltaX} (DeltaX = DeltaY in this case)
            fourth_order_one_sided_difference = (-2*col_checked[max_index-3] + 9*col_checked[max_index-2] - 18*col_checked[max_index-1] + 11*col_checked[max_index])/(6*delta_x)
            # attaching value to list to use in integral. 
            temp_derivative_approx_array.append(fourth_order_one_sided_difference)
        #print("approximated temp gradient:")
        #print(temp_derivative_approx_array)

        # method to integrate temperature gradient array using 1/3 simpsons method. Returns double value 
        # simpson(lower_boundary, upper boundary, number of divisions, approximate temperature gradient array)
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

        #print("true_heatflux:")
        
        K_specific = .7
        true_heatflux = -200 * K_specific/ np.tanh(np.pi) 
        #print("True Heat Flux:", true_heatflux)

        heat_flux_estimate = - K_specific * simpson(0, 1, 2**n, temp_derivative_approx_array)
        #print("Heat Flux Estimate:", heat_flux_estimate)

        heat_flux_error_array.append(true_heatflux - heat_flux_estimate)

        if(len(heat_flux_error_array) > 1):
            error_deltax = heat_flux_estimate - true_heatflux
            #print("erro(Delx/2)")
            error_deltax_over_2 = -1*heat_flux_error_array[len(heat_flux_error_array)-2]
            # Beta = 1/log(2) * log(error at DeltaX) - log(error at DeltaX / 2)
            Beta = 1/(np.log(2))*(np.log(error_deltax) - np.log(error_deltax_over_2))
            #print("Heat Flux Beta:", Beta)

    #print(Z)
    temp_extrapolated_array.append(Z[int(len(Z)/2)][int(len(Z)/2)])
    #print(temp_extrapolated_array)

    if(len(temp_extrapolated_array) > 2):
        Q_delx_div_1 = temp_extrapolated_array[len(temp_extrapolated_array) - 1]
        Q_delx_div_2 = temp_extrapolated_array[len(temp_extrapolated_array) - 2]
        Q_delx_div_4 = temp_extrapolated_array[len(temp_extrapolated_array) - 3]
        Q_extrapolated = (Q_delx_div_2**2 - Q_delx_div_1*Q_delx_div_4)/(2*Q_delx_div_2+Q_delx_div_1+Q_delx_div_4)
        extrapolated_error = (Q_extrapolated - approximate)/Q_extrapolated
        Beta_extrapolated = np.log(np.abs((Q_extrapolated - Q_delx_div_1)/(Q_delx_div_1**2-Q_delx_div_2)))/np.log(2)
        #print("Beta Extrapolated:", Beta_extrapolated)
        #print("% Extrapolated Error: ")
        percent_extrapolated_error.append(extrapolated_error)

    #print("Method of Manufactured Solutions Section:")
    def manufactured_soluution_output(x):
        return 100*x**2
    
    MMM_center = manufactured_soluution_output(.5)
    FDM_center = Z[int(len(Z)/2)][int(len(Z)/2)]
    print("MMM Output:", MMM_center)
    print("FDM T(.5,.5):", FDM_center)
    MMM_error_array.append(-FDM_center + MMM_center)
    #print(MMM_error_array)
    if(len(MMM_error_array) > 2):
        error_deltax = -FDM_center + MMM_center
        #print("erro(Delx/2)")
        error_deltax_over_2 = MMM_error_array[len(MMM_error_array)-2]
        # Beta = 1/log(2) * log(error at DeltaX) - log(error at DeltaX / 2)
        Beta = 1/(np.log(2))*(np.log(error_deltax) - np.log(error_deltax_over_2))
        print("MMS Beta:", Beta)


    print("---------------------------------------------------------")
    #print("Method took", time.time() - start_time, "seconds to run")

#print(log_relative_error_list)
#print(log_deltax_list)

plt.title(r'4th order FDM -$log_{10}$($\Delta$x) vs -$log_{10}$(Relative Error), K = ' + str(K))
plt.xlabel(r'$-log_{10}$($\Delta$x)')
plt.ylabel(r'$-log_{10}$(Relative Error)')
plt.plot(log_deltax_list, log_relative_error_list)
plt.show()

plt.title(r'4th order FDM -$log_{10}$($\Delta$x) vs -$log_{10}$($Error_{Extrapolated}$), K = ' + str(K))
plt.xlabel(r'$-log_{10}$($\Delta$x)')
plt.ylabel(r'$-log_{10}$(Etrapolated Error)')
log_relative_error_list = np.log(np.abs(percent_extrapolated_error))
plt.plot(log_deltax_list[2:], log_relative_error_list)
plt.show()