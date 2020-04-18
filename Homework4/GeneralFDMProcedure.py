#2nd Order General FDM
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import scipy
from scipy import sparse
from scipy import integrate

K = 1

def analyticalHeatFunction(x, y):
    return 100 * np.sinh(K*np.pi*y)*np.sin(np.pi*x) / np.sinh(K*np.pi)


log_error_list = []
log_relative_error_list = []
log_deltax_list = []
temp_extrapolated_array = []
percent_extrapolated_error = []
heat_flux_error_array = []
MMM_error_array = []

for div in range(1, 8):
    true_val = analyticalHeatFunction(.5, .5)
    #print(true_val)

    # can replace number of divisions by degree
    n=div
    delta_x = (1/2)**n
    divisions=2**n

    # creating assembly matrix using other Python packages.
    I = scipy.sparse.eye(divisions-1) # (n-1) x (n-1) identity matrix
    e = np.ones(divisions-1) # vector (1, 1, ..., 1) of length n-1
    e0 = np.ones(divisions-2) # vector (1, 1, ..., 1) of length n-2
    A = scipy.sparse.diags([K**2*e0, (-2*K**2-2)*e, K**2*e0], [-1, 0, 1]) # (n-1) x (n-1) tridiagonal matrix
    J = scipy.sparse.diags([e0, e0], [-1, 1]) # same with zeros on diagonal
    Lh = scipy.sparse.kronsum(A, J, format='csr') # the desired matrix
    #print("assembled matrix:")
    A = Lh.toarray()
    
    # uncommenting the two lines below can be helpful in seeing larger matrices used in code.
    # import sys
    # np.set_printoptions(threshold=sys.maxsize)
   
    num_of_unknowns = (2**n+1)**2-2**(2+n)
    #print("num of unknowns:")
    #print(num_of_unknowns)

    # Right Hand Side Matrix set to dimensions of 0 x 0: 
    resultant_matrix = np.zeros((num_of_unknowns, 1))

    # Filling in boundary conditions. 
    for num in range(0, 2**n-1):
        resultant_matrix[num][0] = -100*np.sin(np.pi*(num+1)*delta_x)

    # using linear algebra included packages to multiply inverse of A by resultant matrix:
    temp_outputs = np.linalg.solve(A, resultant_matrix)
    
    # getting middle section of temp outputs to compare to true value later in code.
    approximate = temp_outputs[int(len(temp_outputs)/2)][0]

    results_length = len(temp_outputs)
    # attaching error to log error list. Raw error will eventually be used in graph.
    log_error_list.append(true_val - approximate)


    # calculating Beta for each iteration when n > 1
    if div > 1:
        #print("erro(Delx):")
        error_deltax = approximate - true_val
        #print("erro(Delx/2)")
        error_deltax_over_2 = -1*log_error_list[len(log_error_list)-2]
        # Beta = 1/log(2) * log(error at DeltaX) - log(error at DeltaX / 2)
        Beta = 1/(np.log(2))*(np.log(error_deltax) - np.log(error_deltax_over_2))
        #print("Beta value:", Beta)

    # converting deltaX list to absolute value log(DeltaX) for graph.
    log_deltax_list.append(-1*np.log(1.0/2**n))
    # converting relative error to absolute value log scale for graph.
    # print(log_deltax_list)
    log_relative_error_list.append(-1*np.log(np.abs(true_val - approximate)/approximate))

    # setting up matrix for graphing 3D graph to get a sense of values:
    Z = np.zeros([(2**n+1), (2**n+1)])
    for num in range(1, len(Z)-1):
        #setting last row to boundary conditions
        Z[len(Z)-1][num] = 100*np.sin(np.pi*num*delta_x)

    # annoyingly complex way to convert my list of temperature outputs to the nice format matplotlib needs. 😤
    counter = 1
    for num in range(1, 2**n):
        for col in range(1, 2**n):
            Z[num][col] = temp_outputs[len(temp_outputs)-counter][0]
            counter += 1
    #print("Temp matrix organized to follow node order, but upside down:")
    #print(Z)

    temp_derivative_approx_array = []
    # approximation of temperature gradients using forward difference.
    for col in range(0, len(Z)):
        # setting each temperature gradient calculation to be done by column. 
        col_checked = Z[:,col]
        # minus one taken into account for starting index of 0 in Python.
        max_index = len(col_checked)-1
        # forward difference second order = (T_{DeltaX - 2} - 4 * T_{DeltaX - 1} + 3 * T_{DeltaX}) / {2*DeltaX} (DeltaX = DeltaY in this case)
        second_order_one_sided_difference = (col_checked[max_index-2] - 4*col_checked[max_index-1] + 3*col_checked[max_index])/(2*delta_x)
        # attaching value to list to use in integral. 
        temp_derivative_approx_array.append(second_order_one_sided_difference)

    #method to integrate temperature gradient array using 1/3 simpsons method. Returns double value 
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

    print("DeltaX:", delta_x)

    K_specific = .7
    true_heatflux = -200 * K_specific / np.tanh(np.pi) 
    #print("True_heatflux", true_heatflux)

    heat_flux_estimate = -K_specific * simpson(0, 1, 2**n, temp_derivative_approx_array)
    #print("Heat Flux Estimate:", heat_flux_estimate)
    
    heat_flux_error_array.append(true_heatflux - heat_flux_estimate)

    if(len(heat_flux_error_array) > 1):
        error_deltax = heat_flux_estimate - true_heatflux
        #print("erro(Delx/2)")
        error_deltax_over_2 = -1*log_error_list[len(log_error_list)-2]
        # Beta = 1/log(2) * log(error at DeltaX) - log(error at DeltaX / 2)
        Beta = 1/(np.log(2))*(np.log(error_deltax) - np.log(error_deltax_over_2))
        #print("Heat Flux Beta:", Beta)

    #making array of estimated values from FDM at center of node matrix. Using T(.5, .5) to compare to extrapolated. 
    temp_extrapolated_array.append(Z[int(len(Z)/2)][int(len(Z)/2)])

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


    # Plot the surface.

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.title(r'2nd order FDM Output T(x, y)$\degree$ C, $\Delta$X = ' + str(delta_x))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel(r'T(x, y)$\degree$C')

    X = np.linspace(0, 1.0, 2**n + 1, endpoint = True)
    Y = np.linspace(0, 1.0, 2**n + 1, endpoint = True)
    X, Y = np.meshgrid(X, Y)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.inferno,
                        linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(0, 100)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.show()
    
    #print("Length of temp array:")
    #print(results_length)
    
    print("---------------------------------------------------------")
    
    

#print(log_relative_error_list)
#print(log_deltax_list)
#print(temp_extrapolated_array)
#print(percent_extrapolated_error)

plt.title(r'2nd order FDM -$log_{10}$($\Delta$x) vs -$log_{10}$(Relative Error), K = ' + str(K))
plt.xlabel(r'$-log_{10}$($\Delta$x)')
plt.ylabel(r'$-log_{10}$(Relative Error)')
plt.plot(log_deltax_list, log_relative_error_list)
plt.show()

plt.title(r'2nd order FDM -$log_{10}$($\Delta$x) vs -$log_{10}$($Error_{Extrapolated}$), K = ' + str(K))
plt.xlabel(r'$-log_{10}$($\Delta$x)')
plt.ylabel(r'$-log_{10}$(Etrapolated Error)')
log_relative_error_list = np.log(np.abs(percent_extrapolated_error))
plt.plot(log_deltax_list[2:], log_relative_error_list)
plt.show()