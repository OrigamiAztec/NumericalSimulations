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

for n in range(1, 7):
    start_time = time.time()
    K = .5
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
    #print(temp_outputs)

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
    Z = np.zeros([(2**n+1), (2**n+1)])

    for num in range(1, len(Z)-1):
        #setting last row to boundary conditions
        Z[len(Z)-1][num] = 100*np.sin(np.pi*num*delta_x)

    counter = 1
    for num in range(1, 2**n):
        for col in range(1, 2**n):
            Z[num][col] = temp_outputs[len(temp_outputs)-counter][0]
            counter += 1

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
        print("DeltaX:")
        print(delta_x)
        print("Beta value:")
        print(Beta)
    
    #print("Method took", time.time() - start_time, "seconds to run")

print(log_relative_error_list)
print(log_deltax_list)

plt.title(r'4th order FDM -$log_{10}$($\Delta$x) vs -$log_{10}$(Relative Error), K = ' + str(K))
plt.xlabel(r'$-log_{10}$($\Delta$x)')
plt.ylabel(r'$-log_{10}$(Relative Error)')
plt.plot(log_deltax_list, log_relative_error_list)
plt.show()
