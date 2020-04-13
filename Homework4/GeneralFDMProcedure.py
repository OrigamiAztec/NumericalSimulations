#2nd Order General FDM
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import scipy
from scipy import sparse

K = 2

def analyticalHeatFunction(x, y):
    return 100 * np.sinh(K*np.pi*y)*np.sin(np.pi*x) / np.sinh(K*np.pi)

log_error_list = []
log_relative_error_list = []
log_deltax_list = []

for div in range(1, 8):
    true_val = analyticalHeatFunction(.5, .5)
    n=div
    delta_x = (1/2)**n
    divisions=2**n
    I = scipy.sparse.eye(divisions-1) # (n-1) x (n-1) identity matrix
    e = np.ones(divisions-1) # vector (1, 1, ..., 1) of length n-1
    e0 = np.ones(divisions-2) # vector (1, 1, ..., 1) of length n-2
    A = scipy.sparse.diags([K**2*e0, (-2*K**2-2)*e, K**2*e0], [-1, 0, 1]) # (n-1) x (n-1) tridiagonal matrix
    J = scipy.sparse.diags([e0, e0], [-1, 1]) # same with zeros on diagonal
    Lh = scipy.sparse.kronsum(A, J, format='csr') # the desired matrix
    A = Lh.toarray()
    #print("assembled matrix:")
    #print(delta_x)
    import sys
    np.set_printoptions(threshold=sys.maxsize)
   

    num_of_unknowns = (2**n+1)**2-2**(2+n)
    #print("num of unknowns:")
    #print(num_of_unknowns)
    resultant_matrix = np.zeros((num_of_unknowns, 1))

    for num in range(0, 2**n-1):
        resultant_matrix[num][0] = -100*np.sin(np.pi*(num+1)*delta_x)

    temp_outputs = np.linalg.solve(A, resultant_matrix)
    
    approximate = temp_outputs[int((num_of_unknowns-1)/2)][0]

    results_length = len(temp_outputs)
    log_error_list.append(true_val - approximate)


    
    if div > 1:
        #print("erro(Delx):")
        error_deltax = approximate - true_val

        #print("erro(Delx/2)")
        error_deltax_over_2 = -1*log_error_list[len(log_error_list)-2]

        Beta = -1/(np.log(2))*(np.log(error_deltax) - np.log(error_deltax_over_2))
        print("Beta value:")
        print(Beta)

    log_deltax_list.append(-1*np.log(1.0/2**n))
    # print(log_deltax_list)
    log_relative_error_list.append(-1*np.log(np.abs(true_val - approximate)/approximate))

    Z = np.zeros([(2**n+1), (2**n+1)])
    for num in range(1, len(Z)-1):
        #setting last row to boundary conditions
        Z[len(Z)-1][num] = 100*np.sin(np.pi*num*delta_x)

    counter = 1
    for num in range(1, 2**n):
        for col in range(1, 2**n):
            Z[num][col] = temp_outputs[len(temp_outputs)-counter][0]
            counter += 1

    #print("node outputs")
    #print(Z)

    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.title(r'2nd order FDM Output T(x, y)$\degree$ C, $\Delta$X = ' + str(delta_x))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel(r'T(x, y)$\degree$C')

    # Plot the surface.
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
    print("Length of temp array:")
    print(results_length)
    """

#print(log_relative_error_list)
#print(log_deltax_list)
plt.title(r'2nd order FDM -$log_{10}$($\Delta$x) vs -$log_{10}$(Relative Error), K = ' + str(K))
plt.xlabel(r'$-log_{10}$($\Delta$x)')
plt.ylabel(r'$-log_{10}$(Relative Error)')
plt.plot(log_deltax_list, log_relative_error_list)
plt.show()