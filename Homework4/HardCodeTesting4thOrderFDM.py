from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import scipy
from scipy import sparse

n = 2
delta_x = (.5)**n
num_of_unknowns = (2**n+1)**2-2**(2+n)

K = 1.4

#A = (-2+4/12)*(K**2+1)
A = 5/3 * (K**2 + 1)

#B = K**2-2/12*(K**2+1)
B = 1/6 * (K**2 + 1) - K**2

#C = 1 - 2/12 * (K**2 + 1)
C = 1/6 * (K**2 + 1) - 1

#D = 1/12 * (K**2+1)
D = -1/12 * (K**2 + 1)

#print(A, B, C, D)

# hardcode test DeltaX = .25
global_matrix = np.zeros([9, 9])
global_matrix[0] = [A, B, 0, C, D, 0, 0, 0, 0]
global_matrix[1] = [B, A, B, D, C, D, 0, 0, 0]
global_matrix[2] = [0, B, A, 0, D, C, 0, 0, 0]
global_matrix[3] = [C, D, 0, A, B, 0, C, D, 0]
global_matrix[4] = [D, C, D, B, A, B, D, C, D]
global_matrix[5] = [0, D, C, 0, B, A, 0, D, C]
global_matrix[6] = [0, 0, 0, C, D, 0, A, B, 0]
global_matrix[7] = [0, 0, 0, D, C, D, B, A, B]
global_matrix[8] = [0, 0, 0, 0, D, C, 0, B, A]
print("Global Matrix:")
print(global_matrix[6])

resultant_matrix = np.zeros([9, 1])
resultant_matrix[0] = -C*100*np.sin(np.pi*.25) - D * 100*np.sin(np.pi*.5)
resultant_matrix[1] = -C*100*np.sin(np.pi*.5) - 2* D * 100*np.sin(np.pi*.25)
resultant_matrix[2] = -C*100*np.sin(np.pi*.75) - D * 100*np.sin(np.pi*.5)
#print(resultant_matrix)

temp_outputs = np.linalg.solve(global_matrix, resultant_matrix)
print(temp_outputs)


fig = plt.figure()
ax = fig.gca(projection='3d')
plt.title(r'4th order FDM Output T(x, y)$\degree$ C, $\Delta$X = ' + str(delta_x))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel(r'T(x, y)$\degree$C')

# Plot the surface.
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

def analyticalHeatFunction(x, y):
    return 100 * np.sinh(K*np.pi*y)*np.sin(np.pi*x) / np.sinh(K*np.pi)
true = analyticalHeatFunction(.5, .5)


#print("True:")
#print(true)
#print("Approx:")
#approximate = temp_outputs[int((num_of_unknowns-1)/2)][0]
#print(approximate)