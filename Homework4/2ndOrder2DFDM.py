#Hardcode for DeltaX = .25, .5
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# Delta X = (1/2)
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.title(r'2nd order FDM Output, $\Delta$ X = (1/2) T(x, y)$\degree$ C')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('T(x, y)\xb0C')


K = 1
X = np.arange(0, 1.1, .5)
Y = np.arange(0, 1.1, .5)
X, Y = np.meshgrid(X, Y)
K_const = 50/(K**2+1)
Z = np.array([[0, 0, 0],[0, K_const, 0], [0, 100, 0]])
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.magma,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(0, 100)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

fig,ax=plt.subplots(1,1)
cp = ax.contour(X, Y, Z, 10)
fig.colorbar(cp) # Add a colorbar to a plot
ax.clabel(cp, inline=1, fontsize=10)
ax.set_title(r'2nd order FDM Output, $\Delta$ X = (1/2) T(x, y)$\degree$ C')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()


# Delta X = (1/4)
K = 2
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.title(r'2nd order FDM Output, $\Delta$ X = (1/4) T(x, y)$\degree$ C')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel(r'T(x, y)$\degree$C')

X = np.arange(0, 1.1, .25)
Y = np.arange(0, 1.1, .25)
X, Y = np.meshgrid(X, Y)

K = 2
global_matrix = np.zeros([6, 6])

global_matrix[0][0] = -2*(K**2+1)
global_matrix[0][1] = K**2
global_matrix[0][2] = 1

global_matrix[0+1][0] = 2*K**2
global_matrix[0+1][0+1] = -2*(K**2+1)
global_matrix[0+1][0+3] = 1

global_matrix[0+2][0] = 1
global_matrix[0+2][0+2] = -2*(K**2+1)
global_matrix[0+2][0+3] = K**2
global_matrix[0+2][0+4] = 1

global_matrix[0+3][0+1] = 1
global_matrix[0+3][0+2] = 2*K**2
global_matrix[0+3][0+3] = -2*(K**2+1)
global_matrix[0+3][1+4] = 1

global_matrix[0+4][0+2] = 1
global_matrix[0+4][0+4] = -2*(K**2+1)
global_matrix[0+4][1+4] = K**2

global_matrix[0+5][0+3] = 1
global_matrix[0+5][0+4] = 2*K**2
global_matrix[0+5][0+5] = -2*(K**2+1)

print("global matrix")
print(global_matrix)

resultant_matrix = np.zeros((6, 1))
resultant_matrix[0][0] = -100*np.sin(np.pi*.25)
resultant_matrix[1][0] = -100*np.sin(np.pi*.5)
print("Resultant matrix")
print(resultant_matrix)

temp_outputs = np.linalg.solve(global_matrix, resultant_matrix)
print("temp outputs")
print(temp_outputs)

#reformatting for numpy to plot in polygons
Z = np.zeros([5, 5])
Z[4][1] = 100*np.sin(np.pi*.25)
Z[4][2] = 100*np.sin(np.pi*.5)
Z[4][3] = 100*np.sin(np.pi*.75)

Z[1][1] = temp_outputs[4][0]
Z[2][1] = temp_outputs[2][0]
Z[3][1] = temp_outputs[0][0]

Z[1][2] = temp_outputs[5][0]
Z[2][2] = temp_outputs[3][0]
Z[3][2] = temp_outputs[1][0]

Z[1][3] = Z[1][1]
Z[2][3] = Z[2][1]
Z[3][3] = Z[3][1]

print("X meshgrid:")
print(X)
print("Y meshgrid:")
print(Y)
print("Z meshgrid:")
print(Z)
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.inferno,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(0, 100)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

fig,ax=plt.subplots(1,1)
cp = ax.contour(X, Y, Z, 10)
fig.colorbar(cp) # Add a colorbar to a plot
ax.clabel(cp, inline=1, fontsize=10)
ax.set_title(r'2nd order FDM Output, $\Delta$ X = (1/2) T(x, y)$\degree$ C')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()