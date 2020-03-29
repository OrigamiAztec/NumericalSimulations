# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')
plt.title("Tempurature Distribution across x, y")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('T(x, y)\xb0C')

# Make data.
X = np.arange(0, 1.1, 0.01)
Y = np.arange(0, 1.1, 0.01)
X, Y = np.meshgrid(X, Y)
K = .5
#Z = np.cos(np.sqrt(X**2 + Y**2))
Z = 100 * np.sinh(K*np.pi*Y)*np.sin(np.pi*X) / np.sinh(K*np.pi)
print(Z)
print("Analytical outputs:")
print(X)
print(Y)
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

K = 2

def analyticalHeatFunction(x, y):
    return 100 * np.sinh(K*np.pi*y)*np.sin(np.pi*x) / np.sinh(K*np.pi)

X = np.arange(0, 1, 0.01)
Y = np.arange(0, 1, 0.01)
X, Y = np.meshgrid(X, Y)
#Z = np.cos(np.sqrt(X**2 + Y**2))
Z = analyticalHeatFunction(X, Y)
fig,ax=plt.subplots(1,1)
cp = ax.contour(X, Y, Z, 12)
fig.colorbar(cp) # Add a colorbar to a plot
ax.clabel(cp, inline=1, fontsize=10)
ax.set_title('Contours Plot Temperature across infinite plate T(x, y)\xb0C')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
plt.title("Tempurature Distribution across x, y")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('T(x, y)\xb0C')

print(analyticalHeatFunction(1-.125, .125))
