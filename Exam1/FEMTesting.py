#Antonio Diaz UIN 327003625 TAMU 2022
#Numerical Simulations 430 
#Exam 1 graphing and reporting data for FEM results

import matplotlib.pyplot as plt
import numpy as np

# Quadratic Lagrange Shape functions
class LagrangeShapeFnc(object):
    # Defined on local interval [0, 1], midpoint at .5
    def __init__(self):
        # creating array Phi and derivative of Phi 
        self.phi = [lambda x_i: 2 * (x_i - .5) * (x_i - 1.0), lambda x_i: 4.0 * x_i * (1.0 - x_i), lambda x_i: 2.0 * x_i * (x_i - 0.5)]
        # d(phi) / d(x_i)  
        self.derivative_phi = [lambda x_i: 2.0 * (x_i-0.5) + 2.0*(x_i - 1.0), lambda x_i: -4.0 * x_i + 4.0*(1.0 - x_i), lambda x_i: 2.0 * x_i + 2.0*(x_i - 0.5)]
        self.number_nodes = 2

    def eval(self,n,xi):
        """
        the function phi[n](xi), for any xi
        """
        return self.phi[n](xi)

    def ddx(self,n,xi):
        """
        the function dphi[n](xi), for any xi
        """
        return self.derivative_phi[n](xi)
    
    def size(self):
        """
        the number of points
        """
        return self.number_nodes

class Mesh(object):
    # N is number of elements, a is left endpoint, b is right endpoint 
    def __init__(self, N, a, b):
        self.N = N
        self.a = a
        self.b = b
    
    def coordinates(self):
        return self.N
    
    def cells(self, a, b):
        return (self.a - self.b)/self.N
    
    def size(self, N):
        return self.a*self.b


N=5
rightpt = 5.0
print("FEM1D.py Test case, dx=",rightpt/N)
mesh = Mesh(N, 0.0, rightpt)
coords = mesh.coordinates()
print("mesh.coordinates()=",coords)
sfns = LagrangeShapeFnc()
print("sfns.size()-3=", sfns.size()-3)
xi = np.linspace(0,1,100)
for n in range(3):
    plt.plot(xi,sfns.eval(n,xi))
    plt.show()
    plt.plot(xi,sfns.ddx(n,xi))
    plt.show()

