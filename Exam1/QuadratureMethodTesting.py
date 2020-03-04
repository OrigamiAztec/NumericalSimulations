#testing polynomial quadrature and hierarchical basis functions 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

"""
#returns sample points and weights for gaussian quadrature
deg = 2
print("sample points:")
print(np.polynomial.legendre.leggauss(deg)[0])
print("weights:")
print(np.polynomial.legendre.leggauss(deg)[1])
"""

elements = 2
delta_x = 1.0/(2**elements)

#hierarchical shape functions
class hierarchicalShape:
    # Defined on local interval [0, 1], midpoint at .5
    def __init__(self):
        # creating array psi and derivative of Phi 
        self.psi = [0, lambda x_i: 1 - 1/delta_x* x_i, lambda x_i: x_i / delta_x, lambda x_i: (x_i/delta_x) - (x_i**2/delta_x**2), lambda x_i: 2*x_i**3/(delta_x**3) - 3*x_i**2/(delta_x**2) + x_i/delta_x, lambda x_i: x_i**4/(delta_x**4)-2*x_i**3/(delta_x**3) + x_i**2/(delta_x**2), lambda x_i: x_i**2/(delta_x**2) - 4*x_i**3/(delta_x**3) + 5*x_i**4/(delta_x**4) - 2*x_i**5/(delta_x**5)]
        # d(psi) / d(x_i)  
        self.derivative_psi = [0, lambda x_i: -1/delta_x+(x_i*0), lambda x_i: 1/delta_x+(x_i*0), lambda x_i: 1/delta_x-2*x_i/(delta_x**2), lambda x_i: 6*x_i**2/(delta_x**3) - 6*x_i/(delta_x**2) + 1/delta_x, lambda x_i: 4*x_i**3/(delta_x**4) - 6*x_i**2/(delta_x**3)+ 2*x_i/(delta_x**2), lambda x_i: -10*x_i**4/(delta_x**5) + 20*x_i**3/(delta_x**4)-12*x_i**2/(delta_x**3)+2*x_i/(delta_x**2)]
        self.number_elements = elements

    def eval(self,n,xi):
        """
        the function phi[n](xi), for any xi
        """
        return self.psi[n](xi)

    def ddx(self,n,xi):
        """
        the function dphi[n](xi), for any xi
        """
        return self.derivative_psi[n](xi)
    
    #function for stiffness matrix coefficients
    def k(self, m, n, alpha, xi):
        """
        make sure inputs for m and n are integers
        k_mn = integral of (psi_m_prime*psi_n_prime + alpha**2*psi_m*psi_n) dx over values of x
        """
        return integrate.simps(self.derivative_psi[m](xi) * self.derivative_psi[n](xi) + alpha**2*self.psi[m](xi)*self.psi[n](xi), xi)

hierarchicalTest = hierarchicalShape()


alpha = 4

print("p = 1, delta X = .5, 2 elements, 3 nodes")
# p = 1, deltaX = .5
delta_x = .5
print(delta_x)

# element sections
node_section = 1
graphing_node_1 = np.linspace(delta_x*(node_section-1), delta_x*node_section, 5000)

node_section = 2
graphing_node_2 = np.linspace(delta_x*(node_section-1), delta_x*node_section, 5000)

global_matrix = np.zeros((3, 3))
print(global_matrix)

penality_factor = 10**20
global_matrix[0][0] = hierarchicalTest.k(1, 1, alpha, graphing_node_1) + penality_factor
global_matrix[0][1] = hierarchicalTest.k(1, 2, alpha, graphing_node_1)

global_matrix[1][0] = hierarchicalTest.k(2, 1, alpha, graphing_node_1)
global_matrix[1][1] = hierarchicalTest.k(2, 2, alpha, graphing_node_1) + hierarchicalTest.k(1, 1, alpha, graphing_node_1)
global_matrix[1][2] = hierarchicalTest.k(1, 2, alpha, graphing_node_1)

global_matrix[2][1] = hierarchicalTest.k(2, 1, alpha, graphing_node_1)
global_matrix[2][2] = hierarchicalTest.k(2, 2, alpha, graphing_node_1) + penality_factor 
print(global_matrix)

resultant_matrix = np.zeros((3, 1))
resultant_matrix[len(resultant_matrix)-1][0] = 100*penality_factor
print(resultant_matrix)

temp_outputs = np.linalg.solve(global_matrix,resultant_matrix)
print(temp_outputs)

for num in temp_outputs:
    print(num)

plt.title("p = 1, delta X = .5, 2 elements, 3 nodes")
plt.ylabel(u'T(x)\xb0C')
plt.xlabel("x pos along rod")
plt.plot(np.linspace(0, 1, 3), temp_outputs)
plt.show()



# attempt to create global matrix for p = 2

print("trying the 5x5:")
p = 2
number_of_elements = 2
global_matrix_dim = number_of_elements*p + 1
global_matrix = np.zeros((global_matrix_dim, global_matrix_dim))
print(global_matrix)
delta_x = .25
node_section = 1
graphing_node_1 = np.linspace(delta_x*(node_section-1), delta_x*node_section, 5000)

penality_factor = 10**20
global_matrix[0][0] = hierarchicalTest.k(1, 1, alpha, graphing_node_1) + penality_factor 
global_matrix[0][1] = hierarchicalTest.k(1, 3, alpha, graphing_node_1)
global_matrix[0][2] = hierarchicalTest.k(1, 2, alpha, graphing_node_1)

global_matrix[1][0] = hierarchicalTest.k(3, 1, alpha, graphing_node_1)
global_matrix[1][1] = hierarchicalTest.k(3, 3, alpha, graphing_node_1) 
global_matrix[1][2] = hierarchicalTest.k(2, 2, alpha, graphing_node_1)

global_matrix[2][0] = hierarchicalTest.k(2, 1, alpha, graphing_node_1)
global_matrix[2][1] = hierarchicalTest.k(2, 3, alpha, graphing_node_1)
global_matrix[2][2] = hierarchicalTest.k(2, 3, alpha, graphing_node_1) + hierarchicalTest.k(1, 1, alpha, graphing_node_2)
global_matrix[2][3] = hierarchicalTest.k(1, 3, alpha, graphing_node_1)
global_matrix[2][4] = hierarchicalTest.k(1, 2, alpha, graphing_node_1)

global_matrix[3][2] = hierarchicalTest.k(3, 1, alpha, graphing_node_1)
global_matrix[3][3] = hierarchicalTest.k(3, 3, alpha, graphing_node_1)
global_matrix[3][4] = hierarchicalTest.k(2, 2, alpha, graphing_node_1)

global_matrix[4][2] = hierarchicalTest.k(2, 1, alpha, graphing_node_1)
global_matrix[4][3] = hierarchicalTest.k(2, 3, alpha, graphing_node_1)
global_matrix[4][4] = hierarchicalTest.k(2, 3, alpha, graphing_node_1) + penality_factor

print(global_matrix)

resultant_matrix = np.zeros((global_matrix_dim, 1))
resultant_matrix[len(resultant_matrix)-1][0] = 100*penality_factor
print(resultant_matrix)

temp_outputs = np.linalg.solve(global_matrix,resultant_matrix)
print(temp_outputs)

for num in temp_outputs:
    print(num)


# attempt to create global matrix for p = 1
print("p=1, deltaX = .25, 4 elements, 5 nodes")
delta_x = .25
print("node section 1")
node_section = 1
graphing_node_1 = np.linspace(delta_x*(node_section-1), delta_x*node_section, 5000)
print(delta_x*(node_section-1))
print(delta_x*node_section)

print("node section 2")
node_section = 2
graphing_node_2 = np.linspace(delta_x*(node_section-1), delta_x*node_section, 5000)
print(delta_x*(node_section-1))
print(delta_x*node_section)


print("node section 3")
node_section = 3
graphing_node_3 = np.linspace(delta_x*(node_section-1), delta_x*node_section, 5000)
print(delta_x*(node_section-1))
print(delta_x*node_section)

print("node section 4")
node_section = 4
graphing_node_4 = np.linspace(delta_x*(node_section-1), delta_x*node_section, 5000)
print(delta_x*(node_section-1))
print(delta_x*node_section)
p = 1
number_of_elements = 4
global_matrix_dim = number_of_elements*p + 1
global_matrix = np.zeros((global_matrix_dim, global_matrix_dim))
print(global_matrix)

penality_factor = 10**20
global_matrix[0][0] = hierarchicalTest.k(1, 1, alpha, graphing_node_1) + penality_factor 
global_matrix[0][1] = hierarchicalTest.k(1, 2, alpha, graphing_node_1)

global_matrix[1][0] = hierarchicalTest.k(2, 1, alpha, graphing_node_1)
global_matrix[1][1] = hierarchicalTest.k(2, 2, alpha, graphing_node_1) +  hierarchicalTest.k(1, 1, alpha, graphing_node_1)
global_matrix[1][2] = hierarchicalTest.k(1, 2, alpha, graphing_node_1)

global_matrix[2][1] = hierarchicalTest.k(2, 1, alpha, graphing_node_1)
global_matrix[2][2] = hierarchicalTest.k(2, 2, alpha, graphing_node_1) +  hierarchicalTest.k(1, 1, alpha, graphing_node_1)
global_matrix[2][3] = hierarchicalTest.k(1, 2, alpha, graphing_node_1)

global_matrix[3][2] = hierarchicalTest.k(2, 1, alpha, graphing_node_1)
global_matrix[3][3] = hierarchicalTest.k(2, 2, alpha, graphing_node_1) +  hierarchicalTest.k(1, 1, alpha, graphing_node_1)
global_matrix[3][4] = hierarchicalTest.k(1, 2, alpha, graphing_node_1)

global_matrix[4][3] = hierarchicalTest.k(2, 1, alpha, graphing_node_1)
global_matrix[4][4] = hierarchicalTest.k(2, 2, alpha, graphing_node_1) + penality_factor

print(global_matrix)

resultant_matrix = np.zeros((global_matrix_dim, 1))
resultant_matrix[len(resultant_matrix)-1][0] = 100*penality_factor
print(resultant_matrix)

temp_outputs = np.linalg.solve(global_matrix,resultant_matrix)
print(temp_outputs)

for num in temp_outputs:
    print(num)

plt.title("p=1, deltaX = .25, 4 elements, 5 nodes")
plt.plot(np.linspace(0, 1, 5), temp_outputs)
plt.show()
