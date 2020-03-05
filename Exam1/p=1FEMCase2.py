# testing p = 1 case 2 with FEM 
# running code for p = 1 with increasing number of elements for 2**1 

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import stats
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

#analytical solution function with degree n as argument:
def analytical_sol_case1(n):
    # Bar Parameters ----------------------------------------------------------------------------------------------------------------------------------
    k = .5              # thermal conductivity of material
    R = .1              # cross section radius
    Ac = np.pi * R**2   # cross sectional area
    L = 1               # length

    # Case Parameters -----------------------------------------------------------------------------------------------------------------------------
    Tb = 0              # T(0), base temperature
    T_l = 100            # T(L), tip temperature
    Ta = 0              # ambient temperature

    # Processing & Output ----------------------------------------------------------------------------------------------------------------------------
    x = np.linspace(0, 1, 2**n+1)   
    a=4
    h = a**2 * k * R / 2
    Tl = 100
    C = h/(k*a)*(Tl/(h/(k*a)*np.sinh(a*L)+np.cosh(a*L))-Ta)
    D = Tl/(h/(k*a)*np.sinh(a*L)+np.cosh(a*L))-Ta
    T = C*np.sinh(a*x) + D*np.cosh(a*x) + Ta
    analytical_sol_list = []
    for i in np.nditer(T):
        analytical_sol_list.append(i.tolist())
    return analytical_sol_list

alpha = 4

p1_error_array = []
p1_delta_array = []

def calculateAssembleOutput(delta_x, number_of_elements):
    node_section = 1
    graphing_node_1 = np.linspace(delta_x*(node_section-1), delta_x*node_section, 5000)

    p = 1
    global_matrix_dim = number_of_elements*p + 1
    global_matrix = np.zeros((global_matrix_dim, global_matrix_dim))
    #print(global_matrix_dim)
    #print(global_matrix)

    penality_factor = 10**20
    # Bar Parameters
    k = .5                         # hermal conductivity of material
    R = .1                         # radius
    h = alpha**2 * k * R / 2       # heat transfer coefficient 
    global_matrix[0][0] = hierarchicalTest.k(1, 1, alpha, graphing_node_1) + h/k
    global_matrix[0][1] = hierarchicalTest.k(1, 2, alpha, graphing_node_1)

    row_start = 0
    for num in range(1, number_of_elements):
        global_matrix[num][row_start] = hierarchicalTest.k(2, 1, alpha, graphing_node_1)
        global_matrix[num][row_start+1] = hierarchicalTest.k(2, 2, alpha, graphing_node_1) +  hierarchicalTest.k(1, 1, alpha, graphing_node_1)
        global_matrix[num][row_start+2] = hierarchicalTest.k(1, 2, alpha, graphing_node_1)
        row_start += 1

    global_matrix[number_of_elements][number_of_elements-1] = hierarchicalTest.k(2, 1, alpha, graphing_node_1)
    global_matrix[number_of_elements][number_of_elements] = hierarchicalTest.k(2, 2, alpha, graphing_node_1) + penality_factor
    #print(global_matrix)

    resultant_matrix = np.zeros((global_matrix_dim, 1))
    
    resultant_matrix[len(resultant_matrix)-1][0] = 100*penality_factor
    #print(resultant_matrix)

    temp_outputs = np.linalg.solve(global_matrix,resultant_matrix)
    #returning value of FEM at x = .5 to compare to analytical:
    middle = len(analytical_sol_case1(6))/2
    true_val = analytical_sol_case1(6)[int(middle)]
    error = (temp_outputs[len(temp_outputs)/2]) - true_val
    p1_error_array.append(error)
    p1_delta_array.append(delta_x)
    print("Estimated Temp at x = 0")
    print(temp_outputs[0])
    print("True temp")
    print(analytical_sol_case1(n)[0])

    plt.plot(np.linspace(0, 1, len(temp_outputs)), temp_outputs, label = r'$\Delta x = {:.6f}$'.format(delta_x))


for n in range(2, 10):
    delta_x = 1.0/(2**n)
    number_of_elements = 2**n
    calculateAssembleOutput(delta_x, number_of_elements)


plt.ylabel(u'T(x)\xb0C')
plt.xlabel("x pos along rod")
plt.plot(np.linspace(0, 1, len(analytical_sol_case1(14))), analytical_sol_case1(14), '--', label = "True")

plt.title("p=1, Case 2 Temperature FEM output with increasing number of nodes:")
plt.legend()
plt.show()

p1_delta_array = -np.log(np.abs(p1_delta_array))
p1_error_array = -np.log(np.abs(p1_error_array))
plt.plot(p1_delta_array, p1_error_array, '--')
print("log error values:")
for num in p1_error_array:
    print(str(num))
print("log delta values:")
for num in p1_delta_array:
    print(num)

plt.title("P =1 Case 2, -log(error) vs -log(DeltaX)")
plt.xlabel("-log(Delta X)")
plt.ylabel("-log(Error)")
plt.legend()
plt.show()
