#running code for p = 5, running from deltaX =  2**1 to 2**12
# case 2

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

elements = 2**2 
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

#manual p = 2, delta_x = .5
delta_x  = .5
penality_factor = 10**20
hierarchicalTest = hierarchicalShape()

p1_error_array = []
p1_delta_array = []

alpha = 4

def outputAssembledPlot():    
    alpha = 4
    graphing_node_1 = np.linspace(delta_x*0, delta_x, 5000)
    k_11 = hierarchicalTest.k(1, 1, alpha, graphing_node_1)
    k_12 = hierarchicalTest.k(1, 2, alpha, graphing_node_1)
    k_13 = hierarchicalTest.k(1, 3, alpha, graphing_node_1)
    k_14 = hierarchicalTest.k(1, 4, alpha, graphing_node_1)
    k_15 = hierarchicalTest.k(1, 5, alpha, graphing_node_1)
    k_16 = hierarchicalTest.k(1, 6, alpha, graphing_node_1)
    k_21 = hierarchicalTest.k(2, 1, alpha, graphing_node_1)
    k_22 = hierarchicalTest.k(2, 2, alpha, graphing_node_1)
    k_23 = hierarchicalTest.k(2, 3, alpha, graphing_node_1)
    k_24 = hierarchicalTest.k(2, 4, alpha, graphing_node_1)
    k_25 = hierarchicalTest.k(2, 5, alpha, graphing_node_1)
    k_26 = hierarchicalTest.k(2, 6, alpha, graphing_node_1)
    k_31 = hierarchicalTest.k(3, 1, alpha, graphing_node_1)
    k_32 = hierarchicalTest.k(3, 2, alpha, graphing_node_1)
    k_33 = hierarchicalTest.k(3, 3, alpha, graphing_node_1)
    k_34 = hierarchicalTest.k(3, 4, alpha, graphing_node_1)
    k_35 = hierarchicalTest.k(3, 5, alpha, graphing_node_1)
    k_36 = hierarchicalTest.k(3, 6, alpha, graphing_node_1)
    k_41 = hierarchicalTest.k(4, 1, alpha, graphing_node_1)
    k_42 = hierarchicalTest.k(4, 2, alpha, graphing_node_1)
    k_43 = hierarchicalTest.k(4, 3, alpha, graphing_node_1)
    k_44 = hierarchicalTest.k(4, 4, alpha, graphing_node_1)
    k_45 = hierarchicalTest.k(4, 5, alpha, graphing_node_1)
    k_46 = hierarchicalTest.k(4, 6, alpha, graphing_node_1)
    k_51 = hierarchicalTest.k(5, 1, alpha, graphing_node_1)
    k_52 = hierarchicalTest.k(5, 2, alpha, graphing_node_1)
    k_53 = hierarchicalTest.k(5, 3, alpha, graphing_node_1)
    k_54 = hierarchicalTest.k(5, 4, alpha, graphing_node_1)
    k_55 = hierarchicalTest.k(5, 5, alpha, graphing_node_1)
    k_56 = hierarchicalTest.k(5, 6, alpha, graphing_node_1)
    k_61 = hierarchicalTest.k(6, 1, alpha, graphing_node_1)
    k_62 = hierarchicalTest.k(6, 2, alpha, graphing_node_1)
    k_63 = hierarchicalTest.k(6, 3, alpha, graphing_node_1)
    k_64 = hierarchicalTest.k(6, 4, alpha, graphing_node_1)
    k_65 = hierarchicalTest.k(6, 5, alpha, graphing_node_1)
    k_66 = hierarchicalTest.k(6, 6, alpha, graphing_node_1)
    
    A = np.zeros((2, 2))
    A[0][0] = k_11
    A[0][1] = k_12
    A[1][0] = k_21
    A[1][1] = k_22

    B = np.zeros((2, 4))
    B[0][0] = k_13
    B[0][1] = k_14
    B[0][2] = k_15
    B[0][3] = k_16
    B[1][0] = k_23
    B[1][1] = k_24
    B[1][2] = k_25
    B[1][3] = k_26

    D = np.zeros((4, 4))
    D[0][0] = k_33
    D[0][1] = k_34
    D[0][2] = k_35
    D[0][3] = k_36
    D[1][0] = k_43
    D[1][1] = k_44
    D[1][2] = k_45
    D[1][3] = k_46
    D[2][0] = k_53
    D[2][1] = k_54
    D[2][2] = k_55
    D[2][3] = k_56
    D[3][0] = k_63
    D[3][1] = k_64
    D[3][2] = k_65
    D[3][3] = k_66

    C = np.zeros((4, 2))
    C[0][0] = k_31
    C[0][1] = k_32
    C[1][0] = k_41
    C[1][1] = k_42
    C[2][0] = k_51
    C[2][1] = k_52
    C[3][0] = k_61
    C[3][1] = k_62

    #the matrix that will be assembled:
    K = np.subtract(A, np.linalg.multi_dot([B, np.linalg.inv(D), C]))

    k_11_bar = K[0,0]
    k_12_bar = K[0, 1]
    k_21_bar = K[1, 0]
    k_22_bar = K[1, 1]

    penality_factor = 10**20
    global_matrix_dim = number_of_elements + 1
    #print(global_matrix_dim)
    global_test_1 = np.zeros((global_matrix_dim, global_matrix_dim))

    k = .5                         # hermal conductivity of material
    R = .1                         # radius
    h = alpha**2 * k * R / 2       # heat transfer coefficient 

    global_test_1[0][0] = k_11_bar + h/k
    global_test_1[0][1] = k_12_bar

    row_start = 0
    for num in range(1, number_of_elements):
        global_test_1[num][row_start] = k_21_bar
        global_test_1[num][row_start+1] = k_22_bar + k_11_bar
        global_test_1[num][row_start+2] = k_12_bar
        row_start += 1

    global_test_1[number_of_elements][number_of_elements-1] = k_21_bar
    global_test_1[number_of_elements][number_of_elements] = k_22_bar + penality_factor

    resultant_matrix = np.zeros((global_matrix_dim, 1))
    resultant_matrix[-1][-1] = 100*penality_factor

    #print(global_test_1)
    #print(resultant_matrix)

    temp_outputs = np.linalg.solve(global_test_1,resultant_matrix)
    #print(temp_outputs)
    middle = len(analytical_sol_case1(6))/2
    true_val = analytical_sol_case1(6)[int(middle)]
    error = (temp_outputs[len(temp_outputs)/2]) - true_val
    p1_error_array.append(error)
    p1_delta_array.append(delta_x)

    plt.plot(np.linspace(0, 1, len(temp_outputs)), temp_outputs,  label = r'$\Delta x = {:.4f}$'.format(delta_x))

for n in range(1, 13):
    delta_x  = 1.0/(2**n)
    graphing_node_1 = np.linspace(delta_x*0, delta_x, 5000)
    number_of_elements = 2**n
    outputAssembledPlot()

p1_delta_array = -np.log(np.abs(p1_delta_array))
p1_error_array = -np.log(np.abs(p1_error_array))

print("log error values:")
for num in p1_error_array:
    print(num)
print("log delta values:")
for num in p1_delta_array:
    print(num)

plt.title("p=5, Case 2, Temperature FEM output with increasing number of nodes:")
plt.plot(np.linspace(0, 1, len(analytical_sol_case1(14))), analytical_sol_case1(14), '--', label = "true")
plt.ylabel(u'T(x)\xb0C')
plt.xlabel("x pos along rod")
plt.legend()
plt.show()

plt.title("p=5, case 2, -log(deltaX) vs -log(error)")
plt.plot(p1_delta_array, p1_error_array,'--')
plt.xlabel(r'-log($\Delta$ x)')
plt.ylabel("log(error)")
plt.legend()
plt.show()
