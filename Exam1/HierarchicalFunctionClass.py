#testing polynomial quadrature and hierarchical basis functions 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

#analytical solution from past homework:
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
    C = (T_l - Ta - (Tb-Ta)*np.cosh(a*L))/np.sinh(a*L)
    D = 0 
    T = C*np.sinh(a*x) + D*np.cosh(a*x) + Ta
    analytical_sol_list = []
    for i in np.nditer(T):
        analytical_sol_list.append(i.tolist())
    return analytical_sol_list

elements = 2
delta_x = 1.0/(2**elements)
#hierarchical shape functions for evalutaing function,  derivatives, and stiffness coefficients.
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
    
    def k(self, m, n, alpha, xi):
        """
        make sure inputs for m and n are integers
        k_mn = integral of (psi_m_prime*psi_n_prime + alpha**2*psi_m*psi_n) dx over values of x
        """
        return integrate.simps(self.derivative_psi[m](xi) * self.derivative_psi[n](xi) + alpha**2*self.psi[m](xi)*self.psi[n](xi), xi)

hierarchicalTest = hierarchicalShape()

# plotting psi functions over range to deltaX
graphing_x = np.linspace(0, .25, 300)

#plt.plot(graphing_x, hierarchicalTest.eval(1, graphing_x), label  = r'$\psi_1$')
plt.plot(graphing_x, hierarchicalTest.ddx(1, graphing_x), label  = r'$\psi\backprime_1$')
#plt.plot(graphing_x, hierarchicalTest.eval(2, graphing_x), label  = r'$\psi_2$')
plt.plot(graphing_x, hierarchicalTest.ddx(2, graphing_x), label  = r'$\psi\backprime_2$')
#plt.plot(graphing_x, hierarchicalTest.eval(3, graphing_x), label  = r'$\psi_3$')
plt.plot(graphing_x, hierarchicalTest.ddx(3, graphing_x), label  = r'$\psi\backprime_3$')
#plt.plot(graphing_x, hierarchicalTest.eval(4, graphing_x), label  = r'$\psi_4$')
plt.plot(graphing_x, hierarchicalTest.ddx(4, graphing_x), label  = r'$\psi\backprime_4$')
#plt.plot(graphing_x, hierarchicalTest.eval(5, graphing_x), label  = r'$\psi_5$')
plt.plot(graphing_x, hierarchicalTest.ddx(5, graphing_x), label  = r'$\psi\backprime_5$')
#plt.plot(graphing_x, hierarchicalTest.eval(6, graphing_x), label  = r'$\psi_6$')
plt.plot(graphing_x, hierarchicalTest.ddx(6, graphing_x), label  = r'$\psi\backprime_6$')

plt.title(r"$\psi\backprime(x)$ from 0 to $\Delta x$")
plt.ylabel(r'$\psi\backprime(x)$')
plt.xlabel("x")
plt.legend()
plt.show()

