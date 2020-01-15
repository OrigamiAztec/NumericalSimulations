import numpy as np
import matplotlib.pyplot as plt

# Bar Parameters ----------------------------------------------------------------------------------------------------------------------------------
k = .5              # thermal conductivity of material
R = .1              # cross section radius
Ac = np.pi * R**2   # cross sectional area
L = 1               # length

# Case Parameters -----------------------------------------------------------------------------------------------------------------------------
Tb = 0              # T(0), base temperature
Tl = 100            # T(L), tip temperature
Ta = 0              # ambient temperature

# Processing & Output ----------------------------------------------------------------------------------------------------------------------------
x = np.linspace(0, L)
qdot = []
i = 0
for a in [0.25, 0.5, 1, 2, 4, 8]:
    i += 1
    h = a**2 * k * R / 2
    for case in [1, 2]:
        if case == 1:
            C = (Tl - Ta - (Tb-Ta)*np.cosh(a*L))/np.sinh(a*L)
            D = 0
        elif case == 2:
            C = h/(k*a)*(Tl/(h/(k*a)*np.sinh(a*L)+np.cosh(a*L))-Ta)
            D = Tl/(h/(k*a)*np.sinh(a*L)+np.cosh(a*L))-Ta
        T = C*np.sinh(a*x) + D*np.cosh(a*x) + Ta
        plt.subplot(2, 3, i)
        plt.plot(x, T, label="Case %i" % case)
        plt.xlabel("Position (x)")
        plt.ylabel("Temperature (T)")
        plt.title("[alpha=%i]" % a)
        plt.legend()
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.grid(True)
    qdot.append(-k*Ac*a*(C*np.sinh(a*L) + D*np.cosh(a*L)))
print(qdot)
plt.suptitle("Temperature vs Position, T(x)")
plt.show()