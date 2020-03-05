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