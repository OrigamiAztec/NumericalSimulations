import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from richardsons import richardsons
from TDMAsolver import TDMAsolver as tridiag
from mpltools import annotation


def tableFormat(Q):
    return ['{:.6f}'.format(x) for x in Q]


def getError(exact, approx):
    return np.multiply(np.multiply(np.subtract(approx, exact), np.divide(1, exact)), 100)


def getBeta(exact, approx, nels):
    error = getError(exact, approx)
    betas = [(np.log(error[x-1])-np.log(error[x])) /
             (np.log(1/nels[x-1])-np.log(1/nels[x])) for x in range(9)]
    betas.insert(0, float('nan'))
    return betas


# Bar Parameters
k = .5                         # hermal conductivity of material
R = .1                         # radius
Ac = np.pi * R**2              # cross sectional area
L = 1                          # length
alpha = 4                      # material parameter, alpha = hP/kA
h = alpha**2 * k * R / 2       # heat transfer coefficient


# Model Parameters
case = 2
Tb = 0      # T(0), base temperature
Tl = 100    # T(L), tip temperature
Ta = 0      # ambient temperature

# Initialize Lists for data storage
nels = []
Ts_2nd = []
Ts_ex = []
Ts_extr = []
betas_T_extr = []
qs_2nd = []
qs_ex = []
qs_extr = []
betas_q_extr = []

for i in range(2, 12):
    # Define Mesh Parameters
    n = 2 ** (i) + 1
    nel = n-1
    dx = L/nel
    x = np.linspace(0, L, n)
    # number of equations/unknown nodes, dependent on boundary conditions
    neq = n-2 if case == 1 else n-1

    # Initialize K matrix and X vector
    K = np.zeros((neq, neq))
    X = np.zeros(neq)
    X[0] = Tb
    X[neq-1] = Tl

    # Mesh Dependent Parameters
    kappa = 2 + alpha**2 * dx**2

    # for i in range(neq):
    #     for j in range(neq):
    #         dif = abs(i-j)
    #         if case == 2 and i == 0 and dif == 0:
    #             K[i, j] = (2*dx*h/k + kappa)
    #         elif case == 2 and i == 0 and dif == 1:
    #             K[i, j] = -2
    #         elif dif == 0:
    #             K[i, j] = kappa
    #         elif dif == 1:
    #             K[i, j] = -1

    # Solve for approx T vector
    # T = np.linalg.solve(K, X).tolist()

    a = np.multiply(np.ones(neq-1), -1)
    b = np.multiply(np.ones(neq), kappa)
    c = np.multiply(np.ones(neq-1), -1)
    d = np.zeros(neq)

    if case == 1:
        d[neq-1] = Tl
    elif case == 2:
        b[0] = (2*dx*h/k + kappa)
        c[0] = -2
        d[neq-1] = Tl

    T = tridiag(a, b, c, d).tolist()

    # Apply x = 0 BC if case ==1
    if case == 1:
        T.insert(0, Tb)
    # Apply x = L BC
    T.insert(len(T), Tl)

    # Exact Solution
    if case == 1:
        C = (Tl - Ta - (Tb-Ta)*np.cosh(alpha*L))/np.sinh(alpha*L)
        D = 0
    elif case == 2:
        C = h/(k*alpha)*(Tl/(h/(k*alpha)*np.sinh(alpha*L)+np.cosh(alpha*L))-Ta)
        D = Tl/(h/(k*alpha)*np.sinh(alpha*L)+np.cosh(alpha*L))-Ta

    # Append iteration n's values to the data lists.
    nels.append(nel)

    T_2nd = T[0]
    Ts_2nd.append(T[0])

    q_2nd = -k*Ac*(T[n-2] - T[n-1] - dx**2 / 2*(alpha**2 * T[n-1]))/(-dx)
    qs_2nd.append(q_2nd)

    T_ex = C*np.sinh(alpha*x) + D*np.cosh(alpha*x) + Ta
    Ts_ex.append(T_ex[0])

    q_ex = -k*Ac*alpha*(C*np.cosh(alpha*L) + D*np.sinh(alpha*L))
    qs_ex.append(q_ex)

    # We can nly extrapolate if there are at least 3 values.
    if len(Ts_2nd) >= 3:
        if case != 1:
            (T_extr, beta_T_extr) = richardsons(Ts_2nd)
            Ts_extr.append(T_extr)
            betas_T_extr.append(beta_T_extr)

        (q_extr, beta_q_extr) = richardsons(qs_2nd)
        qs_extr.append(q_extr)
        betas_q_extr.append(beta_q_extr)


# Post Processing
if case == 2:
    T_table = go.Figure(data=[go.Table(header=dict(values=['dx', 'T\u0305(0)', 'T(0)', '%%error', r'$ \beta $']),
                                       cells=dict(values=[['1/'+str(x) for x in nels], tableFormat(Ts_2nd), tableFormat(Ts_ex), tableFormat(getError(Ts_ex, Ts_2nd)), tableFormat(getBeta(Ts_ex, Ts_2nd, nels))]))
                              ])
    T_extr_table = go.Figure(data=[go.Table(header=dict(values=['dx', 'T\u0305(0)', 'T<sub>EXTR</sub>(0)', '%%error', r'$ \beta $Extra']),
                                            cells=dict(values=[['1/'+str(x) for x in nels[2:]], tableFormat(Ts_2nd[2:]), tableFormat(Ts_extr), tableFormat(getError(Ts_extr, Ts_2nd[2:])), tableFormat(betas_T_extr)]))
                                   ])
    T_table.show()
    T_extr_table.show()

q_table = go.Figure(data=[go.Table(header=dict(values=['dx', 'q\u0307', 'q\u0307<sub>exact</sub>', '%%error', r'$ \beta $']),
                                   cells=dict(values=[['1/'+str(x) for x in nels], tableFormat(qs_2nd), tableFormat(qs_ex), tableFormat(getError(qs_ex, qs_2nd)), tableFormat(getBeta(qs_ex, qs_2nd, nels))]))
                          ])
q_extr_table = go.Figure(data=[go.Table(header=dict(values=['dx', 'q\u0307', 'q\u0307<sub>EXTR</sub>', '%%error', r'$ \beta $Extra']),
                                        cells=dict(values=[['1/'+str(x) for x in nels[2:]], tableFormat(qs_2nd[2:]), tableFormat(qs_extr), tableFormat(getError(qs_extr, qs_2nd[2:])), tableFormat(betas_q_extr)]))
                               ])
q_table.show()
q_extr_table.show()


def convPlot(exact, approx, nels, title):
     error = getError(exact, approx)
     dx = np.divide(1, nels)
     fig, ax = plt.subplots()
     ax.loglog(dx, error)
     ax.set_xlabel("dx")
     ax.set_ylabel("% Error")
     ax.set_title(title)
     annotation.slope_marker((.01, .01), (2, 1), ax=ax)
     plt.grid(True)
     plt.gca().invert_xaxis()
     plt.tight_layout()
     plt.show()


if case != 1:
    convPlot(Ts_ex, Ts_2nd, nels, 'Convergence of T(0)')
    convPlot(Ts_extr, Ts_2nd[2:], nels[2:],'Convergence of T(0) Against $T_{EXTR}$')
    convPlot(qs_ex, qs_2nd, nels, 'Convergence of q\u0307')
    convPlot(qs_extr, qs_2nd[2:], nels[2:], 'Convergence of q\u0307 Against $q\u0307_{EXTR}$')
