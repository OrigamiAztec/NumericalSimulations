import numpy as np


def richardsons(Q):
    i = len(Q) - 1
    Qe = (Q[i-1]**2 - Q[i-2]*Q[i])/(2*Q[i-1]-Q[i-2]-Q[i])
    beta = (np.log((Qe-Q[i-2])/(Qe-Q[i-1])))/np.log(2)
    return (Qe, beta)

