#Antonio Diaz UIN 327003625 TAMU 2022
#Numerical Simulations 430 
#Hmwk 1 richardson method returns array to make it easier for other files to access.

import numpy as np


def richardsons(Q):
    #print(len(Q))
    i = len(Q) - 1
    #print(Q[i])
    #print(Q[i-1])
    #print(Q[i-2])
    #print(Q[i-1]**2 - Q[i-2]*Q[i])
    #print(2*Q[i-1]-Q[i-2]-Q[i])
    Qe = (Q[i-1]**2 - Q[i-2]*Q[i]) / (2*Q[i-1]-Q[i-2]-Q[i])
    beta = (np.log((Qe-Q[i-2])/(Qe-Q[i-1])))/np.log(2)
    return [Qe, beta]

