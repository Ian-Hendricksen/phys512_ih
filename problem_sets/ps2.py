# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 15:12:27 2021

@author: Ian Hendricksen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

#-----------------------------------------------------------------------------
# (Q1)

k = 9e9 # 1/4*pi*e_o

# Evaluate first using quad:

def eval_E_quad(z, R, sigma):
    
    e_0 = (1/4*np.pi*k)
    
    E = np.zeros(len(z))
    err = np.zeros(len(z))
    
    for i in range(len(z)):
    
        if z[i] < R:
            E[i] = 0
            err[i] = 0
            
        # elif z == R:
        #     the = 1 # temp
            
        else: # i.e. z >= R
            u = np.linspace(-1, 1, 1000)
            
            def integ(u):
                return (z[i] - R*u)/(R**2 + z[i]**2 - 2*R*z[i]*u)**(3/2)
            
            integ_tup = integrate.quad(integ, min(u), max(u)) # contains integrated answer + err
            
            E[i] = ((R**2 * sigma)/(2*e_0)) * integ_tup[0]
            err[i] = integ_tup[1]
    
    return E, err

z = np.linspace(0, 10, 1000)
R = 1
sigma = 1 # C/m^2

plt.plot(z, eval_E_quad(z,R,sigma)[0])

#-----------------------------------------------------------------------------
# (Q2)



#-----------------------------------------------------------------------------
# (Q3)

