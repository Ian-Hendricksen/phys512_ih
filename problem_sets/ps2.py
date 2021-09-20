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
    
            u = [-1,1]
            
            def integ(u):
                return (z[i] - R*u)/(R**2 + z[i]**2 - 2*R*z[i]*u)**(3/2)
            
            integ_tup = integrate.quad(integ, min(u), max(u)) # contains integrated answer + err
            
            E[i] = ((R**2 * sigma)/(2*e_0)) * integ_tup[0]
            err[i] = integ_tup[1]
    
    return E, err

n = 25
R = 1 # m
sigma = 1 # C/m^2

z = np.linspace(0, 10, n) # m
z = np.insert(z, n, R) # This makes sure there is a point where z = R
z.sort()

E_quad, E_quad_err = eval_E_quad(z, R, sigma)
print('Quad error:', np.std(E_quad_err))

# Evaluate using my own integrator:
    
    

plt.scatter(z, E_quad, label = 'quad')
plt.xlabel('z')
plt.ylabel('E')
plt.title(f'Spherical Shell E-Field, R = {R}, $\sigma$ = {sigma}')
plt.legend()

"""

There is indeed a singularity at z = R since there is a 1/|z-R| term
that pops out after integration, which goes to infinity at that point.
However, quad doesn't seem to care that such a point blows up, although
this could result from the fact that the singularity is apparent after 
integrating as opposed to being under the integral, since for z = R the
integrand is:
    
    z(1-u)/(2z^2(1-u))^(3/2) # Check this assumption is correct


"""


#-----------------------------------------------------------------------------
# (Q2)



#-----------------------------------------------------------------------------
# (Q3)

