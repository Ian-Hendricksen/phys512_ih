# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:03:39 2021

@author: Ian Hendricksen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

#-----------------------------------------------------------------------------
# (Q1)

"""

(a)



(b)



"""

#-----------------------------------------------------------------------------
# (Q2)

def ndiff(fun, x, full=False):
        
    eps = 1e-16 # Python uses double precision
    
    fp = np.zeros(len(x))
    dx = np.zeros(len(x))
    err = np.zeros(len(x))
    
    for i in range(len(x)):
        
        if x[i] == 0: x[i] = 1e-10 # Arbitrary offset
        
        xc = x[i] 
        dx[i] = eps**(1/3)*xc # Following from Numerical Recipes
        
        temp = x[i] + dx[i]
        dx[i] = temp - x[i]
        
        fp[i] = (fun(temp) - fun(x[i]-dx[i]))/(2*dx[i]) # fp --> "f prime"        
        
        # NEED TO ADD ERROR
        
        err[i] = 10 #temporary
    
    if full == False:
        return fp
    else:
        return fp, dx, err

#-----------------------------------------------------------------------------
# (Q3)

dat = np.loadtxt('lakeshore.txt')
T = dat[:, 0]
V = dat[:, 1]

# plt.ion()
# plt.scatter(T,V)

def lakshore(V, data):
    return V

#-----------------------------------------------------------------------------
# (Q4)

#----------------------------
# cos(x)

npt = 8
xmin = -np.pi/2
xmax = np.pi/2
x = np.linspace(xmin, xmax, npt)
y = np.cos(x)

xx = np.linspace(xmin, xmax, 1000)

# Polynomial:
    
X = np.empty([npt, npt])
for i in range(npt):
    X[:, i] = x**i
Xinv = np.linalg.inv(X)
c = Xinv@y

XX = np.empty([len(xx), npt])
for i in range(npt):
    XX[:, i] = xx**i
y1 = XX@c

# Error?

# Cubic Spline:
    
spln = interpolate.splrep(x, y)
y2 = interpolate.splev(xx, spln)
    
# Error?

# Rational



#----------------------------
# Lorentzian