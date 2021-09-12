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
    
    x = np.asarray(x)
                
    fp = np.zeros(x.size)
    dx = np.zeros(x.size)
    err = np.zeros(x.size)
    
    """
    
    The reason I use size instead of len is because the function should
    handle floats, ints, and arrays. However, I encountered a lot of issues
    when trying to convert x to an array if x is a float or int. If I use
    np.asarray(x) in the way I have now, it does not affect x if it is an
    array, but will convert a float or int, say 0.5, to array(0.5), which
    will return TypeError: len() of unsized object. The other alternative is
    to use np.asarray([x]), but this will convert 1D arrays to 2D arrays.
    For a 1D array, size is no different from len, so I chose the former.
    You may ask why I have taken the time to write this, and it is only
    fuelled by my frustration with the way python chooses to do things.
    
    """
    
    for i in range(x.size):
        
        if x[i] == 0: x[i] = 1e-10 # Arbitrary offset for cases of 0
        
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
# plt.scatter(V,T)

def lakeshore(V, data):
        
    V = np.asarray(V)
                            
    npt = data.shape[0] # Number of rows in data is number of points (npt)
    x = data[:, 1].flatten()
    y = data[:, 0].flatten()
        
    X = np.empty([npt, npt])
    for i in range(npt):
        X[:, i] = x**i
    Xinv = np.linalg.inv(X)
    c = Xinv@y

    XX = np.empty([V.size, npt]) # Here we see the same size vs. len issue!
    for i in range(npt):
        XX[:, i] = V**i
    y = XX@c
    
    return y
    
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
    
def polyfit(xx, x, y, npt):
    
    X = np.empty([npt, npt])
    for i in range(npt):
        X[:, i] = x**i
        Xinv = np.linalg.inv(X)
        c = Xinv@y

    XX = np.empty([len(xx), npt])
    for i in range(npt):
        XX[:, i] = xx**i
    y1 = XX@c
    
    return y1

# y1_cos = polyfit(xx, x, y, npt) # I keep getting a singular matrix here

# Error?

# Cubic Spline:
    
spln_cos = interpolate.splrep(x, y)
y2_cos = interpolate.splev(xx, spln_cos)
    
# Error?

# Rational:



#----------------------------
# Lorentzian

npt = 8 
xmin = -1
xmax = 1
x = np.linspace(xmin, xmax, npt)
y = 1/(1 + x**2)

# I know this is repetitive, but now I can use different 
# numbers for each function!

xx = np.linspace(xmin, xmax, 1000)

# Polynomial:

# y1_lor = polyfit(xx, x, y, npt) # Singular here too

# Error?

# Cubic Spline:

spln_lor = interpolate.splrep(x, y)
y2_lor = interpolate.splev(xx, spln_lor)

# Error?

# Rational:
    
    

