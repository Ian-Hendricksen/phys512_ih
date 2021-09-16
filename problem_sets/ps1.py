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

See the pdf under the same problem_sets folder! (titled Phys 512 A1.pdf)

(b)

See the same pdf for a derivation of the optimal delta

"""

def fpdiff(fun, x): # "four-point differentiator"
    eps = 1e-16 # Python uses double precision
    delta = eps**(1/6)
    fdp = fun(x+delta)
    fdm = fun(x-delta)
    f2dp = fun(x+2*delta)
    f2dm = fun(x-2*delta)
    return (8*(fdp - fdm) - (f2dp - f2dm))/(12*delta)

x = np.linspace(0.5, 10, 10)

print('(1b) exp(x) error = ', np.std(np.exp(x) - fpdiff(np.exp, x)))
print('(1b) exp(0.01x) error = ', 
      np.std(0.01*np.exp(0.01*x) - fpdiff(np.exp, 0.01*x)))

# The errors are not terrible, so this seems to be a reasonable assumption
# for the optimal delta.

#-----------------------------------------------------------------------------
# (Q2)

def ndiff(fun, x, full=False):
        
    eps = 1e-16
        
    if type(x) == float or type(x) == int:  
        x = np.asarray([x])
                
    fp = np.zeros(len(x))
    dx = np.zeros(len(x))
    err = np.zeros(len(x))
            
    for i in range(len(x)):
                
        if x[i] == 0: x[i] = 1e-10 # Arbitrary offset for cases of 0
        
        xc = x[i] 
        dx[i] = eps**(1/3)*xc # Following from Numerical Recipes
                
        temp = x[i] + dx[i]
        dx[i] = temp - x[i]
        
        fp[i] = (fun(temp) - fun(x[i]-dx[i]))/(2*dx[i]) # fp --> "f prime"        
                
        err[i] = (eps * abs(fun(x[i])))**(1/3)
    
    if full == False:
        return fp
    else:
        return fp, dx, err
    
# Let's try this with np.sin:
    
fun = np.sin
x = np.linspace(0, 2*np.pi, 10)
fp, dx, err = ndiff(fun, x, full = True)

# What's the error w.r.t. (d/dx)np.sin(x) = np.cos(x)?

print('(2) Error between ndiff and real fun = ', np.std(np.cos(x) - fp))

#-----------------------------------------------------------------------------
# (Q3)

dat = np.loadtxt('lakeshore.txt')
T = dat[:, 0]
Vo = dat[:, 1]

def lakeshore(V, data):
    
    T = dat[:, 0]
    Vo = dat[:, 1]
    
    #----------------------------
    
    # I originally tried using polynomial interpolation, but this turned out 
    # pretty bad (probably because the data has a kink and has some
    # oscillating behavior):
    
    # if type(V) == float or type(V) == int:  
    #     x = np.asarray([V])
                                    
    # npt = data.shape[0] # Number of rows in data is number of points (npt)
    # x = data[:, 1].flatten()
    # y = data[:, 0].flatten()
        
    # X = np.empty([npt, npt])
    # for i in range(npt):
    #     X[:, i] = x**i
    # Xinv = np.linalg.inv(X)
    # c = Xinv@y

    # XX = np.empty([len(V), npt]) # Here we see the same size vs. len issue!
    # for i in range(npt):
    #     XX[:, i] = V**i
    # y = XX@c
    
    #----------------------------
    
    # Set up spline, which gives an annoying error if V & T are 
    # not rearranged in ascending order, hence we have Vo[::-1]
    # to reverse the order
    
    spln = interpolate.splrep(Vo[::-1], T[::-1]) 
                                                
    y = interpolate.splev(V, spln)
    
    # Error?
    
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
y_true_cos = np.cos(xx)

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

yp_cos = polyfit(xx, x, y, npt) # 'y poly cos'
print('(4) Poly error (cos) = ', np.std(yp_cos - y_true_cos))

# Cubic Spline:
    
spln_cos = interpolate.splrep(x, y)
ys_cos = interpolate.splev(xx, spln_cos) # 'y spline cos'
print('(4) Spline error (cos) = ', np.std(ys_cos - y_true_cos))

# Rational:



#----------------------------
# Lorentzian

npt = 8 
xmin = -1
xmax = 1
x = np.linspace(xmin, xmax, npt)
y = 1/(1 + x**2)

xx = np.linspace(xmin, xmax, 1000)
y_true_lor = 1/(1+xx**2)

# I know this is repetitive, but now I can use different 
# numbers for each function!

# Polynomial:

yp_lor = polyfit(xx, x, y, npt) # Singular here too
print('(4) Poly error (Lorentzian) = ', np.std(yp_lor - y_true_lor))

# Cubic Spline:

spln_lor = interpolate.splrep(x, y)
ys_lor = interpolate.splev(xx, spln_lor)
print('(4) Spline error (Lorentzian) = ', np.std(ys_lor - y_true_lor))

# Rational:
    
    