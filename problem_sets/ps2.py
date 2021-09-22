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
print('(1) Quad error:', np.std(E_quad_err))

# Evaluate using Legendre:
    
def get_legendre_weights(n):
    x=np.linspace(-1,1,n+1)
    P=np.polynomial.legendre.legvander(x,n)
    Pinv=np.linalg.inv(P)
    coeffs=Pinv[0,:]
    return coeffs*n

def leg_eval(x, fun, n):
    coeffs = get_legendre_weights(n)
    dx = x[1] - x[0]
    return np.sum(coeffs*fun(x))*dx
    
def eval_E_leg(z, R, sigma):
    
    e_0 = (1/4*np.pi*k)
    E = np.zeros(len(z))    
    n = len(z)
    u = np.linspace(-1, 1, n)
    
    for i in range(len(z)):
    
        def fun(u):
            return (z[i] - R*u)/(R**2 + z[i]**2 - 2*R*z[i]*u)**(3/2)
        
        E = ((R**2 * sigma)/(2*e_0)) * leg_eval(u, fun, n)
    
    return E

# print(eval_E_leg(z, R, sigma))

# x=np.linspace(0,1,len(coeffs))
# y=np.exp(x)
# dx=x[1]-x[0]
# my_int=np.sum(coeffs*y)*dx

# Plot:

# plt.scatter(z, E_quad, label = 'quad')
# plt.xlabel('z')
# plt.ylabel('E')
# plt.title(f'Spherical Shell E-Field, R = {R}, $\sigma$ = {sigma}')
# plt.legend()

"""

There is indeed a singularity at z = R since there is a 1/|z-R| term
that pops out after integration, which goes to infinity at that point.
However, quad doesn't seem to care that such a point blows up, although
this could result from the fact that the singularity is apparent after 
integrating as opposed to being under the integral, since for z = R the
integrand is:
    
    z(1-u)/(2z^2(1-u))^(3/2) # Check this assumption is correct


"""

print('#-------------------------------------')

#-----------------------------------------------------------------------------
# (Q2)

# def integrate_adaptive(fun, a, b, tol, extra = None):


#-----------------------------------------------------------------------------
# (Q3)

def cheb_log2(xx):
    n = 150
    x = np.linspace(0.5, 1, 1000)
    x_resc = np.interp(x, (min(x), max(x)), (-1, +1))
    y = np.log2(x)
    # y = np.log2(x_resc)
    coeffs = np.polynomial.chebyshev.chebfit(x, y, n)
    print(coeffs)
    yy = np.polynomial.chebyshev.chebval(xx, coeffs)
    max_errs = np.zeros(n)
    for i in range(n):
        max_errs[i] = np.sum(abs(coeffs[-i:]))
    print(max_errs)
    return yy
    
xx = np.linspace(0.5, 1, 100)
# plt.plot(xx, np.log2(xx))
# plt.scatter(xx, cheb_log2(xx))
print(np.std(np.log2(xx) - cheb_log2(xx)))