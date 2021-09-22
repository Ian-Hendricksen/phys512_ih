# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 15:12:27 2021

@author: Ian Hendricksen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import interpolate

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

def integrate_adaptive(fun, a, b, tol, extra = None):
    
    print('integrating between ', a, b)
    x = np.linspace(a, b, 5)
    dx = (b - a)/(len(x) - 1)
        
    if type(extra) == type(None):
        y = fun(x)
        area1 = 2*dx*(y[0] + 4*y[2] + y[4])/3 # coarse step
        area2 = dx*(y[0] + 4*y[1] + 2*y[2] + 4*y[3] + y[4])/3 # finer step
        err = np.abs(area1 - area2)
        
    else:
        for i in range(len(extra[0, :])):
            index = np.where(extra[0, :][i] == x)[0][0]
            x = np.delete(x, index)
        
        y = fun(x)
                
        x = np.concatenate((x, extra[0, :]))
        x = np.sort(x)
        
        y = np.concatenate((y, extra[1, :]))
        y = np.sort(y)
        
        area1 = 2*dx*(y[0] + 4*y[2] + y[4])/3 # coarse step
        area2 = dx*(y[0] + 4*y[1] + 2*y[2] + 4*y[3] + y[4])/3 # finer step
        err = np.abs(area1 - area2)
        
        # if err < tol:
        #     return area2
        
        
    if err < tol:
        return area2
    
    else:
        xmid = (a + b)/2
        xl = np.linspace(a, xmid, 5)
        xr = np.linspace(xmid, b, 5)
        
        extra_xl = []
        extra_yl = []
        
        extra_xr = []
        extra_yr = []
        
        # for i in range(len(xl)):
        #     index = np.where(xl[i] == x)[0][0]
        #     print(index)
        #     extra_xl.append(x[index])
        #     extra_yl.append(y[index])
            
        extra_xl = x[np.where(xl == x)[0]]
        extra_xr = x[np.where(xr == x)[0]]
        
        extra_yl = y[np.where(xl == x)[0]]
        extra_yr = y[np.where(xr == x)[0]]
            
        # for i in range(len(xr)):
        #     index = np.where(xr[i] == x)[0][0]
        #     extra_xr.append(x[index])
        #     extra_yr.append(y[index])
                    
        extra_l = np.vstack((extra_xl, extra_yl))
        extra_r = np.vstack((extra_xr, extra_yr))
                        
        left = integrate_adaptive(fun, a, xmid, tol/2, extra = extra_l)
        right = integrate_adaptive(fun, xmid, b, tol/2, extra = extra_r)
        
        return left + right
    
a = -100
b = 100
print(integrate_adaptive(np.exp, a, b, 1e-7))

#-----------------------------------------------------------------------------
# (Q3)

# Evaluate log_2(x) using Chebyshev polynomials:

def cheb_log2(xx):
    
    n = 150
    x = np.linspace(0.5, 1, 1000)
    
    def scale(x, out_range=(-1, 1)):
        domain = np.min(x), np.max(x)
        y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
        return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
    
    x_resc = scale(x)
    # x_resc = np.interp(x, (min(x), max(x)), (-1, +1))
    y = np.log2(x_resc)
    
    # y = np.log2(x)
    coeffs = np.polynomial.chebyshev.chebfit(x, y, n)
    yy = np.polynomial.chebyshev.chebval(xx, coeffs)
    
    """
    I first tried applying the method to find the minimum number of
    coefficients needed from Jon's polynomial notes, but decided it
    was easier to evaluate yy for each number of coefficients - i.e. 
    I evaluate yy many times, looping through the number of coefficients
    from 0 to n, define some tolerance, then ask python to find the 
    number of coefficients corresponding with the first instance where
    the error drops below the tolerance.
    """
    
    #-----------------------
    
    # max_errs = np.zeros(n)
    # for i in range(n):
    #     max_errs[i] = np.sum(abs(coeffs[-i+1:])) # the errs are in descending order w.r.t. coeffs
    # max_errs = max_errs[::-1] # return to ascending order w.r.t. coeffs
    # cutoff_index= np.where(max_errs < 1e-4)[0][0]
    # coeffs = coeffs[:cutoff_index+1]
    # yy = np.polynomial.chebyshev.chebval(xx, coeffs)
    
    #-----------------------
    
    # Routine to check errors for truncated coefficients:
    
    trunc_errs = np.zeros(n)
    
    for i in range(n):
        trunc_coeffs = coeffs[:i+1]
        yyy = np.polynomial.chebyshev.chebval(xx, trunc_coeffs)
        trunc_errs[i] = abs(np.std(yyy - np.log2(xx)))
    
    tol = 1e-3 # arbitrarily set tolerance/error threshold
        
    tol_inds = np.where(trunc_errs < tol)[0] # get indices where the error is
                                             # less than tol
                                                 
    if len(tol_inds) == 0:
        return yy # if tol_inds is empty, the only # of coeffs with err < tol is n
    else:
        tol_index = tol_inds[0]
        tol_coeffs = coeffs[:tol_index] # gather the coefficients up to this index
        yy = np.polynomial.chebyshev.chebval(xx, tol_coeffs) # re-evaluate yy with
                                                             # truncated coeffs                                       
        
        return yy
    
xx = np.linspace(0.5, 1, 100)
# plt.plot(xx, np.log2(xx))
# plt.scatter(xx, cheb_log2(xx))
# print(np.std(np.log2(xx) - cheb_log2(xx)))

# Evaluate ln(x) for any positive number:
    
def mylog2(x):
    
    e = np.exp(1)
    
    # np.frexp returns (mantissa, exponent), where x = mantissa * 2**exponent`. 
    # The mantissa lies in the open interval(-1, 1), while the twos exponent 
    # is a signed integer.