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

"""
The following is a comparative method between different types of integration
techniques to solve for the electric field at a distance z above a thin
spherical shell with radius R and surface charge sigma, inside and outside the 
shell. The integral (taken from Griffiths) is of the following form:
    
                          /\ 1
    E(z) = k*2piR^2*sigma  |    (z - R*u)/(R**2 + z**2 - 2*R*z*u)**(3/2) du
                           \/ -1

The explicit solution of E(z) is E(z) = 0                       (z < R)
                                        k*2piR^2*sigma / z^2    (z > R)
"""

# Define some useful constants that are used for each type of integration:

k = 9e9 # 1/4*pi*e_o
e_0 = (1/4*np.pi*k)

#-------------------------------

# Define integrator using quad:

def eval_E_quad(z, R, sigma):
    
    E = np.zeros(len(z))
    selferr = np.zeros(len(z))
    
    for i in range(len(z)): # loop through each z
    
            u = [-1,1] # define limits of integ
            
            def integ(u):
                return (z[i] - R*u)/(R**2 + z[i]**2 - 2*R*z[i]*u)**(3/2) # integrand
            
            integ_tup = integrate.quad(integ, min(u), max(u)) # contains integrated answer + inherent err
            
            E[i] = (2*np.pi*R**2*sigma*k) * integ_tup[0] # multiply by relevant constants
            selferr[i] = integ_tup[1] # grab error
    
    return E, selferr

#-------------------------------

# Define integrator using Legendre:
    
def legendre_mat(npt):
    x=np.linspace(-1,1,npt)
    mat=np.zeros([npt,npt])
    mat[:,0]=1.0
    mat[:,1]=x
    if npt>2:
        for i in range(1,npt-1):
            mat[:,i+1]=((2.0*i+1)*x*mat[:,i]-i*mat[:,i-1])/(i+1.0)
    return mat

def integration_coeffs_legendre(npt):
    mat=legendre_mat(npt)
    mat_inv=np.linalg.inv(mat)
    coeffs=mat_inv[0,:]
    coeffs=coeffs/coeffs.sum()*(npt-1.0)
    return coeffs

def integrate_legendre(fun,xmin,xmax,dx_targ,ord=2):
    coeffs=integration_coeffs_legendre(ord+1)
    npt=int((xmax-xmin)/dx_targ)+1
    nn=(npt-1)%(ord)
    if nn>0:
        npt=npt+(ord-nn)
    assert(npt%(ord)==1)
    npt=int(npt)

    x=np.linspace(xmin,xmax,npt)
    dx=np.median(np.diff(x))    
    dat=fun(x)

    mat=np.reshape(dat[:-1],[(npt-1)//(ord),ord]).copy()
    mat[0,0]=mat[0,0]+dat[-1] 
    mat[1:,0]=2*mat[1:,0] 

    vec=np.sum(mat,axis=0)
    tot=np.sum(vec*coeffs[:-1])*dx
    return tot

# Using Jon's integrate_legendre method above, we evaluate the integral
# for every z value:

def eval_E_leg(z, R, sigma):
    
    u = [-1, 1] # set up limits and dx_targ
    dx_targ = 0.1
    integrated_vals = np.zeros(len(z))
    
    for i in range(len(z)): # as with quad, loop through integrand for each z
    
        def integrand(u):
            return (z[i] - R*u)/(R**2 + z[i]**2 - 2*R*z[i]*u)**(3/2)

        integrated_vals[i] = integrate_legendre(integrand, min(u), max(u), dx_targ) # apply legendre integration
    
    return (2*np.pi*R**2*sigma*k) * integrated_vals

#-------------------------------

# Set up the points for which we want to integrate, and our constants R and
# sigma:

n = 35
R = 1 # m
sigma = 1 # C/m^2

z = np.linspace(0, 10, n) # m
z = np.insert(z, n, R) # This makes sure there is a point where z = R
z.sort()

#-------------------------------

# Calculate exact E according to explicit (by-hand) integration:

z_exact = np.linspace(min(z), max(z), 1000)

def E_exact(z, R, sigma):
    E_exact = np.zeros(len(z))
    for i in range(len(E_exact)):
        if z[i] < R:
            E_exact[i] = 0
        if z[i] > R:
            E_exact[i] = k*4*np.pi*R**2*sigma/z[i]**2
    return E_exact

#-------------------------------

# Do some plotting and error estimation:
    
E_quad, E_quad_selferr = eval_E_quad(z, R, sigma)
E_leg = eval_E_leg(z, R, sigma)

print('(1)')

"""
Since E_exact doesn't evaluate E(z = R) (i.e. because of singularity), 
we need to delete the point in E_quad and E_leg where z = R to get our 
estimate of the error:
"""

err_quad = np.std(np.delete(E_quad - E_exact(z, R, sigma), np.where(z==R)))
err_leg = np.std(np.delete(E_leg - E_exact(z, R, sigma), np.where(z==R)))
print('Quad error = ', err_quad) 
print('My Integrator (Legendre) error = ', err_leg)

# Plot:

f1 = plt.figure()
plt.scatter(z, E_quad, label = 'Quad')
plt.scatter(z, E_leg, label = 'Legendre')
plt.plot(z_exact, E_exact(z_exact, R, sigma), color = 'green', label = 'Exact')
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
    
    z(1-u)/(2z^2(1-u))^(3/2) = (1/2)^(3/2)*(1/z^2(u-1)^(1/2))
    
However, this suggests a singularity at z = 0, but quad doesn't seem to 
care. It also seems that my integrator doesn't care about the singularity
at z = 0, although it does return nan for z = R.

Ultimately quad seems to perform far better than Legendre. The residuals
are much smaller, and there isn't any weird behavior around z ~ R like
we observe with Legendre.
"""

print('-------------------------------------')

#-----------------------------------------------------------------------------
# (Q2)

"""
My integrate_adaptive currently gets stuck in a big loop. For some reason it
doesn't meet the err < tol condition for a while, and continues to integrate
between smaller and smaller intervals until the whole thing crashes. I am at
a loss to explain where the error is coming from, since the bones of what it
is doing are essentially equivalent to Jon's. The only thing I have added is
a few bits that pull out the already evaluated x and y (extra_l and extra_r), 
and passes them into the recursive function call in the last else statement 
so that this call of the function doesn't have to reevaluate those points again. 

Suffice it to say, this calls f(x) far more times than the one written in
class. Call this function (at the bottom of Q2) at your own (computer's) risk!
"""

def integrate_adaptive(fun, a, b, tol, extra = None, calls = [0]):
    print('-----')
    print('integrating between ', a, b)
    calls[0]+=1
    x = np.linspace(a, b, 5)
    dx = (b - a)/(len(x) - 1)
        
    if type(extra) == type(None): # This if/else statement checks if type(extra)
                                  # is None (i.e. the first call of the function)
        y = fun(x)
        area1 = 2*dx*(y[0] + 4*y[2] + y[4])/3
        area2 = dx*(y[0] + 4*y[1] + 2*y[2] + 4*y[3] + y[4])/3 
        err = np.abs(area1 - area2)
        
    else: # all else will be a recursive function call from the below if/else
        
        # Delete the x values for which we already evaluated y:
        
        for i in range(len(extra[0, :])):
            index = np.where(extra[0, :][i] == x)[0][0]
            x = np.delete(x, index)

        # Evaluate y for the remaining x values:
        
        y = fun(x)
        # print('x trunc', x)
        # print('extra x', extra[0,:])
        
        # Recombine the newly evaluated x and y values with the old ones:
                
        x = np.concatenate((x, extra[0, :]))
        x = np.sort(x)
        
        y = np.concatenate((y, extra[1, :]))
        y = np.sort(y)
        # print('x', x)
        
        area1 = 2*dx*(y[0] + 4*y[2] + y[4])/3
        area2 = dx*(y[0] + 4*y[1] + 2*y[2] + 4*y[3] + y[4])/3
        err = np.abs(area1 - area2)
        
    # print(f'{err} < {tol} -->', err<tol)   
    
    if err < tol:
        return area2

    else:
        xmid = (a + b)/2
        xl = np.linspace(a, xmid, 5)
        xr = np.linspace(xmid, b, 5)
        
        # Grab the x and y points we've already evaluated and
        # throw them into an "extra" array, where the first row 
        # contains already used x values and the second row 
        # contains their associated y values:
            
        extra_xl = x[np.where(xl == x)[0]]
        extra_xr = x[np.where(xr == x)[0]]
        
        extra_yl = y[np.where(xl == x)[0]]
        extra_yr = y[np.where(xr == x)[0]]
                    
        extra_l = np.vstack((extra_xl, extra_yl))
        extra_r = np.vstack((extra_xr, extra_yr))
                        
        left = integrate_adaptive(fun, a, xmid, tol/2, extra = extra_l)
        right = integrate_adaptive(fun, xmid, b, tol/2, extra = extra_r)
        
        lpr =  left + right
    
        return lpr
    
a = -100
b = 100
fun = np.exp
# ans = integrate_adaptive(fun, a, b, 1e-7, extra = None, calls = [0])
print('(2)')
# print('integrate_adaptive error = ', ans - (fun(b) - fun(a)))
print('-------------------------------------')

#-----------------------------------------------------------------------------
# (Q3)

# Evaluate log_2(x) using Chebyshev polynomials:

def cheb_log2(xx):
    
    n = 150
    x = np.linspace(0.5, 1, 1000) # our function needs to be valid from 0.5 to 1
    x_resc = np.linspace(-1, 1, 1000) # rescale the x values
    y = np.log2(x)
    coeffs = np.polynomial.chebyshev.chebfit(x_resc, y, n) # grab coeffs
    
    # Check to see if xx is float or array:
    
    if type(xx) == np.float64 or type(xx) == float:
        yy = np.polynomial.chebyshev.chebval(xx, coeffs) 
    else:
        xx_resc = np.linspace(-1, 1, len(xx))
        yy = np.polynomial.chebyshev.chebval(xx_resc, coeffs) # evaluate at
                                                              # -rescaled-
                                                              # values, not xx
         
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
        trunc_coeffs = coeffs[:i+1] # truncate coefficients up to i+1 
        yyy = np.polynomial.chebyshev.chebval(xx, trunc_coeffs) # reevaluate with truncated polynomial
        if type(xx) == int or type(xx) == float: # this if/else statement suppresses an index error
            trunc_errs[i] = abs(yyy - np.log2(xx))
        else:
            trunc_errs[i] = abs(np.std(yyy - np.log2(xx)))
    
    tol = 1e-6 # set tolerance/error threshold
        
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
   
f2 = plt.figure()
xx = np.linspace(0.5, 1, 25)
plt.plot(xx, np.log2(xx), color = 'red', label = 'log$_{2}$(x)')
plt.scatter(xx, cheb_log2(xx), color = 'green', label = 'Chebyshev fit')
plt.title('Chebyshev Fit to log_2(x) from 0.5 to 1')
plt.legend()

print('(3)')
print('log_2 Chebyshev error = ', np.std(np.log2(xx) - cheb_log2(xx)))

# Evaluate ln(x) for any positive number:
    
def mylog2(xx):
    
    """
    np.frexp returns (mantissa, exponent), where x = mantissa * 2**exponent`. 
    The mantissa lies in the open interval(-1, 1), while the twos exponent 
    is a signed integer. (For my own reference)
    
    Say we want ln(x). We can use some log rules to state that
    
        ln(x) = log_e(x) = log_2(x) / log_2(e)
        
    Further, if we break up x and e into their mantissa & exponent, say
    x = m*2^n and e = a*2^b, we can get everything into a form that can be
    handled by Chebyshev polynomials:
        
        log_2(m2^n) / log_2(a2^b)
                                   = (n + log_2(m)) / (b + log_2(a)) (*) 
        
    where -1 < m, a < 1. Now we simply construct log_2(xx) using Chebyshev
    polynomials and calculate Equation (*):
    """
    
    e = np.exp(1) # first, get the numerical value for e
    a, b = np.frexp(e)
    m, n = np.frexp(xx)
    
    def cheb_man(xxx): # chebyshev fit for mantissa's
        N = 150
        n = 1000
        x = np.linspace(1e-15, 1, n) # mantissa lies in -1 < a, m < 1, 
                                     # but log_2(x) --> inf as x --> 0; 
                                     # need to rescale!
        x_resc = np.linspace(-1, 1, n)
        y = np.log2(x)
        coeffs = np.polynomial.chebyshev.chebfit(x_resc, y, N)
        return np.polynomial.chebyshev.chebval(xxx, coeffs)
    
    ln = (n + cheb_man(m))/(b + cheb_man(a)) # using Equation (*)
    return ln

xx = np.linspace(1, 1e15, 1000)
print('ln(x) Chebyshev error = ', np.std(mylog2(xx) - np.log(xx)))
f3 = plt.figure()
plt.plot(xx, mylog2(xx), label = 'mylog2(x)')
plt.plot(xx, np.log(xx), label = 'log$_2$(x)')
plt.legend()

"""
The error improves over what I had previously, but there's still something
funky going on with the the fit that ends up having some weird steps (I
believe something similar was seen in Rigel's tutorial).
"""