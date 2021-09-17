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
print('----------------------------')

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

print('(2) Error between ndiff and real f\' = ', np.std(np.cos(x) - fp))
print('----------------------------')

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
    
    if type(V) == float or type(V) == int:  
        V = np.asarray([V])
                                    
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
    
    # Error (using bootstrap resampling):
                
    B = 10 # number of resamples
    n = 50 # number of samples per resample
    new_y = np.zeros([B, len(V)]) # An array to hold interpolated y values,
                                  # where each row is a new set of resampled 
                                  # data. Each column is associated with the 
                                  # same value of V
            
    for i in range(B):
        
        # As a bit of a narrative, the following method to randomly
        # select some indices and choose them from Vo and T produced
        # garbage results for many hours until I wrote "replace = False"... 
        # a mistake to never make again!
        
        indices = np.random.choice(len(Vo), n, replace = False) # Grab random indices
                                     
        Vsamps = Vo[indices] # Use random indices to select raw data values
        Tsamps = T[indices]
        
        sort_inds = np.argsort(Vsamps) # Grab sorted indices
        
        Vsamps = Vsamps[sort_inds] # Sort x and y data for splrep
        Tsamps = Tsamps[sort_inds]
                                                                
        new_spln = interpolate.splrep(Vsamps, Tsamps) 
        new_y[i] = interpolate.splev(V, new_spln)
        
    errs = np.zeros(len(V)) # Empty array for errors
        
    for i in range(len(V)):
        errs[i] = np.std(new_y[:,i]) # Determine std dev of each column of 
                                     # new_y, which contains each resampled 
                                     # interpolated value associated with a
                                     # specific V
                                        
    return y, errs

V = np.linspace(min(Vo), max(Vo), 100)
Tnew, Tnew_errs = lakeshore(V, dat)

plt.plot(V, Tnew, label = 'Interpolated Data')
plt.scatter(Vo, T, label = 'Raw Data')
plt.errorbar(V, Tnew, yerr = Tnew_errs, fmt = 'none', label = 'Errors on Interpolated Vals')
plt.xlabel('V')
plt.ylabel('T')
plt.title('Interpolated Lakeshore Data')
plt.legend()

print('(3) Mean error on interpolated temperatures =', np.mean(Tnew_errs))
print('----------------------------')
    
#-----------------------------------------------------------------------------
# (Q4)

#----------------------------
# cos(x)

npt = 9
xmin = -np.pi/2
xmax = np.pi/2
x = np.linspace(xmin, xmax, npt)
y = np.cos(x)

xx = np.linspace(xmin, xmax, 1000)
ytrue_cos = np.cos(xx)

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
yp_cos_err = np.std(yp_cos - ytrue_cos)
print('(4) Poly error (cos) = ', yp_cos_err)

# Cubic Spline:
    
spln_cos = interpolate.splrep(x, y)
ys_cos = interpolate.splev(xx, spln_cos) # 'y spline cos'
ys_cos_err = np.std(ys_cos - ytrue_cos)
print('(4) Spline error (cos) = ', ys_cos_err)

# Rational:

def rat_eval(p,q,x): # Using Jon's rat_eval & rat_fit from class...
    top=0
    for i in range(len(p)):
        top=top+p[i]*x**i
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*x**(i+1)
    return top/bot

def rat_fit(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    pars=np.dot(np.linalg.pinv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q

n = 4
m = npt+1-n # Need to ensure n+m-1 = npt
assert(n<m) # Interpolated values need to die as x --> inf
p, q = rat_fit(x, y, n, m)

yr_cos = rat_eval(p, q, xx) # 'y rational cos'
yr_cos_err = np.std(yr_cos - ytrue_cos)
print('(4) Rational error (cos) = ', yr_cos_err)

# Comparison

function = np.array(['Polynomial', 'Spline', 'Rational'])
q4cos_errs = np.array([yp_cos_err, ys_cos_err, yr_cos_err])
min_ind_cos = np.argmin(q4cos_errs)
print(f'(4) Best error for cos fits is {function[min_ind_cos]}')

#----------------------------
# Lorentzian

xmin = -1
xmax = 1
x = np.linspace(xmin, xmax, npt)
y = 1/(1 + x**2)

xx = np.linspace(xmin, xmax, 1000) # need to "reset" xx with new xmin/xmax
ytrue_lor = 1/(1+xx**2)

# Polynomial:

yp_lor = polyfit(xx, x, y, npt) # Singular here too
yp_lor_err = np.std(yp_lor - ytrue_lor)
print('(4) Poly error (Lorentzian) = ', yp_lor_err)

# Cubic Spline:

spln_lor = interpolate.splrep(x, y)
ys_lor = interpolate.splev(xx, spln_lor)
ys_lor_err = np.std(ys_lor - ytrue_lor)
print('(4) Spline error (Lorentzian) = ', ys_lor_err)

# Rational:
    
# This is really bad!
    
p, q = rat_fit(x, y, n, m)
yr_lor = rat_eval(p, q, xx) # 'y rational lor'
yr_lor_err = np.std(yr_lor - ytrue_cos)
print('(4) Rational error (Lorentzian) = ', yr_lor_err) 

# Comparison

q4lor_errs = np.array([yp_lor_err, ys_lor_err, yr_lor_err])
min_ind_lor = np.argmin(q4lor_errs)
print(f'(4) Best error for Lorentzian fits is {function[min_ind_lor]}')

"""

The error for the Lorentzian rational function fit should ideally be 0 
since the Lorentzian itself is a rational function, with p(x) = 1 and
q(x) = 1 + 0x + 1x**2 (I'm a little confused as to how to write p
and q in the manner that Jon set things up - he writes p = [1,2] and
q = [0,1,2], but I would expect it to be something like p = [1], and
q = [0,2], since we don't count the 1 in --> 1 + ax +bx^2 + cx^3 + ...).

The error for the Lorentzian rational function fit should be improved
(smaller) in comparison to the polynomial and cubic spline fits, since
a Lorentzian contains poles at +/- i in the complex plane, and only a 
rational function (of the 3 fits) has poles itself. When the order is higher,
the error appears

Switching from inv to pinv improves the quality of the rational function
fit (reduces the error). We notice a stark difference in p and q when 
comparing the use of inv and pinv; for inv, p and q are much larger (in this
case, some on the order of 1e16 or higher), while for pinv, p and q are much 
smaller (either around 1e-1 or 1e-16). This is likely related to the removal
of larger eigenvalues in A^-1, which is (I believe) directly related to the
coefficients for p and q. 


"""