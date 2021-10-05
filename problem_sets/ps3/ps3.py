# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 21:10:40 2021

@author: Ian Hendricksen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

import sys
orig_stdout = sys.stdout
f = open('A3 Print Output.txt', 'w')
sys.stdout = f

#-----------------------------------------------------------------------------
# (Q1)

# Define our function and the basic RK4:

def dydx(x, y):
    return y/(1 + x**2)

#-------------------------------

# One step RK4:
    
def rk4_step(fun, x, y, h):
    
    for i in range(len(x)-1):
        k1 = fun(x[i], y[i])*h
        k2 = h*fun(x[i] + h/2, y[i] + k1/2)
        k3 = h*fun(x[i] + h/2, y[i] + k2/2)
        k4 = h*fun(x[i] + h, y[i] + k3)
        dy = (k1 + 2*k2 + 2*k3 + k4) / 6
        y[i+1] = y[i] + dy
        
    return y

#-------------------------------

# Comparative single step & two half-steps RK4:
    
def rk4_stepd(fun, x, y, h):
    
    y_2n = np.zeros(len(x))
    y_2n[0] = y[0]
    
    y_new = np.zeros(len(x))
    y_new[0] = y[0]
        
    for i in range(len(x)-1):
        
        #-------------------------------
        # Single step:
        k1 = fun(x[i], y[i])*h
        k2 = h*fun(x[i] + h/2, y[i] + k1/2)
        k3 = h*fun(x[i] + h/2, y[i] + k2/2)
        k4 = h*fun(x[i] + h, y[i] + k3)
        dy = (k1 + 2*k2 + 2*k3 + k4) / 6
        y[i+1] = y[i] + dy
        
        #-------------------------------
        # Two half-steps:
        k1_s1 = k1 # we already have this
        k2_s1 = (h/2)*fun(x[i] + h/4, y[i] + k1_s1/2)
        k3_s1 = (h/2)*fun(x[i] + h/4, y[i] + k2_s1/2)
        k4_s1 = (h/2)*fun(x[i] + h/2, y[i] + k3_s1)
        dy_s1 = (k1_s1 + 2*k2_s1 + 2*k3_s1 + k4_s1) / 6
        step1 = y[i] + dy_s1
                
        k1_s2 = fun(x[i] + h/2, y[i])*(h/2)
        k2_s2 = (h/2)*fun(x[i] + 3*h/4, y[i] + k1_s2/2)
        k3_s2 = (h/2)*fun(x[i] + 3*h/4, y[i] + k2_s2/2)
        k4_s2 = (h/2)*fun(x[i] + h, y[i]  + k3_s2)
        dy_s2 = (k1_s2 + 2*k2_s2 + 2*k3_s2 + k4_s2) / 6
        step2 = step1 + dy_s2
        
        # print(step1,step2)
        
        y_2n[i+1] = step2 
        
        y_new[i+1] = y_2n[i+1] + (y_2n[i+1] - y[i])/15 # Equation 17.2.3
            
    return y_new

#-------------------------------
"""
We first want to integrate the ODE with each solver having
the same number of steps:
"""

n_step = 201 # 200 steps + starting point
x = np.linspace(-20, 20, n_step)
y = np.zeros(len(x))
y[0] = 1 # y(-20) = 1
h = np.median(np.diff(x)) # length of step

ans_step = rk4_step(dydx, x, y, h) # 200 steps
ans_stepd = rk4_stepd(dydx, x, y, h) # 200 steps

"""
Now we want to integrate the ODE using rk4_stepd such that 
we evaluate the function the same number of times as rk4_step.
Each step in rk4_step requires 4 function evaluations, and each
step in rk4_stepd requires 11 function evaluations. For 200 steps,
rk4_step evaluates the function 800 times. For rk4_stepd
to evaluate the function 800 times, rk4_stepd needs to have
800/11 ~ 73 steps.
"""

n_step_new = 73
x_new = np.linspace(-20, 20, n_step_new)
y_new = np.zeros(len(x_new))
y_new[0] = 1
h_new = np.median(np.diff(x_new))

ans_stepd_new = rk4_stepd(dydx, x_new, y_new, h_new) # 73 steps

def ytrue(x):
    c0 = y[0]/np.exp(np.arctan(x[0]))
    return c0 * np.exp(np.arctan(x))

# f1 = plt.figure()
# plt.plot(x, ytrue(x), label = 'True Function')
# plt.plot(x, ans_step, label = 'rk4_step, 200 steps')
# plt.plot(x, ans_stepd, label = 'rk4_stepd, 200 steps')
# plt.plot(x_new, ans_stepd_new, label = 'rk4_stepd, 73 steps')
# plt.legend()
# plt.savefig("./A3Q1.png")

print('-------------------------------')
print('(1)')
print('rk4_step error (200 steps) = ', np.std(ytrue(x) - ans_step))
print('rk4_stepd error (200 steps) = ', np.std(ytrue(x) - ans_stepd))
print('rk4_stepd error (73 steps) = ', np.std(ytrue(x_new) - ans_stepd_new))

# Something wrong here - should the stepd error be smaller?

print('-------------------------------')

#-----------------------------------------------------------------------------
# (Q2)

#-------------------------------
# (a) U238_products solves for the decay products of U238 using
# scipy.integrate's solve_ivp. 

half_lives = np.array([1.41e17, 2.0822e6, 24120, 7.74e12,
              2.38e12, 5.05e10, 3.3e5, 186, 1608,
              1194, 164.3e-6, 7.03e8, 1.58e8, 1.20e7, 1e30])*3.171e-08 # years

def U238_products(half_lives, t, N0):
    
    def decays(t, N, half_lives = half_lives): # Define our ODE equations
                                               # in a loop.
        
        dNdt = np.zeros(len(half_lives))
        
        for i in range(1, len(half_lives)):
        
            dNdt_init = -N[i-1]/half_lives[i-1]
            dNdt_both = -N[i]/half_lives[i] + N[i-1]/half_lives[i-1]
            dNdt_fin = N[i]/half_lives[i] # how to include this?
        
            dNdt[i-1] = dNdt_init + dNdt[i-1] # Update the previous product
                                              # to account for decays into
                                              # a new product
                                              
            dNdt[i] = dNdt_both # need to figure out what to update
        
        return dNdt
        
    solve_ode = integrate.solve_ivp(decays, [min(t), max(t)], N0, method = 'Radau', t_eval = t)
    
    return solve_ode

# We need to define an array N0 of initial conditions. N0[0] = 1 to represent
# a pure sample of Uranium, the 1 being convenient so as to have a normalized
# output. N contains all the information return by the ODE solver under
# U238_products, where N.y[i] returns the ith product for all times t.
        
t = np.logspace(0, 11, 10000) # 1000 time points over 1e11 (100 billion) years
N0 = np.zeros(len(half_lives))
N0[0] = 1
N = U238_products(half_lives, t, N0)

# for i in range(len(half_lives)):
#     N.y[i] = abs(N.y[i])

#-------------------------------
# (b)

# NEED TO FIX WHATEVER'S GOING ON HERE

# f2 = plt.figure()
# plt.plot(t, N.y[-1]/N.y[0]) # Plot ratio of Pb206/U238
# plt.xlabel('Time (years)')
# plt.ylabel('$N_{Pb206}$ / $N_{U238}$')
# plt.title('Ratio of Pb206 to U238')
# plt.savefig("./A3Q2_Pb206_U238_ratio.png")

# f3 = plt.figure()
# plt.plot(t, N.y[4]/N.y[3]) # Plot ratio of Th230/U234
# plt.xlabel('Time (years)')
# plt.ylabel('$N_{Th230}$ / $N_{U234}$')
# plt.title('Ratio of Th230 to U234')
# plt.savefig("./A3Q2_Th230_U234_ratio.png")

#-----------------------------------------------------------------------------
# (Q3)

data = np.loadtxt('dish_zenith.txt')
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

"""
A rotationally symmetric paraboloid is described by
    
    z = z_0 + a((x - x_0)^2 + (y - y_0)^2),
    
and we want to fit for x_0, y_0, z_0, and a. However, we
see that x_0 and y_0 are not linear in the equation. In order
to make this equation linear in its parameters, we must substitute
new parameters. Let b = -2ax_0, c = -2ay_0, and d = a(x_0^2 + y_0^2) + z_0;
our new equation becomes

    z = a(x^2 + y^2) + bx + cy + d
"""

def dish_fit(data):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    
    nd = len(z) # n data
    nm = 4 # n model, i.e. a,b,c,d --> 4 parameters
    
    A = np.zeros((nd, nm))
    A[:, 0] = x**2 + y**2
    A[:, 1] = x
    A[:, 2] = y
    A[:, 3] = np.ones(nd)
    
    N = np.eye(nd)*(0.01)**2 # temporary
    
    Ninv = np.linalg.pinv(N)
    
    m = np.linalg.pinv(A.T@Ninv@A)@A.T@Ninv@z # a, b, c, d
    
    m_errs = np.sqrt(np.diag(np.linalg.pinv(A.T@Ninv@A)))
    
    return m, m_errs

m, m_errs = dish_fit(data)
z_new = m[0]*(x**2 + y**2) + m[1]*x + m[2]*y + m[3]
print('-------------------------------')
print('(3)')
print('Std Dev Between Fit and True = ', np.std(z_new - z)) # Not the complete error, need chi_sq!
print('-------------------------------')

# f4 = plt.figure()
# ax = f4.add_subplot(111, projection='3d')
# ax.scatter(x,y,z)

# x_new = np.linspace(min(x), max(x), 50)
# y_new = np.linspace(min(y), max(y), 50)
# X, Y = np.meshgrid(x_new, y_new)
# z_new = a*(X**2 + Y**2) + b*X +c*Y + d
# ax.scatter(X,Y,z_new)

sys.stdout = orig_stdout
f.close()