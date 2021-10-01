# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 21:10:40 2021

@author: Ian Hendricksen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

#-----------------------------------------------------------------------------
# (Q1)

# Define our function and the basic RK4:

def dydx(x, y):
    return y/(1 + x**2)

def rk4(fun, x, y, h):
    k1 = fun(x, y)*h
    k2 = h*fun(x + h/2, y + k1/2)
    k3 = h*fun(x + h/2, y + k2/2)
    k4 = h*fun(x + h, y + k3)
    dy = (k1 + 2*k2 + 2*k3+k4) / 6
    return y + dy

#-------------------------------

# One step RK4:
    
def rk4_step(fun, x, y, h):

    for i in range(len(x)-1):
        y[i+1] = rk4(fun, x[i], y[i], h)
        
    return y

#-------------------------------

# Comparative single step & two half-steps RK4:
    
def rk4_stepd(fun, x, y, h):
    
    # n_2step = 2*len(x)
    
    # x_2n = np.linspace(min(x), max(x), n_2step)
    y_2n = np.zeros(len(x))
    y_2n[0] = y[0]
        
    for i in range(len(x)-1):
        y[i+1] = rk4(fun, x[i], y[i], h)

        step1 = rk4(fun, x[i], y[i], h/2)
        step2 = rk4(fun, x[i]+h/2, y[i] + step1, h/2)
        y_2n[i+1] = step2 - step1 # <----------- is this correct?
        
    y_new = y_2n + (y_2n - y)/15 # Equation 17.2.3
    
    return y_new

#-------------------------------
        
n_step = 201 # 200 steps + starting point
x = np.linspace(-20, 20, n_step)
y = np.zeros(len(x))
y[0] = 1 # y(-20) = 1
h=np.median(np.diff(x)) # length of step
diff = rk4_step(dydx, x, y, h) - rk4_stepd(dydx,x,y,h)

def ytrue(x):
    c0 = y[0]/np.exp(np.arctan(x[0]))
    return c0 * np.exp(np.arctan(x))

# plt.plot(x, ytrue(x))
# plt.plot(x, rk4_step(dydx, x, y, h))
# plt.plot(x, rk4_stepd(dydx, x, y, h))
print('-------------------------------')
print('(1)')
print('rk4_step error = ', np.std(ytrue(x) - rk4_step(dydx, x, y, h)))
print('rk4_stepd error = ', np.std(ytrue(x) - rk4_stepd(dydx, x, y, h)))

# Something wrong here - the stepd error should be smaller.

print('-------------------------------')

#-----------------------------------------------------------------------------
# (Q2)

half_lives = np.array([1.41e17, 2.0822e6, 24120, 7.74e12,
              2.38e12, 5.05e10, 3.3e5, 186, 1608,
              1194, 164.3e-6, 7.03e8, 1.58e8, 1.20e7])*3.171e-08 # years

def U238_products(half_lives, t, N0):
    
    def decays(t, N, half_lives = half_lives):
        # for i in range(1, len(half_lives)):
        #     dydx=np.zeros(len(half_life)+1)
        #     dydx[0]=-y[0]/half_life[0]
        #     dydx[1]=y[0]/half_life[0]-y[1]/half_life[1]
        #     dydx[2]=y[1]/half_life[1]
            
        # dNdt = np.zeros(len(half_lives)+1)
        
        # dNdt[0] = -N[i-1]/half_lives[i-1]
        # dNdt[1] = -N[i]/half_lives[i] + N[i-1]/half_lives[i-1]
        # dNdt[2] = N[i]/half_lives[i]
        
        dNdt = []
        
        for i in range(1, len(half_lives)):
        
            dNdt_init = -N[i-1]/half_lives[i]
            dNdt_both = -N[i]/half_lives[i] + N[i-1]/half_lives[i-1]
            dNdt_fin = N[i]/half_lives[i]
            
            dNdt.append(dNdt_init)
            dNdt.append(dNdt_both)
            dNdt.append(dNdt_fin)
        
        dNdt = np.array(dNdt)
        
    solve_ode = integrate.solve_ivp(decays, [min(t), max(t)], N0, method = 'Radau', t_eval = t)
    
    return solve_ode
        
t = np.logspace(0, 9, 1000) # 1000 time points over 1e9 (1 billion) years
N0 = np.zeros(len(half_lives))
N0[0] = 1
# N = U238_products(half_lives, t, N0)

#-----------------------------------------------------------------------------
# (Q3)

data = np.loadtxt('dish_zenith.txt')
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
plt.ion()
ax.scatter(data[:,0], data[:,1], data[:,2])