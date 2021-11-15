# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 13:08:03 2021

@author: Ian Hendricksen
"""

import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# (1)

rand_points = np.loadtxt('rand_points.txt')
x = rand_points[:, 0]
y = rand_points[:, 1]
z = rand_points[:, 2]

# f1 = plt.figure()
# ax = f1.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, s = 1)

np_rand_points = np.random.rand(len(x), 3)
X = np_rand_points[:, 0]
Y = np_rand_points[:, 1]
Z = np_rand_points[:, 2] 

# f2 = plt.figure()
# ax = f2.add_subplot(111, projection='3d')
# ax.scatter(X, Y, Z, s = 1)

# If time, check to see if you can get the last bit of the question to work.

#-----------------------------------------------------------------------------
# (2)

n = 100000
y = np.pi*(np.random.rand(n)-0.5)
x = np.tan(y)

# P(x) = 1/(1+x^2) Lorentzian goes to 1 at 0, so this is better than a 
# power law, which goes to inf at 0 and becomes difficult to deal with.

P = 1.5/(1+x**2)
Pp = np.random.rand(n)*P

bins=np.linspace(0,10,501)
cents=0.5*(bins[1:]+bins[:-1])

lor = 1.5/(1 + cents**2)
exp = np.exp(-cents)

f3 = plt.figure()
plt.scatter(x, Pp, label = 'Randomized points', c = 'C1')
plt.plot(cents, lor, label = 'Lorenztian bound', c = 'g')
plt.plot(cents, exp, label = 'Exponential', c = 'r')
plt.xlim(0, 10)
plt.legend()
plt.savefig('rand_points_under_bound.png')

accept = Pp < np.exp(-x)
x_acc = x[accept]

hist, bin_edges = np.histogram(x_acc, bins)
hist = hist/np.sum(hist)
exp = exp/np.sum(exp)

f4 = plt.figure()
plt.scatter(cents, hist, marker = '.')
plt.plot(cents, exp)

# How efficient can I make this? (Part of the question)

#-----------------------------------------------------------------------------#
# (3)