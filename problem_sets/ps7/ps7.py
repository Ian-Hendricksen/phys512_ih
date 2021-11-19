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

#-----------------------------------------------------------------------------
# (2)

n = 100000
y = np.pi*(np.random.rand(n)-0.5)
x = np.tan(y)

# P(x) = 1/(1+x^2) Lorentzian goes to 1 at 0, so this is better than a 
# power law, which goes to inf at 0 and becomes difficult to deal with.

P = 1.1/(1+x**2)
Pp = np.random.rand(n)*P

bins=np.linspace(0, 10, 51) 
cents=0.5*(bins[1:] + bins[:-1])

lor = 1.1/(1 + cents**2)
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

accept_inds = np.where(accept == True)[0]
acc_percent = 100*len(accept_inds)/n

print(f'(2) From {n} points, acceptance rate is {acc_percent}%.')

hist, bin_edges = np.histogram(x_acc, bins, range = (0, 10))
hist = hist/np.sum(hist)
exp = exp/np.sum(exp)

f4 = plt.figure()
plt.bar(cents, hist, 0.15, label = 'Histogram of Deviates')
plt.plot(cents, exp, c = 'r', label = 'Exponential')
plt.legend()
# plt.savefig('hist_deviates_10000.png') 

# How efficient can I make this? --> n = 10000 appears to be lower limit
# for deviates to be considered reasonably exponentially distributed. Could
# use a bounding function that is closer in shape to an exponential, but 
# the lorentzian does a pretty good job of this as is. A Gaussian ends up
# dipping below the exponential and requires a relatively large sigma to
# remain above the exp, and a power law goes to infinity at 0, undesirable 
# behavior. The lorentzian is greater than the exp for all values (at least
# those I use here) and reasonably follows a decaying exponential.

#-----------------------------------------------------------------------------
# (3)

u = np.linspace(0, 1, 1000)
u = u[1:]
# u < exp(-x), where x = v/u
# ln(u) = -x = -v/u --> v = -uln(u)

v = -u*np.log(u)
vmax = max(v)
limits = [min(v), max(v)]
print('limits on v are', limits)

n2 = 100000
u2 = np.random.rand(n2)
v2 = np.random.rand(n2)*vmax
x2 = v2/u2
accept2 = u2 < np.exp(-x2)
x2_acc = x2[accept2]

accept2_inds = np.where(accept2 == True)[0]
acc_percent = 100*len(accept2_inds)/n2

print(f'(3) From {n2} points, acceptance rate is {acc_percent}%.')

# This provides a good visualization for what's going on:
    
f6 = plt.figure()
plt.plot(u, v, c = 'r')
plt.plot(u, -v, c = 'r')
plt.scatter(u2,v2)
plt.plot(u,v, c='r')
plt.scatter(u2[accept2], v2[accept2])
plt.savefig('uv_bound.png')

hist2, bin_edges2 = np.histogram(x2_acc, 51, range = (0, 3))
cents2 = 0.5*(bin_edges2[1:] + bin_edges2[:-1])
hist2 = hist2/np.sum(hist2)

# Maybe we have to multiply cents2 by 2 in the exponential here because
# we have to consider the 2 regions in u,v space? In any case, this seems to
# make things work!

exp2 = np.exp(-2*cents2)*np.sum(accept2)*(cents2[2]-cents2[1]) 
exp2 = exp2/np.sum(exp2)

f7 = plt.figure()
plt.bar(cents2, hist2, 0.05, label = 'Histogram of Deviates')
plt.plot(cents2, exp2, c = 'r', label = 'Exponential')
plt.legend()
# plt.savefig('ratio_of_uniforms_hist_100000.png')