# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 11:47:03 2021

@author: Ian Hendricksen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def norm(d):
    n = len(d)
    norms = np.zeros(n)
    for i in range(n):
        norms[i] = (d[i] - max(d))/(max(d)-min(d))
    return norms

#-----------------------------------------------------------------------------
# (Q1)

print('=======================')
print('(Q1)')

def shift(y, dy):
    # Shift an array y by dy using a *convolution*. Note that this only
    # works when dy is some integer of N (ex. if we have 1000, smallest
    # shift is 1, since this is the smallest multiple of 1000)
    
    assert(type(dy) is int)
    
    N = len(y)    
    y_ft = np.fft.fft(y)
    
    imp = np.zeros(N)
    imp[dy] = 1 
    imp_ft = np.fft.fft(imp)
    
    conv = np.fft.ifft(imp_ft * y_ft)
    return conv
    
def gauss(x, mu, sigma, amp, offset):
    return offset + amp*np.exp(-(x - mu)**2/(2*sigma**2))

x1 = np.linspace(-5, 5, 1000) # x1 --> "x for problem 1"
g1 = gauss(x1, 0, 0.5, 1, 0) # g1 --> "gauss for problem 1"
g1_shift = shift(g1, int(len(x1)/2) + 250)

# f1 = plt.figure()
# plt.plot(x1, g1, label = 'Initial Gaussian')
# plt.plot(x1, abs(g1_shift.real), label = 'Shifted Gaussian')
# plt.legend()
# plt.savefig('gauss_shift.png')

print('This currently shifts by 2.5 not by 5 - FIX!')

#-----------------------------------------------------------------------------
# (Q2)

print('=======================')
print('(Q2)')

def corr(f, g):
    f_ft = np.fft.fft(f)
    g_ft_conj = np.conj(np.fft.fft(g))
    fg_corr = np.fft.ifft(f_ft * g_ft_conj)
    return np.fft.fftshift(fg_corr)

x2 = np.linspace(-10, 10, 1000)
g2 = gauss(x2, 0, 2, 10, 0)
g_corr = corr(g2, g2)

# f2 = plt.figure()
# plt.plot(x2, g2/max(g2), label = 'Gaussian') # Normalized
# plt.plot(x2, abs(g_corr)/max(abs(g_corr)), label = 'Correlation of Gaussian with Itself')
# plt.legend()
# plt.savefig('gauss_corr.png')
    
#-----------------------------------------------------------------------------
# (Q3)

print('=======================')
print('(Q3)')

x3 = np.linspace(-5, 5, 1000)
g3 = gauss(x1, 0, 0.5, 1, 0)
g3_shift = shift(g3, int(len(x3)/2) + 250)
g3_shift_corr = corr(g3, g3_shift)

# f3 = plt.figure()
# plt.plot(x3, g3)
# plt.plot(x3, abs(g3_shift)/max(abs(g3_shift)))
# plt.plot(x3, abs(g3_shift_corr)/max(abs(g3_shift_corr)))

#-----------------------------------------------------------------------------
# (Q4)

print('=======================')
print('(Q4)')

def conv_safe(f, g, N_zeros):
        
    if len(f) > len(g):
        zeros_f = np.zeros(N_zeros)
        zeros_g = np.zeros(N_zeros + len(f) - len(g))
    if len(g) > len(f):
        zeros_f = np.zeros(N_zeros + len(g) - len(f))
        zeros_g = np.zeros(N_zeros)
    else: # i.e. len(f) == len(g)
        zeros_f = np.zeros(N_zeros)
        zeros_g = np.zeros(N_zeros)
        
    extd_f = np.concatenate((f, zeros_f)) # "extended f"
    extd_g = np.concatenate((g, zeros_g))
    
    extd_f_ft = np.fft.fft(extd_f)
    extd_g_ft = np.fft.fft(extd_g)
    
    conv = np.fft.ifft(extd_f_ft * extd_g_ft)
    
    return conv

x4 = np.linspace(-10, 10, 1000)
g4 = gauss(x4, 0, 2, 1, 0)
N_zeros = 10
g4_conv = conv_safe(g4, g4, N_zeros) # maybe want to try 2 different gaussians?
x4_extd = np.linspace(min(x4), max(x4) + N_zeros, len(x4) + N_zeros)

# f4 = plt.figure()
# plt.plot(x4, g4)
# plt.plot(x4_extd, g4_conv)

#-----------------------------------------------------------------------------
# (Q5)

print('=======================')
print('(Q5)')

x5 = np.linspace(-500, 500, 1000)
k = 50.125
sin = np.sin(2*np.pi*k*x5/len(x5))
sin_ft = np.fft.fft(sin) # np.fft.fftshift(np.fft.fft(sin))

def sin_ft_anl(y, k): # "sin fourier transform analytical"
    N = len(y)
    kp = np.arange(N)
    J = np.complex(0, 1)
    F = np.empty(N, dtype = complex)
    
    for i in range(N):
        num1= 1 - np.cos(2*np.pi*(k - kp[i])) - J*np.sin(2*np.pi*(k - kp[i]))
        den1 = 1 - np.cos(2*np.pi*(k - kp[i])/N) - J*np.sin(2*np.pi*(k - kp[i])/N)
        num2 = 1 - np.cos(2*np.pi*(k + kp[i])) + J*np.sin(2*np.pi*(k + kp[i]))
        den2 = 1 - np.cos(2*np.pi*(k + kp[i])/N) + J*np.sin(2*np.pi*(k + kp[i])/N)
        F[i] = (1/(2*J))*((num1/den1) - (num2/den2))
    
    return F

anl_sin_ft = sin_ft_anl(sin, k)

f5c = plt.figure()
plt.plot(abs(sin_ft), label = 'FFT') # Negative frequencies aliased to high frequencies
plt.plot(abs(anl_sin_ft), label = 'Analytical')
plt.legend()
plt.savefig('sin_FFT_and_analytical.png')

fqs5 = np.arange(len(x5))

print('Difference in peak values b/w fft and analytical is', fqs5[np.argmax(abs(sin_ft[0:int(len(fqs5)/2)]))]
      - fqs5[np.argmax(abs(anl_sin_ft[0:int(len(fqs5)/2)]))])

def cos_wind(x):
    return 0.5-0.5*np.cos(2*np.pi*x/len(x))

sin_win = sin*cos_wind(x5)
sin_win_ft = np.fft.fft(sin_win)
    
# Things are currently not very well normalized here. This goes for the
# entire script really. Need to fix this

f5d = plt.figure()
plt.plot(abs(sin_ft)/len(sin_ft), label = 'No window')
plt.plot(abs(sin_win_ft)/len(sin_win_ft), label = 'With window')
plt.legend()

#-----------------------------------------------------------------------------
# (Q6)

print('=======================')
print('(Q6)')

n = 100
x6 = np.arange(n) # the random walk has x-axis as just integer steps
rand_walk = np.cumsum(np.random.randn(n))
rand_walk_ft = np.fft.fft(rand_walk * cos_wind(x6))
rand_walk_ft = np.delete(rand_walk_ft, 0) # remove k=0

f6 = plt.figure()
plt.plot(abs(rand_walk_ft))
# plt.plot(max(abs(rand_walk_ft))*(1/np.arange(n)**2))