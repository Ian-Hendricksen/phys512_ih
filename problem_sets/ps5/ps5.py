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
    
    delta = np.zeros(N)
    delta[dy-1] = 1 
    delta_ft = np.fft.fft(delta)
    
    conv = np.fft.ifft(delta_ft * y_ft)
    return conv
    
def gauss(x, mu, sigma, amp, offset):
    return offset + amp*np.exp(-(x - mu)**2/(2*sigma**2))

x1 = np.linspace(-5, 5, 10000) # x1 --> "x for problem 1"
g1 = gauss(x1, -2.5, 0.5, 1, 0) # g1 --> "gauss for problem 1"
g1_shift = shift(g1, int(len(x1)/2))

# f1 = plt.figure()
# plt.plot(x1, g1, label = 'Initial Gaussian')
# plt.plot(x1, abs(g1_shift), label = 'Shifted Gaussian')
# plt.legend()
# plt.savefig('gauss_shift.png')

#-----------------------------------------------------------------------------
# (Q2)

print('=======================')
print('(Q2)')

def corr(f, g):
    f_ft = np.fft.fft(f)
    g_ft_conj = np.conj(np.fft.fft(g))
    fg_corr = np.fft.ifft(f_ft * g_ft_conj)
    return fg_corr

x2 = np.linspace(-10, 10, 1000)
g2 = gauss(x2, 0, 2, 0.1, 0)
g_corr = corr(g2, g2)

# f2 = plt.figure()
# plt.plot(x2, g2, label = 'Gaussian')
# g_corr = np.fft.fftshift(g_corr)
# plt.plot(x2, abs(g_corr), label = 'Correlation of Gaussian with Itself')
# plt.legend()
# plt.savefig('gauss_corr.png')
    
#-----------------------------------------------------------------------------
# (Q3)

print('=======================')
print('(Q3)')

x3 = np.linspace(-5, 5, 1000)
g3 = gauss(x3, -2.5, 0.5, 0.1, 0)
g3_shift = shift(g3, int(len(x3)/4))
g3_shift_corr = corr(g3, g3_shift)

# f3 = plt.figure()
# plt.plot(x3, g3, label = 'Original Gaussian')
# plt.plot(x3, abs(g3_shift), label = 'Shifted Gaussian')
# plt.plot(x3, abs(g3_shift_corr), label = 'Shifted Gaussian Correlated with Itself')
# plt.legend()
# plt.savefig('gauss_shift_corr.png')

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
g4 = gauss(x4, 0, 2, 0.1, 0)
N_zeros = 1000
g4_conv = conv_safe(g4, g4, N_zeros) # maybe want to try 2 different gaussians?
x4_extd = np.linspace(min(x4), max(x4) + N_zeros, len(x4) + N_zeros)

# f4 = plt.figure()
# plt.plot(abs(g4))
# plt.plot(abs(g4_conv))
# plt.savefig('conv_safe.png')

#-----------------------------------------------------------------------------
# (Q5)

print('=======================')
print('(Q5)')

print('------')
print('(b)')

def multiples(N):
    mults = []
    not_mults = np.arange(1,N+1)
    for i in range(1, N):
        div = f'{N/i}'
        intg, rem = div.split('.')
        if float(rem) == 0:
            mults.append(int(intg))
    mults.append(1)
    mults = np.array(mults[::-1])
    index = []
    for i in range(len(mults)):
        index.append(np.where(not_mults == mults[i])[0][0])
    not_mults = np.delete(not_mults, index)
    return mults, not_mults

def complex_sum(k, N):
    J = complex(0, 1)
    num = 1 - np.cos(2*np.pi*k) + J*np.sin(2*np.pi*k)
    den = 1 - np.cos(2*np.pi*k/N) + J*np.sin(2*np.pi*k/N)
    return num/den

N = 100
k_mults, k_not_mults = multiples(N)
f = open('non_multiples_k.txt', 'w')
f.write(f'N = {N}')
f.write(f'\nThe multiples of N are {k_mults}')
f.write(f'\nThe non-multiple integers of N are {k_not_mults}')
f.write('\n==========================================================')
f.write('\nThe following are the values of the sum from question 5(a)')
f.write('\nfor integers k that are not a multiple of N:')
f.write('\n')
for i in range(len(k_not_mults)):
    f.write(f'\n{abs(complex_sum(k_not_mults[i], N))}')
f.close()
        
print('------')
print('(c)')

x5 = np.arange(100)
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

# f5c = plt.figure()
# plt.plot(abs(sin_ft), label = 'FFT') # Negative frequencies aliased to high frequencies
# plt.plot(abs(anl_sin_ft), label = 'Analytical')
# plt.legend()
# plt.savefig('sin_FFT_and_analytical.png')

print('Standard deviation between FFT and analytical DFT is ', np.std(abs(sin_ft) - abs(anl_sin_ft)))
print('Average residual is ', np.mean(abs(sin_ft) - abs(anl_sin_ft)))

print('------')
print('(d)')

def cos_wind(x):
    return 0.5-0.5*np.cos(2*np.pi*x/len(x))

sin_win = sin*cos_wind(x5)
sin_win_ft = np.fft.fft(sin_win)
    
# f5d = plt.figure()
# plt.plot(abs(sin_ft), label = 'No window')
# plt.plot(abs(sin_win_ft), label = 'With window')
# plt.legend()
# plt.savefig('sin_FFT_window.png')

print('------')
print('(e)')

cos_wind_ft = np.fft.fft(cos_wind(x5))
N_cos_wind = len(cos_wind(x5))

f = open('cos_wind_ft.txt', 'w')
f.write('DFT of Cosine Window Function:')
f.write(f'\nN = {N_cos_wind}')
f.write(f'\nThe first element in the DFT of the window is {abs(cos_wind_ft[0])}, which should be N/2 = {N_cos_wind/2}')
f.write(f'\nThe second element in the DFT of the window is {abs(cos_wind_ft[1])}, which should be N/4 = {N_cos_wind/4}')
f.write(f'\nThe last element in the DFT of the window is {abs(cos_wind_ft[-1])}, which should be N/4 = {N_cos_wind/4}')
f.write('\nThe rest of the elements in the DFT should be ~0, and are:')
f.write(f'\n{abs(cos_wind_ft)[2:-1]}')
f.close()

sin_neighb_ft = np.zeros(N_cos_wind)
for i in range(1, N_cos_wind-1):
    sin_neighb_ft[i] = -(1/4)*abs(sin_ft)[i-1] + (1/2)*abs(sin_ft)[i] + -(1/4)*abs(sin_ft)[i+1]

f5d = plt.figure()
plt.plot(abs(sin_ft), label = 'No window')
plt.plot(abs(sin_win_ft), label = 'With window')
plt.plot(abs(sin_neighb_ft), label = 'No window scaled by neighbors')
plt.legend()
plt.savefig('sin_ft_neighbors.png')

#-----------------------------------------------------------------------------
# (Q6)

print('=======================')
print('(Q6)')

n = 100
x6 = np.arange(n) # the random walk has x-axis as just integer steps
rand_walk = np.cumsum(np.random.randn(n))
rand_walk_ft = np.fft.fft(rand_walk * cos_wind(x6))
rand_walk_ft = np.delete(rand_walk_ft, 0) # remove k=0
rand_walk_ps = abs(rand_walk_ft)**2

f6 = plt.figure()
plt.plot(rand_walk_ps, label = 'Random Walk PS')
plt.plot((1/np.arange(1, n)**2)*max(rand_walk_ps), label = 'Scaled $1/k^2$')
plt.xlim(-1, 20)
plt.legend()
# plt.savefig('rw_ps.png')