# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 17:35:31 2021

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt

mcmc_info_merged = np.loadtxt('planck_chain_tauprior.txt')

chain1_mcmc = mcmc_info_merged[:, 1:7]

m_mcmc = np.zeros(chain1_mcmc.shape[1])
m_errs_mcmc = np.zeros(chain1_mcmc.shape[1])
for i in range(chain1_mcmc.shape[1]):
    m_mcmc[i] = np.mean(chain1_mcmc[:, i])
    m_errs_mcmc[i] = np.std(chain1_mcmc[:, i])
    
f1 = plt.figure()

plt.subplot(1,2,1)
plt.plot(chain1_mcmc[:, 1]) # Should be white noise

plt.subplot(1,2,2)
plt.plot(abs(np.fft.fft(chain1_mcmc[:, 1])))
plt.yscale('log'); plt.xscale('log'); # plt.xlim(1e-3, 1e3)