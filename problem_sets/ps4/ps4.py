# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 14:53:56 2021

@author: Ian Hendricksen
"""

import numpy as np
import matplotlib.pyplot as plt
import camb
import datetime

t1 = datetime.datetime.now()

#-----------------------------------------------------------------------------
# (Q1)

# Answers are recorded on the pdf document for this problem set.

#-----------------------------------------------------------------------------
# (Q2)

print('=======================')
print('(Q2)')

planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0] # x data
spec=planck[:,1] # y data
errs=0.5*(planck[:,2]+planck[:,3])

def get_spectrum(pars,lmax=3000):
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]
    return tt[2:]

m_guess = np.array([69,0.022,0.12,0.06,2.1e-9,0.95])
# m_guess = np.array([65,0.02,0.1,0.07,2.00e-9,0.97])

def get_spectrum_derivs(fun, m):
    """
    A little convoluted and perhaps confusingly written, what this for loop does
    is to implement the 4-point method to differentiate w.r.t. each parameter,
    hence the range(n_m). For the ith parameter, we calculate a small offset
    dm, and 4 new -sets- of parameters, each applying the offset in a slightly
    different way (whether it is + or - or a factor of 1 or 2). These new sets
    of parameters, each with a different offset on the -ith- parameter, are
    used to calculate the derivative for every set of "y" points in the 
    function get_spectrum, w.r.t. the ith derivative. The loop then does the 
    same for every parameter, and returns the full derivs matrix.
    """
    
    n_data = len(fun(m))
    n_m = len(m)
    derivs = np.zeros([n_data, n_m])
    
    for i in range(n_m):
        eps = 1e-16
        dm = eps**(1/6)*m[i] # <-- if this becomes a problem, make dm bigger
        
        m1 = m
        m1[i] = m[i]+dm
        fdp = fun(m1)
        
        m2 = m
        m2[i] = m[i]-dm
        fdm = fun(m2)
        
        m3 = m
        m3[i] = m[i] + 2*dm
        f2dp = fun(m3)
        
        m4 = m
        m4[i] = m[i] - 2*dm
        f2dm = fun(m4)
        
        # print('-----')
        # print(m1)
        # print(m2)
        # print(m3)
        # print(m4)
        # print('-----')
        
        fp = (8*(fdp - fdm) - (f2dp - f2dm))/(12*dm)
        derivs[:, i] = fp
        
    return derivs

# derivs_m_guess = get_spectrum_derivs(get_spectrum, m_guess)          

# Levenberg - Marquardt fitter:

def lvmq(m, fun, deriv_fun, x_data, y_data, y_errs, n):
    
    # m: model parameters (guess)
    # fun: function to fit
    # n: number of iterations
    
    t1 = datetime.datetime.now()
    
    print('==============')
    print('---LVMQ Fit---')
    print('==============')
    
    y_model_0 = fun(m)
    y_model_0 = y_model_0[:len(y_data)]
    chisq_0 = np.sum(((y_model_0 - y_data)/y_errs)**2) # fun(m)[0] = y_model
    print('chisq_0 is', chisq_0)
    m_new = m
    lamda = 0
    Ninv = np.eye(len(y_data))*(1/y_errs**2)
    
    for i in range(n):
        
        print('-------------')
        print('Step', i)
        
        if m_new[3] < 0.01:
            m_new[3] = 0.01
        
        y_model = fun(m_new)
        y_model = y_model[:len(y_data)]
        derivs_model = deriv_fun(fun, m_new)
        derivs_model = derivs_model[:len(y_data)]
        
        r = y_data - y_model
        
        # dm = (A'^T N^-1 A' + lamda @ diag(A'^T N^-1 A'))^-1 @ A'^T N^-1 r
        dm = np.linalg.inv(derivs_model.T @ Ninv @ derivs_model + lamda * 
                            np.diag(np.diag(derivs_model.T @ Ninv @ derivs_model))
                            ) @ derivs_model.T @ Ninv @ r
        
        print('d_H0 =', dm[0])
        print('d_ombh2 =', dm[1])
        print('d_omch2 =', dm[2])
        print('d_tau =', dm[3])
        print('d_As =', dm[4])
        print('d_ns =', dm[5])
        print('lamda = ', lamda)
            
        y_model_step = fun(m_new + dm)
        y_model_step = y_model_step[:len(y_data)]
        # derivs_model_step = deriv_fun(fun, m_new + dm)
        chisq_step = np.sum(((y_model_step - y_data)/errs)**2)
        
        # Get curvature matrix: 2 * derivs.T @ N^-1 @ derivs --> ignoring first term.
        curv = 2 * derivs_model.T @ Ninv @ derivs_model
        
        # Update lamda, and m_new if applicable
        
        if chisq_step < chisq_0: # success
            lamda = lamda / 2
            if lamda < 0.5:
                lamda = 0
            m_new = m_new + dm
            chisq_0 = chisq_step
        
        else: 
            if lamda < 0.01:
                lamda = 1
            else:
                lamda = lamda * 2
            
        print('m_new is', m_new)
        print('new chisq is', chisq_step)
        dof = len(y_data) - len(m_new)
        x = (chisq_step - dof)/np.sqrt(2*dof)
        print(f'new chisq is {x} std devs from mean ({dof})')
            
        # Get m errors
        
        cov_mat = np.linalg.inv(curv)
        m_new_errs = np.sqrt(np.diag(cov_mat))
        
    t2 = datetime.datetime.now()
    print(f'LVMQ runtime was {t2-t1}')
    print('==============')
                    
    return m_new, m_new_errs, curv

m_lvmq, m_lvmq_errs, curv_lvmq = lvmq(m_guess, get_spectrum, get_spectrum_derivs, ell, spec, errs, 25)

f = open('planck_fit_params.txt', 'w')
f.write('Levenberg-Marquardt Parameters:')
f.write(f'\nBest-fit parameters are {m_lvmq}')
f.write(f'\nParameter errors are {m_lvmq_errs}')
f.close()

np.savetxt('m_lvmq.txt', m_lvmq)
np.savetxt('curv_lvmq.txt', curv_lvmq)

# This is temporary so I can play with mcmc without running lvmq:

# curv_lvmq = np.loadtxt('curv_lvmq.txt')
    
#-----------------------------------------------------------------------------
# (Q3)

print('=======================')
print('(Q3)')

def mcmc(m, fun, curv, x_data, y_data, y_errs, n, m_priors = None, m_priors_errs = None):
    
    # m: model parameters (guess)
    # fun: function to fit
    # curv: curvature matrix from lvmq
    # n: number of steps
    
    t1 = datetime.datetime.now()
    
    print('==============')
    print('---MCMC Fit---')
    print('==============')
    
    nm = len(m)
    
    y_model_0 = fun(m)
    y_model_0 = y_model_0[:len(y_data)]
    
    # Check if we have priors, and if so, include likelihood of priors:
    
    if m_priors is None:
        chisq_0 = np.sum(((y_model_0 - y_data)/y_errs)**2)
    else:
        chisq_0 = np.sum(((y_model_0 - y_data)/y_errs)**2) + np.sum(((m - m_priors)/m_priors_errs)**2)
    
    print('chisq_0 is', chisq_0)
    
    chain = np.zeros([n, len(m)]) # n_step x n_m
    chisq_vals = np.zeros(n)
    
    L = np.linalg.cholesky(np.linalg.inv(curv)) # L = stepsize
    
    for i in range(n):
        
        print('-------------')
        print('Step', i)
        
        m_trial = m + 0.1 * L @ np.random.randn(nm) # draw trial steps from curvature
                                                    # matrix from lvmq; need to scale it? --> yes
        if m_priors is None:
            if m_trial[3] < 0.01:
                m_trial[3] = 0.01 # This constraint makes my chains happier
            
        print('m_trial is', m_trial)
            
        y_trial = fun(m_trial)
        y_trial = y_trial[:len(y_data)]
        
        # Again check if we have priors and include likelihood if so:
        
        if m_priors is None:
            chisq_trial = np.sum(((y_trial - y_data)/y_errs)**2)
        else:
            chisq_trial = np.sum(((y_trial - y_data)/y_errs)**2) + np.sum(((m - m_priors)/m_priors_errs)**2)
        
        print('chisq_trial is', chisq_trial)
        
        d_chisq = chisq_trial - chisq_0
        prob = np.exp(-0.5*d_chisq**2)
        
        if np.random.randn(1) < prob:
            m = m_trial
            chisq_0 = chisq_trial
            print('chisq accepted')
        chain[i, :] = m
        chisq_vals[i] = chisq_0
        
    t2 = datetime.datetime.now()
    print(f'MCMC runtime was {t2-t1}')
    print('==============')    
    
    return chain, chisq_vals

chain1_mcmc, chisq1_mcmc = mcmc(m_guess, get_spectrum, curv_lvmq, ell, spec, errs, 5000)

mcmc_info_merged = np.concatenate((chisq1_mcmc.reshape(chain1_mcmc.shape[0], 1), chain1_mcmc), axis = 1)
np.savetxt('planck_chain.txt', mcmc_info_merged)

# Determine if chain has converged:
    
f1 = plt.figure()

plt.subplot(1,2,1)
plt.plot(np.arange(chain1_mcmc.shape[0]), chain1_mcmc[:, 0]) # Should be white noise

plt.subplot(1,2,2)
plt.plot(np.arange(chain1_mcmc.shape[0]), abs(np.fft.fft(chain1_mcmc[:, 0])))
plt.yscale('log'); plt.xscale('log'); # plt.xlim(1e-3, 1e3)

plt.savefig("./H0 chain and fft 5000 its.png")

m_mcmc = np.zeros(chain1_mcmc.shape[1])
m_errs_mcmc = np.zeros(chain1_mcmc.shape[1])
for i in range(chain1_mcmc.shape[1]):
    m_mcmc[i] = np.mean(chain1_mcmc[:, i])
    m_errs_mcmc[i] = np.std(chain1_mcmc[:, i])
    
print(f'MCMC parameters are {m_mcmc}')
print(f'MCMC errors are {m_errs_mcmc}')

# Calculate om_lamda and propagate errors:

hsq = (1/100)**2 * (m_mcmc[0])**2 # H0 = m_mcmc[0], h = H0/100
omb = m_mcmc[1]/hsq
omc = m_mcmc[2]/hsq
om_lamda = 1 - omb - omc

hsq_err = abs(2 * hsq * (1/m_mcmc[0]) * m_errs_mcmc[0])
omb_err = omb * np.sqrt((m_errs_mcmc[1]/m_mcmc[1])**2 + (hsq_err/hsq)**2)
omc_err = omb * np.sqrt((m_errs_mcmc[2]/m_mcmc[2])**2 + (hsq_err/hsq)**2)
om_lamda_err = np.sqrt(omb_err**2 + omc_err**2)

print(f'Dark energy density is {om_lamda} with error {om_lamda_err}')

f = open('om_lamda.txt', 'w')
f.write(f'Best-fit for dark energy density (om_lamda) is {om_lamda}')
f.write(f'\nDark energy density error is {om_lamda_err}')
f.close()
            
#-----------------------------------------------------------------------------
# (Q4)

print('=======================')
print('(Q4)')

# First construct our priors arrays to feed to mcmc:

tau_prior = 0.0540
tau_prior_err = 0.0074

m_priors = np.zeros(len(m_guess))
m_priors[3] = tau_prior
m_priors_errs = np.zeros(len(m_guess)) + 1e15
m_priors_errs[3] = tau_prior_err

# Run the chain and save:

chain_mcmc_priors, chisq_mcmc_priors = mcmc(m_guess, get_spectrum, curv_lvmq, ell, spec, errs, 5000, m_priors, m_priors_errs)

mcmc_prior_info_merged = np.concatenate((chisq_mcmc_priors.reshape(chain_mcmc_priors.shape[0], 1), chain_mcmc_priors), axis = 1)
np.savetxt('planck_chain_tauprior.txt', mcmc_prior_info_merged)

m_mcmc_priors = np.zeros(chain_mcmc_priors.shape[1]) # "m from mcmc using priors"
m_errs_mcmc_priors = np.zeros(chain_mcmc_priors.shape[1])
for i in range(chain_mcmc_priors.shape[1]):
    m_mcmc_priors[i] = np.mean(chain_mcmc_priors[:, i])
    m_errs_mcmc_priors[i] = np.std(chain_mcmc_priors[:, i])
    
print(f'MCMC w/ priors parameters are {m_mcmc_priors}')
print(f'MCMC w/ priors errors are {m_errs_mcmc_priors}')

f = open('planck_fit_params_mcmc_priors.txt', 'w')
f.write('MCMC with priors:')
f.write(f'\nBest-fit parameters are {m_mcmc_priors}')
f.write(f'\nParameter errors are {m_errs_mcmc_priors}')
f.close()

# Check for convergence:
    
f2 = plt.figure()

plt.suptitle('H0 and its FFT for MCMC w/ Priors')

plt.subplot(1,2,1)
plt.plot(np.arange(chain_mcmc_priors.shape[0]), chain_mcmc_priors[:, 0]) # Should be white noise

plt.subplot(1,2,2)
plt.plot(np.arange(chain_mcmc_priors.shape[0]), abs(np.fft.fft(chain_mcmc_priors[:, 0])))
plt.yscale('log'); plt.xscale('log'); # plt.xlim(1e-3, 1e3)

plt.savefig("./H0 chain and fft 5000 its w tau prior.png")

# Importance sample our original chain:
    
chisq_imp_samp = np.zeros(len(chisq_mcmc_priors))

for i in range(len(chisq_mcmc_priors)):
    # This is essentially returning delta chisq for for the parameters
    # from each step in the chain:
    chisq_imp_samp[i] = np.sum(((chain_mcmc_priors[i, :] - m_priors)/m_priors_errs)**2)

weight = np.exp(-0.5 * chisq_imp_samp)

m_imp_samp = np.zeros(len(m_mcmc_priors))
for i in range(len(m_mcmc_priors)):
    m_imp_samp[i] = np.sum(weight * chain1_mcmc[:, i])/np.sum(weight)

print(f'Importance sampled parameters are {m_imp_samp}')

f = open('planck_fit_params_imp_samp.txt', 'w')
f.write('Importance sampling original chain in Q3:')
f.write(f'\nImportance sampled parameters are {m_imp_samp}')
f.close()

t2 = datetime.datetime.now()

print(f'Total runtime of ps4 was {t2-t1}')