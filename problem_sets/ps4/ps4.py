# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 14:53:56 2021

@author: Ian Hendricksen
"""

import numpy as np
import matplotlib.pyplot as plt
import camb

#-----------------------------------------------------------------------------
# (Q1)

# Answers are recorded on the pdf document for this problem set.

#-----------------------------------------------------------------------------
# (Q2)

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
    
    y_model_0 = fun(m)
    y_model_0 = y_model_0[:len(y_data)]
    chisq_0 = np.sum(((y_model_0 - y_data)/y_errs)**2) # fun(m)[0] = y_model
    print('chisq_0 is', chisq_0)
    m_new = m
    lamda = 0
    Ninv = np.eye(len(y_data))*(1/y_errs**2)
    
    for i in range(n):
        
        if m_new[3] < 0.01:
            m_new[3] = 0.01
        
        y_model = fun(m_new)
        y_model = y_model[:len(y_data)]
        derivs_model = deriv_fun(fun, m_new)
        derivs_model = derivs_model[:len(y_data)]
        
        r = y_data - y_model
        
        # dm = (A'^T N^-1 A' + lamda @ diag(A'^T N^-1 A'))^-1 @ A'^T N^-1 r
        dm = np.linalg.pinv(derivs_model.T @ Ninv @ derivs_model + lamda * 
                            np.diag(np.diag(derivs_model.T @ Ninv @ derivs_model))) @ derivs_model.T @ Ninv @ r
        
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
        
        else: 
            lamda = lamda * 2
            
        print('m_new is', m_new)
        print('new chisq is', chisq_step)
            
        # Get m errors
        
        cov_mat = np.linalg.pinv(curv)
        m_new_errs = np.sqrt(np.diag(cov_mat))
            
    return m_new, m_new_errs, curv

m_lvmq, m_lvmq_errs, curv_lvmq = lvmq(m_guess, get_spectrum, get_spectrum_derivs, ell, spec, errs, 10)

f = open('planck_fit_params.txt', 'w')
f.write(f'Best-fit parameters are {m_lvmq}')
f.write(f'Parameter errors are {m_lvmq_errs}')
f.close()

"""Make sure to write parameters and errors to planck_fit_params.txt!!!"""
    
#-----------------------------------------------------------------------------
# (Q3)

def mcmc(m, fun, curv, x_data, y_data, n):
    
    # m: model parameters (guess)
    # fun: function to fit
    # curv: curvature matrix from lvmq
    # n: number of steps
    # assert(curv.shape[0] == n) # Should have as many rows in curv as we have steps n --> edit: maybe not?
    
    chisq_0 = np.sum((fun(m)[0] - y_data)**2) # fun(m)[0] = y_model
    chain = np.zeros([len(n), len(m)]) # n_step x n_m
    chisq_vals = np.zeros(len(n))
    
    for i in range(n):
        # m_trial = curv[:, i] # this needs to be checked
        chisq_trial = np.sum((fun(m)[0] - y_data)**2)
        d_chisq = chisq_trial - chisq_0
        prob = np.exp(-0.5*d_chisq**2)
        
        # if chisq_trial < chisq_0:
            
            
#-----------------------------------------------------------------------------
# (Q4)