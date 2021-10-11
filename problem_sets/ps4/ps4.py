# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 14:53:56 2021

@author: Admin
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
errs=0.5*(planck[:,2]+planck[:,3]);

def get_spectrum(pars,lmax=3000):
    
    # Spectrum evaluation:
        
    nm = len(pars)
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
    
    # Spectrum derivatives evaluation:
        
    derivs = np.zeros([len(tt[2:]), nm])
    
    derivs[:, 0] # associated with H0
    
    return tt[2:], derivs

# Levenberg - Marquardt fitter, assuming N = I (for now):

def lvmq(m, fun, x_data, y_data, n):
    
    # m: model parameters (guess)
    # fun: function to fit
    # y_model, derivs_model = fun(m) # fun = get_spectrum
    # r = y_data - y_model # y_model = Am
    
    chisq_0 = np.sum((fun(m)[0] - y_data)**2) # fun(m)[0] = y_model
    m_new = m
    lamda = 0
    for i in range(n):
        y_model, derivs_model = fun(m_new)
        r = y_data - y_model
        
        # dm = (A'^T N^-1 A' + lamda @ diag(A'^T N^-1 A'))^-1 @ A'^T N^-1 r
        dm = np.linalg.pinv(derivs_model.T @ derivs_model + lamda * np.diag(np.diag(derivs_model.T @ derivs_model))) @ derivs_model.T @ r
        
        y_model_step, derivs_model_step = fun(m_new + dm)
        chisq_step = np.sum((y_model_step - y_data)**2)
        
        # Get curvature matrix: 2 * derivs.T @ N^-1 @ derivs --> ignoring first term.
        curv = 2 * derivs_model.T @ derivs_model
        
        # Update lamda, and m_new if applicable
        
        if chisq_step < chisq_0: # success
            lamda = lamda / 2
            if lamda < 1:
                lamda = 0
            m_new = m_new + dm
        
        else: 
            lamda = lamda * 2
            
        # Get m errors -- > y_model = A @ m_new, so A = y_model @ m_new^-1
        # Again we are (for the moment) ignoring N
        A = y_model @ np.linalg.pinv(m)
        cov_mat = np.linalg.pinv(A.T @ A)
        m_new_errs = np.sqrt(np.diag(cov_mat))
            
    return m_new, m_new_errs, curv

"""Make sure to write parameters and errors to planck_fit_params.txt!!!"""
    
#-----------------------------------------------------------------------------
# (Q3)



#-----------------------------------------------------------------------------
# (Q4)