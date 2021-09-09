# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:03:39 2021

@author: Ian Hendricksen
"""

import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# (Q1)

# (a)



# (b)



#-----------------------------------------------------------------------------
# (Q2)

def ndiff(fun, x, full=False):
        
    eps = 1e-7 # Assumption
    
    fp = np.zeros(len(x))
    dx = np.zeros(len(x))
    err = np.zeros(len(x))
    
    for i in range(len(x)):
        
        if x[i]==0: x[i] = 1e-10 # Arbitrary offset
        
        xc = x[i] 
        dx[i] = eps**(1/3)*xc # Following from Numerical Recipes
                
        temp = x[i] + dx[i]
        dx[i] = temp - x[i]
        
        fp[i] = (fun(temp) - fun(x[i]-dx[i]))/(2*dx[i]) # fp --> "f prime"        
        
        # NEED TO ADD ERROR
        
        err[i] = 10 #temporary
    
    if full==False:
        return fp
    else:
        return fp, dx, err

#-----------------------------------------------------------------------------
# (Q3)



#-----------------------------------------------------------------------------
# (Q4)
