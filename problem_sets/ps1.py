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
    xc = x # But what if x = 0? --> worry later
    dx = eps**(1/3)*xc # Following from Numerical Recipes
        
    temp = x + dx
    dx = temp - x
        
    fp = (fun(temp) - fun(x-dx))/(2*dx)
        
    err = 10
    
    if full==False:
        return fp
    else:
        return fp, dx, err

#-----------------------------------------------------------------------------
# (Q3)



#-----------------------------------------------------------------------------
# (Q4)
