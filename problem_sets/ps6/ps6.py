# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 12:11:42 2021

@author: Ian Hendricksen
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from glob import glob
# from scipy.signal import tukey

#-----------------------------------------------------------------------------
# This section will contain general functions to be called at any necessary 
# point in the loop.

# Here you can find the directory of the data if necessary:

data_dir = 'C:\\Users\\Admin\\Downloads\\MSc Year 1 (Downloads)\\Phys 512\\phys512_ih\\problem_sets\\ps6\\LOSC_Event_tutorial\\LOSC_Event_tutorial'
os.chdir(data_dir)

def read_template(filename):
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    th=template[0]
    tl=template[1]
    return th,tl

def read_file(filename):
    dataFile=h5py.File(filename,'r')
    dqInfo = dataFile['quality']['simple']
    qmask=dqInfo['DQmask'][...]
    meta=dataFile['meta']
    gpsStart=meta['GPSstart'][()]
    utc=meta['UTCstart'][()]
    duration=meta['Duration'][()]
    strain=dataFile['strain']['Strain'][()]
    dt=(1.0*duration)/len(strain)
    dataFile.close()
    return strain,dt,utc

def tukey_window(N, alpha):
    """
    I was very determined to write my own Tukey Window! This is my super
    convoluted version that is completely unnecessary. The limits are not
    defined in the same way as the actual function, but hey, it works (barely)!
    Doesn't appear to work for alpha < 0.25. Use (or don't) with caution.
    
    N: length of window
    alpha: the "width" of the flat part. Becomes a cos window for alpha = 1
           and a rect window for alpha = 0
    """
    n = np.arange(0, N)
    cos = 0.5 * (1 + np.cos(-np.pi + 2*np.pi*n[0:int(alpha*(N)/2)]/(alpha*(N))))
    if N % 2 == 0: # even N condition
        rect = np.ones(int(N/2 - (alpha*(N)/2)))
        left = np.concatenate((cos, rect))
        right = left[::-1]
    else: # odd N condition
        rect_l = np.ones(int(N/2 - (alpha*(N)/2))+2)
        rect_r = np.ones(int(N/2 - (alpha*(N)/2)) + 1)
        left = np.concatenate((cos, rect_l))
        right = np.concatenate((rect_r, cos[::-1]))
    window = np.concatenate((left, right))
    assert(len(window) == N)
    return window

#-----------------------------------------------------------------------------
# Now we will loop through each file, and print results for each iteration.

all_templates = glob('*template.hdf5')

for template in all_templates:
    print('==========================================================')
    th, tl = read_template(template)
    template = template.split('_')[0]
    print(f'-----------------Using template {template}-----------------')
    
    # Assign files to their templates (could make this less brutish later):
    if template == 'GW150914':
        h_file = 'H-H1_LOSC_4_V2-1126259446-32.hdf5'
        l_file = 'L-L1_LOSC_4_V2-1126259446-32.hdf5'
    elif template == 'GW151226':
        h_file = 'H-H1_LOSC_4_V2-1135136334-32.hdf5'
        l_file = 'L-L1_LOSC_4_V2-1135136334-32.hdf5'
    elif template == 'GW170104':
        h_file  = 'H-H1_LOSC_4_V1-1167559920-32.hdf5'
        l_file = 'L-L1_LOSC_4_V1-1167559920-32.hdf5'
    elif template == 'LVT151012':
        h_file = 'H-H1_LOSC_4_V2-1128678884-32.hdf5'
        l_file = 'L-L1_LOSC_4_V2-1128678884-32.hdf5'
    
    hl_files = [h_file, l_file]
    
    # Now we will go through our analysis for H and L for every template:
    SNRs = []
    t_arrs = []
    
    for file in hl_files:
        print('==========================================================')
        print(f'Reading file {file} --> {file[0]}')
        strain, dt, utc = read_file(file)
        
        print('---------------------------------')
        print('(a) Determining noise model for', file[0])
                
        # Since we are looping through each event for both H and L,
        # we are generating noise models for H & L separately.
        
        # Apply window and take FT of strain and template:
        print('Windowing...')
        
        # We use the Tukey Window I created at the top so that we
        # force the edges of our data to go to 0, but we have a "band"
        # in the center of the data of some width (set by alpaha) so
        # that the majority of the data retains its original scale after
        # applying the window.
        
        win = tukey_window(len(strain), alpha = 0.25)
        st_w_ft = np.fft.rfft(strain*win)
        if file[0] == 'H':
            tft = np.fft.rfft(th*win)
        else:
            tft = np.fft.rfft(tl*win)
                
        # Smooth data
        print('Smoothing...')
            
        ftsq = abs(st_w_ft)**2 # |FT|^2

        smooth = 10
        ftsq_smooth = np.convolve(ftsq, np.ones(smooth)/smooth, mode = 'same')
        # st_ws_ft = st_w_ft / np.sqrt(ftsq_smooth)
        
        # ftsq_smooth = np.zeros(len(ftsq))
        # for i in range(100): 
        #     ftsq_smooth=(ftsq_smooth+np.roll(ftsq_smooth,1)+np.roll(ftsq_smooth,-1))/3 # --> this is actually doing the smoothing! need a better method
        
        # plt.plot(ftsq_smooth, linestyle = '--')
        # plt.plot(ftsq)
        
        # Whiten FT of windowed strain and template:
        print('Whitening...')
        
        # Need to normalize?
        
        st_wsw_ft = st_w_ft / np.sqrt(ftsq_smooth) # strain FT windowed, smoothed, whitened
        tft_white = tft / np.sqrt(ftsq)
        
        print('---------------------------------')
        print('(b) Searching event')
        print('Creating matched filter...')
        
        mf = np.fft.fftshift(np.fft.ifft(st_wsw_ft * np.conj(tft_white)))
        f = plt.figure()
        plt.plot(abs((mf)), label = f'{file}')
        plt.legend()
        
        print('---------------------------------')
        print('(c) Estimating noise for', file[0])
        
        noise_est = np.std(abs(mf)) # i.e. noise is assumed to be scatter in mf
        print('Scatter in the matched filter is', noise_est)
        SNR_est = abs(mf/noise_est)
        SNR_est_max = SNR_est.max()
        print('Max SNR is', SNR_est_max)
        SNRs.append(SNR_est_max) # Need to print the combined SNRs somewhere
    
        print('---------------------------------')
        print('(d) Comparing calculated to analytical SNR\'s for', file[0])
        
        # sigmasq = 1*(template_fft * template_fft.conjugate() / power_vec).sum() * df
        # sigma = np.sqrt(np.abs(sigmasq))
        # SNR_complex = optimal_time/sigma
        
        freqs = np.fft.fftfreq(len(tft_white))
        df = freqs[1] - freqs[0]
        
        sig = np.sqrt(abs((tft_white*np.conj(tft_white)/ftsq).sum()))
        SNR_an = (mf/ftsq) / sig
        SNR_an_max = abs(max(SNR_an))
        print('Difference between analytical & estimated SNR is', SNR_est_max - SNR_an_max) # This can't be right
        
        print('---------------------------------')
        print('(e) Finding half-weight frequency for', file[0])
        
        print('---------------------------------')
        print('(f) Determining arrival time for', file[0])
        
        t_arr = dt*(np.argmax(SNR_est)) # Not necessarily arrival time! This isn't a spike, but a round peak
        t_arrs.append(t_arr)
        print(f'Arrival time is {t_arr}s')
        
        
        

# Make sure to write everything that is printed and save all relevant info

# # Changing back to my ps6 directory:
# cwd = 'C:\\Users\\Admin\\Downloads\\MSc Year 1 (Downloads)\\Phys 512\\phys512_ih\\problem_sets\\ps6'
# os.chdir(cwd)