from unittest import result
import numpy as np
from unyt import Unit

### PRESSURE MODEL

def SP(frequency):
    return((np.sqrt(1 + (40/frequency)**2)*np.sqrt(1 + (50/frequency)**2))/(0.001*44.4*np.sqrt(1 + (frequency/20000)**2)))

def PhiP(frequency):
    return(np.pi - np.arctan(28/frequency) - np.arctan(40/frequency) - np.arctan(frequency/20000))

### VELOCITY MODEL

def SV(frequency, corrected = True, high = True):
    if(corrected):
        return((np.sqrt(1 + (1/frequency)**2)*np.sqrt(1 + (75/frequency)**2))/(42*(high + (1-high)*0.01)))
    else:
        return((np.sqrt(1 + (1/frequency)**2)*np.sqrt(1 + (frequency/800)**2)*np.sqrt(1 + (frequency/5200)**2)*np.sqrt(1 + (75/frequency)**2))/(42*(high + (1-high)*0.01)))

def PhiV(frequency, corrected = True):
    if(corrected):
        return(np.pi - np.arctan(5/frequency) - np.arctan(75/frequency))
    else:
        return(np.pi - np.arctan(5/frequency) + np.arctan(frequency/690) + np.arctan(frequency/18000) - np.arctan(75/frequency))
        
### PLOT
import matplotlib.pyplot as plt

#F = np.logspace(1,4)
#fig,ax = plt.subplots(2)
#
#ax[0].plot(F,20*np.log10(SP(F)),label="Pressure")
#ax[0].plot(F,20*np.log10(SV(F)),label="Velocity")
#ax[0].set_xscale("log")
#ax[0].set_xlabel("Frequency (Hz)")
#ax[0].set_ylabel("20log(|S|)")
#ax[0].set_title("Gain")
#ax[0].legend()
#
#ax[1].plot(F,PhiP(F),label="Pressure")
#ax[1].plot(F,PhiV(F),label="Velocity")
#ax[1].set_xscale("log")
#ax[1].set_xlabel("Frequency (Hz)")
#ax[1].set_ylabel("Phase (rad)")
#ax[1].set_title("Phase")
#ax[1].legend()
#
#plt.show()

### DATA PROCESSING

import measpy as ms
from scipy.fft import fft,fftfreq,ifft,rfft,irfft,rfftfreq

def dataProcessing(fileName, pressureLabel, velocityLabel, corrected = True, high = True):
    M = ms.Measurement.from_csvwav(fileName.split(".")[0])

    sigP = M.data[pressureLabel] #Pressure
    sigV = M.data[velocityLabel] #Velocity

    fftP = sigP.rfft()
    IminP = np.argwhere(np.floor(fftP.freqs - 10) == 0)[0][0]

    filterP = np.concatenate((np.zeros(IminP),SP(fftP.freqs[IminP:])*np.exp(1j*PhiP(fftP.freqs[IminP:]))))
    filterP = fftP.similar(values=filterP,unit=Unit('1'),desc="PU probe filter")

    outputP = (filterP*fftP).irfft()
    outputP.desc = "Pressure"

    fftV = sigV.rfft()
    IminV = np.argwhere(np.floor(fftV.freqs - 10) == 0)[0][0]

    filterV = np.concatenate((np.zeros(IminV),SP(fftV.freqs[IminV:])*np.exp(1j*PhiP(fftV.freqs[IminV:]))))
    filterV = fftV.similar(values=filterV,unit=Unit('1'),desc="PU probe filter")

    outputV = (filterV*fftV).irfft()
    outputV.desc = "Velocity"

    #outputP.plot()
    #plt.show()

    return(outputP,outputV)