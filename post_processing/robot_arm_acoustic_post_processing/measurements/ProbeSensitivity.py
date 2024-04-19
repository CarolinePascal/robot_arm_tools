import numpy as np
from unyt import Unit

### PRESSURE MODEL

def SP(frequency):
    output = (np.sqrt(1 + (40/frequency)**2)*np.sqrt(1 + (50/frequency)**2))/(0.001*44.4*np.sqrt(1 + (frequency/20000)**2))
    output[np.isinf(output)] = 0.0
    return(output)

def PhiP(frequency):
    output = np.pi - np.arctan(28/frequency) - np.arctan(40/frequency) - np.arctan(frequency/20000)
    output[np.isinf(output)] = 0.0
    return(output)

### VELOCITY MODEL

def SV(frequency, corrected = True, high = True):
    if(corrected):
        output = (np.sqrt(1 + (1/frequency)**2)*np.sqrt(1 + (75/frequency)**2))/(42*(high + (1-high)*0.01))
        output[np.isinf(output)] = 0.0
        return(output)
    else:
        output = (np.sqrt(1 + (1/frequency)**2)*np.sqrt(1 + (frequency/800)**2)*np.sqrt(1 + (frequency/5200)**2)*np.sqrt(1 + (75/frequency)**2))/(42*(high + (1-high)*0.01))
        output[np.isinf(output)] = 0.0
        return(output)

def PhiV(frequency, corrected = True):
    if(corrected):
        output = np.pi - np.arctan(5/frequency) - np.arctan(75/frequency)
        output[np.isinf(output)] = 0.0
        return(output)
    else:
        output = np.pi - np.arctan(5/frequency) + np.arctan(frequency/690) + np.arctan(frequency/18000) - np.arctan(75/frequency)
        output[np.isinf(output)] = 0.0
        return(output)

### DEBUG
#import matplotlib.pyplot as plt

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

fminProbe = 10
fmaxProbe = 10000

def dataProcessing(fileName, pressureLabel, velocityLabel, corrected = True, high = True):
    M = ms.Measurement.from_csvwav(fileName.split(".")[0])

    sigP = M.data[pressureLabel] #Pressure
    sigV = M.data[velocityLabel] #Velocity

    fftP = sigP.rfft().filterout([fminProbe,fmaxProbe])

    filterP = SP(fftP.freqs)*np.exp(1j*PhiP(fftP.freqs))
    filterP = fftP.similar(values=filterP,unit=Unit('1'),desc="Pressure PU probe filter")

    outputP = (filterP*fftP).irfft()
    outputP.desc = "Pressure"

    fftV = sigV.rfft().filterout([fminProbe,fmaxProbe])

    filterV = SV(fftV.freqs)*np.exp(1j*PhiV(fftV.freqs))
    filterV = fftV.similar(values=filterV,unit=Unit('1'),desc="Velocity PU probe filter")

    outputV = (filterV*fftV).irfft()
    outputV.desc = "Velocity"

    ### DEBUG
    #outputP.plot()
    #plt.show()

    #outputV.plot()
    #plt.show()

    return(outputP,outputV)