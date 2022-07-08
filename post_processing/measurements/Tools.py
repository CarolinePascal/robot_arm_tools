import numpy as np
import measpy as mp

def moduloWithBounds(x,boundMin,boundMax):
    try:
        iter(x)
        return(np.array([modulo(item) for item in x]))
    except:
        if(x >= boundMin and x < boundMax):
            return(x)
        elif(x < boundMin):
            return(moduloWithBounds(x + (boundMax - boundMin),boundMin,boundMax))
        else:
            return(moduloWithBounds(x - (boundMax - boundMin),boundMin,boundMax))

def modulo(x):
    return(moduloWithBounds(x,-np.pi,np.pi))

def averageSpectral(spectralList):
    L = len(spectralList)
    for i,item in enumerate(spectralList):
        fft = item.rfft().values
        if(i==0):
            modulus = np.abs(fft)/L
            phase = np.unwrap(np.angle(fft))/L
        else:
            modulus += np.abs(fft)/L
            phase += np.unwrap(np.angle(fft))/L
    return(spectralList[0].rfft().similar(values=modulus*np.exp(1j*phase)))