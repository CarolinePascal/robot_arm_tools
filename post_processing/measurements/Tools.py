import numpy as np

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