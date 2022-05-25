from xml.etree.ElementTree import PI
import numpy as np

###Acoustic parameters

c = 343.4
rho = 1.204
Q = 1
dipoleMomentum = 1

###Acoustic dipole functions

rplus = lambda demid,r,theta,phi : np.sqrt(r**2 + demid**2 - 2*r*demid*np.sin(theta)*np.cos(phi))
rminus = lambda demid,r,theta,phi : np.sqrt(r**2 + demid**2 + 2*r*demid*np.sin(theta)*np.cos(phi))
drplusr = lambda demid,r,theta,phi : (r - demid*np.sin(theta)*np.cos(phi))/rplus(demid,r,theta,phi)
drminusr = lambda demid,r,theta,phi : (r + demid*np.sin(theta)*np.cos(phi))/rminus(demid,r,theta,phi)

G = lambda f,r : np.exp(-1j*(2*np.pi*f/c)*r)/(4*np.pi*r)
dGr = lambda f,r : -G(f,r)*(1j*(2*np.pi*f/c)*r + 1)/r 

def P(f,demid,r,theta,phi):
    if(demid == 0):
        k = 2*np.pi*f/c
        return((-dipoleMomentum*rho*c*k**2)*np.sin(theta)*np.cos(phi)*G(f,r)*(1 + 1/(1j*k*r)))
    else:
        omega = 2*np.pi*f
        return((1j*omega*rho*Q) * (G(f,rplus(demid,r,theta,phi)) - G(f,rminus(demid,r,theta,phi))))

def dPn(f,demid,r,theta,phi):
    if(demid == 0):
        k = 2*np.pi*f/c
        return((1j*dipoleMomentum*rho*c*k)*np.sin(theta)*np.cos(phi)*(G(f,r)/(r**2))*((k*r - 1j)**2 - 1))
    else:
        omega = 2*np.pi*f
        return((1j*omega*rho*Q) * (dGr(f,rplus(demid,r,theta,phi))*drplusr(demid,r,theta,phi) - dGr(f,rminus(demid,r,theta,phi))*drminusr(demid,r,theta,phi)))

analyticalFunctions = {}
analyticalFunctions["P"] = P
analyticalFunctions["dPn"] = dPn

###Finite differences

def DPn(f,demid,r,theta,phi,deltar,order = 1):
    if(order == 1):
        return((-1*P(f,demid,r,theta,phi) + P(f,demid,r+deltar,theta,phi))/deltar)
    elif(order == 2):
        return((-(3/2)*P(f,demid,r,theta,phi) + 2*P(f,demid,r+deltar,theta,phi) - (1/2)*P(f,demid,r+2*deltar,theta,phi))/deltar)

###2D Interpolation tool

def interpolationPhi(phi,phiList,r,theta,f,demid,deltar,order = 1):
    i0 = np.where(phiList<phi)[0][-1]
    phi0 = phiList[i0]
    try:
        i1 = np.where(phiList>phi)[0][0]
        phi1 = phiList[i1]
    except:
        i1 = 0
        phi1 = 2*np.pi + phiList[0]

    delta = (DPn(f,demid,r,theta,phi1,deltar,order) - DPn(f,demid,r,theta,phi0,deltar,order))/(phi1 - phi0)
    return(DPn(f,demid,r,theta,phi0,deltar,order) + delta*(phi-phi0))
