import numpy as np

###Acoustic parameters

c = 343.4
rho = 1.204
Q = 1

###Acoustic dipole functions

rplus = lambda demid,r,theta,phi : np.sqrt(r**2 + demid**2 - 2*r*demid*np.sin(theta)*np.cos(phi))
rminus = lambda demid,r,theta,phi : np.sqrt(r**2 + demid**2 + 2*r*demid*np.sin(theta)*np.cos(phi))
drplusr = lambda demid,r,theta,phi :  (r - demid*np.sin(theta)*np.cos(phi))/rplus(demid,r,theta,phi)
drminusr = lambda demid,r,theta,phi : (r + demid*np.sin(theta)*np.cos(phi))/rminus(demid,r,theta,phi)

G = lambda f,r : np.exp(-1j*(2*np.pi*f/c)*r)/(4*np.pi*r)
dGr = lambda f,r : -G(f,r)*(1j*(2*np.pi*f/c)*r + 1)/r 

P = lambda f,demid,r,theta,phi : (1j*2*np.pi*f*rho*Q) * (G(f,rplus(demid,r,theta,phi)) - G(f,rminus(demid,r,theta,phi)))
dPn = lambda f,demid,r,theta,phi : (1j*2*np.pi*f*rho*Q) * (dGr(f,rplus(demid,r,theta,phi))*drplusr(demid,r,theta,phi) - dGr(f,rminus(demid,r,theta,phi))*drminusr(demid,r,theta,phi))

analyticalFunctions = {}
analyticalFunctions["P"] = P
analyticalFunctions["dPn"] = dPn

###Finite differences

DPn = lambda f,demid,r,theta,phi,deltar : (P(f,demid,r+deltar,theta,phi) - P(f,demid,r,theta,phi))/deltar

###2D Interpolation tool

def interpolationPhi(phi,phiList,r,theta,f,demid,deltar):
    i0 = np.where(phiList<phi)[0][-1]
    phi0 = phiList[i0]
    try:
        i1 = np.where(phiList>phi)[0][0]
        phi1 = phiList[i1]
    except:
        i1 = 0
        phi1 = 2*np.pi + phiList[0]

    delta = (DPn(f,demid,r,theta,phi1,deltar) - DPn(f,demid,r,theta,phi0,deltar))/(phi1 - phi0)
    return(DPn(f,demid,r,theta,phi0,deltar) + delta*(phi-phi0))
