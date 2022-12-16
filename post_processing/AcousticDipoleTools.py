import numpy as np

###Acoustic parameters

c = 343.4
rho = 1.204
Q = 1
dipoleMomentum = 1

###Acoustic dipole functions

rplus = lambda demid,r,theta,phi : np.sqrt(r**2 + demid**2 - 2*r*demid*np.sin(theta)*np.cos(phi))
rminus = lambda demid,r,theta,phi : np.sqrt(r**2 + demid**2 + 2*r*demid*np.sin(theta)*np.cos(phi))

k = lambda f: 2*np.pi*f/c
omega = lambda f: 2*np.pi*f

G = lambda f,r : np.exp(1j*k(f)*r/10)/(4*np.pi*r)

def PM(f,r):
    return(-1j*omega(f)*Q*rho*G(f,r))

def PMn(f,r):
    return(1j*omega(f)*G(f,r))

def P(f,demid,r,theta,phi):
    if(demid == 0):
        return((dipoleMomentum*omega(f)*rho*k(f))*np.sin(theta)*np.cos(phi)*G(f,r)*(1/(1j*k(f)*r) - 1))
    else:
        return(PM(f,rplus(demid,r,theta,phi)) - PM(f,rminus(demid,r,theta,phi)))

analyticalFunctions = {}
analyticalFunctions["P"] = P
analyticalFunctions["PM"] = PM
analyticalFunctions["PMn"] = PMn
