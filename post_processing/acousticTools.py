#!/usr/bin/python3

#Utility packages
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

G = lambda f,r : np.exp(1j*k(f)*r)/(4*np.pi*r)

def monopolePressure(f,r,theta=0,phi=0,demid=0):
    return(-1j*omega(f)*Q*rho*G(f,r))

def dipolePressure(f,r,theta,phi,demid):
    return(PM(f,rplus(demid,r,theta,phi)) - PM(f,rminus(demid,r,theta,phi)))

def infinitesimalDipolePressure(f,r,theta,phi,demid=0):
    return((dipoleMomentum*omega(f)*rho*k(f))*np.sin(theta)*np.cos(phi)*G(f,r)*(1/(1j*k(f)*r) - 1))

###Analytical acoutic functions
analyticalFunctions = {}
analyticalFunctions["monopole"] = monopolePressure
analyticalFunctions["dipole"] = dipolePressure
analyticalFunctions["infinitesimalDipole"] = infinitesimalDipolePressure