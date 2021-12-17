import subprocess
import os

from numpy.lib.function_base import meshgrid

from ThickSphereMesh import createMesh

import numpy as np
import matplotlib.pyplot as plt

Rmin = 0.2
Ntheta = 10
Nphi = 10

if(not os.path.isdir("Derivative/")):
    os.mkdir("Derivative/")

DeltaR = np.linspace(0.001,1,10)
Frequencies = np.round(np.logspace(np.log10(20),np.log10(20000),10),0).astype(int)

for frequency in Frequencies:
    Error = []
    for delta in DeltaR:
        Rmax = np.round(Rmin + delta,5)

        if(not os.path.isfile("meshes/thick_sphere/TS_"+str(Ntheta)+"_"+str(Rmin)+"_"+str(Rmax)+".mesh")):
            createMesh(Ntheta,Nphi,Rmin,Rmax)

        bashCommand = "FreeFem++ DerivativeComparaison.edp -frequency " + str(frequency) + " -angularVerticesNumber " +str(Ntheta) + " -Rmin " + str(Rmin) + " -Rmax " + str(Rmax)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        #DEBUG
        #print(output.decode())

    
