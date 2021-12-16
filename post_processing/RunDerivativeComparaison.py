import subprocess
import os

from numpy.lib.function_base import meshgrid

from ThickSphereMesh import createMesh

import numpy as np
import matplotlib.pyplot as plt

Rmin = 0.2
Ntheta = 10
Nphi = 10

DeltaR = np.linspace(0.001,1,10)
Frequencies = np.round(np.logspace(np.log10(20),np.log10(20000),10),0).astype(int)

fig, ax = plt.subplots()

for frequency in Frequencies:
    Error = []
    for delta in DeltaR:
        Rmax = np.round(Rmin + delta,5)

        if(not os.path.isfile("meshes/thick_sphere/ST_"+str(Ntheta)+"_"+str(Rmin)+"_"+str(Rmax)+".mesh")):
            createMesh(Ntheta,Nphi,Rmin,Rmax)

        bashCommand = "FreeFem++ DerivativeComparaison.edp -frequency " + str(frequency) + " -angularVerticesNumber " +str(Ntheta) + " -Rmin " + str(Rmin) + " -Rmax " + str(Rmax)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        msg = output.decode()
        #print(msg)
        error = float(msg[-msg[::-1].find('\n\n'):-1])
        Error.append(error)
    
    ax.plot(DeltaR,np.log10(Error),label=str(frequency) + "Hz")


ax.set_xlabel("Radius delta (m)")
ax.set_xscale('log')  
ax.set_ylabel("log(Average error)")
ax.set_title("Average error depending on the radius delta")    
ax.legend()
plt.show()
