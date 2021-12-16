import subprocess
import os

from numpy.lib.function_base import meshgrid

from ThickSphereMesh import createMesh

import numpy as np
import matplotlib.pyplot as plt
import glob

folderList = np.array(glob.glob("Baseline*"))

maxFolder = 0
for folder in folderList:
    folderIndex = int(folder[-1])
    if(folderIndex > maxFolder):
        maxFolder = folderIndex

os.mkdir("Baseline_"+str(maxFolder+1))

Rmin = 0.2
Rmax = 0.21
R = 1.0

DeltaR = np.linspace(0.001,1,10)
Frequencies = np.round(np.logspace(np.log10(20),np.log10(20000),5),0).astype(int)
Vertices = np.round(np.logspace(1,2,5),0).astype(int)

fig, ax = plt.subplots()

for vertex in Vertices:
    Error = []
    for frequency in Frequencies:
        Ntheta = vertex
        Nphi = vertex

        if(not os.path.isfile("meshes/thick_sphere/ST_"+str(Ntheta)+"_"+str(Rmin)+"_"+str(Rmax)+".mesh")):
            createMesh(Ntheta,Nphi,Rmin,Rmax)

        bashCommand = "FreeFem++ AcousticDipole.edp -frequency " + str(frequency) + " -angularVerticesNumber " +str(Ntheta) + " -Rmin " + str(Rmin) + " -Rmax " + str(Rmax) + " -R " + str(R) + " -folderName Baseline_" + str(maxFolder+1) 
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        msg = output.decode()
        #print(msg)
        error = float(msg[-msg[::-1].find('\n\n'):-1])
        Error.append(error)
    
    ax.plot(Vertices,np.log10(Error),label=str(frequency) + "Hz")


ax.set_xlabel("Radius delta (m)")
ax.set_xscale('log')  
ax.set_ylabel("log(Average error)")
ax.set_title("Average error depending on the radius delta")    
ax.legend()
plt.show()
