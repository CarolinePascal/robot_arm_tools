import subprocess
import os

from numpy.lib.function_base import meshgrid

from ThickSphereMesh import createMesh

import numpy as np
import matplotlib.pyplot as plt

import glob

#TODO Name folder according to tested parameters ?
folderList = np.array(glob.glob("Derivative_*/"))

maxFolder = 0
for folder in folderList:
    folderIndex = int(folder.split("_")[1][:-1])
    if(folderIndex > maxFolder):
        maxFolder = folderIndex

os.mkdir("Derivative_"+str(maxFolder+1))

Rmin = 0.2

#DeltaR = np.round(np.logspace(-3,0,10),3)
#Frequencies = np.round(np.logspace(np.log10(20),np.log10(20000),10),0)
#Vertices = np.round(np.logspace(1,2,10),0).astype(int)

Vertices = [10,15,20,25,30]
Frequencies = [50,500,5000]
DeltaR = np.round(np.logspace(-3,-0.3,20),5)

for vertex in Vertices:
    print("Vertices : " + str(vertex))
    Ntheta = vertex
    Nphi = vertex

    for frequency in Frequencies:
        print("Frequency : " + str(frequency))

        for delta in DeltaR:
            print("DeltaR : " + str(delta))
            Rmax = np.round(Rmin + delta,5)

            if(not os.path.isfile("meshes/thick_sphere/TS_"+str(Ntheta)+"_"+str(Rmin)+"_"+str(Rmax)+".mesh")):
                createMesh(Ntheta,Nphi,Rmin,Rmax)

            bashCommand = "FreeFem++ DerivativeComparaison.edp -frequency " + str(frequency) + " -angularVerticesNumber " +str(Ntheta) + " -Rmin " + str(Rmin) + " -Rmax " + str(Rmax) + " -folderName Derivative_" + str(maxFolder+1) 
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

            #DEBUG
            #print(output.decode())

    
