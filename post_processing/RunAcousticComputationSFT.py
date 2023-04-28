#!/usr/bin/python3.8

#System packages
import subprocess
import os
import sys

#Utility packages
import numpy as np

#Mesh packages
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) + "/scripts")
from MeshTools import generateSphericMesh

#Custom acoustic tools package
from acousticTools import analyticalFunctions

### Studied function
function = "monopole"

### Studied parameters
Frequencies = [1000]   #100,250,500,750,1000,2500,5000,7500,10000
Radius = [0.1]   #0.1,0.2,0.3,0.4,0.45,0.495
Resolutions = np.round(np.logspace(np.log10(0.001),np.log10(0.05),30),4)[::-1] #0.005,0.0075,0.01,0.025,0.05,0.075,0.1 #np.round(np.linspace(0.003,0.012,20),4)
parametersCombinations = len(Frequencies)*len(Radius)*len(Resolutions)

if(__name__ == "__main__"):
    counter = 1
    for radius in Radius:
        for resolution in Resolutions:

            ### Generate computation mesh
            
            size = 2*radius

            if(not os.path.exists(os.path.dirname(os.path.dirname(os.path.abspath(__name__))) + "/config/meshes/sphere/" + str(size) + "_" + str(resolution) + ".mesh")):
                generateSphericMesh(size,resolution,saveMesh=True)

            for frequency in Frequencies:

                print("Combination : " + str(counter) + "/" + str(parametersCombinations))
                print("Frequency : " + str(frequency) + " Hz")
                print("Size : " + str(size) + " m")
                print("Resolution : " + str(resolution) + " m")
                counter += 1

                ### Run computation

                bashCommand = "FreeFem++ AcousticComputationSFT.edp -wg -frequency " + str(frequency) + " -size " + str(size) + " -resolution " + str(resolution) + " -studiedFunction " + function + " -ns"
                process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()

                #DEBUG
                print(output.decode())
