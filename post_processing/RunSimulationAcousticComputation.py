#!/usr/bin/python3.8

#System packages
import subprocess
import os
import sys

#Utility packages
import numpy as np

#Mesh packages
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) + "/scripts")
from MeshTools import generateSphericMesh, generateCircularMesh
from MeshToolsUncertainty import generateSphericMeshUncertainty

### Studied method
method = "BEM"
try:
    method = sys.argv[1]
except:
    print("Invalid size and resolution, switching to default : " + method)

if(method == "BEM"):
    command = "ff-mpirun -np 4 AcousticComputationBEM.edp -wg"
elif(method == "SFT"):
    command = "FreeFem++ AcousticComputationSFT.edp"

### Studied function
function = "monopole"

### Studied parameters
elementType = "P0"

verificationRadius = 0.5
verificationSize = 2*verificationRadius
verificationResolution = np.round(2*np.pi*verificationRadius/100,4)

Frequencies = [100,250,500,750,1000,2500,5000] 
Radius = [0.1*verificationRadius,0.25*verificationRadius,0.5*verificationRadius]
Resolutions = [0.01*verificationRadius,0.025*verificationRadius,0.05*verificationRadius,0.075*verificationRadius] 

SigmasPosition = [0.0025*verificationRadius,0.005*verificationRadius,0.01*verificationRadius]   
SigmasMeasure = [0.0]
Nsigma = 10

parametersCombinations = len(Frequencies)*len(Radius)*len(Resolutions)*max(1,len(np.nonzero(SigmasPosition)[0]))*max(1,len(np.nonzero(SigmasMeasure)[0]))*Nsigma

if(__name__ == "__main__"):
    counter = 1

    #Generate verification mesh
    if(not os.path.exists(os.path.dirname(os.path.dirname(os.path.abspath(__name__))) + "/config/meshes/circle/P1/" + str(verificationSize) + "_" + str(verificationResolution) + ".mesh")):
            generateCircularMesh(verificationSize,verificationResolution,elementType="P1",saveMesh=True)

    for sigmaMeasure in SigmasMeasure:
        for i in range(Nsigma):
            if(sigmaMeasure == 0.0 and i > 0):
                break
            for sigmaPosition in SigmasPosition:
                for j in range(Nsigma):
                    if(sigmaPosition == 0.0 and j > 0):
                        break
                    for radius in Radius:
                        for resolution in Resolutions:

                            ### Generate computation mesh
                            size = 2*radius

                            if(sigmaPosition != 0.0):
                                generateSphericMeshUncertainty(size,resolution,sigmaPosition,elementType=elementType,saveMesh=True) #Generate a new random mesh !

                            if(not os.path.exists(os.path.dirname(os.path.dirname(os.path.abspath(__name__))) + "/config/meshes/sphere/" + elementType + "/" + str(size) + "_" + str(resolution) + ".mesh")):
                                generateSphericMesh(size,resolution,elementType=elementType,saveMesh=True)  #No need to generate a new mesh if it already exists

                            for frequency in Frequencies:

                                print("Combination : " + str(counter) + "/" + str(parametersCombinations))
                                print("Sigma position : " + str(sigmaPosition) + " m")
                                print("Sigma measure : " + str(sigmaMeasure) + " Pa")
                                print("Frequency : " + str(frequency) + " Hz")
                                print("Size : " + str(size) + " m")
                                print("Resolution : " + str(resolution) + " m")
                                counter += 1

                                ### Run computation

                                tmpFileID = " -fileID " + str(i*Nsigma + j + 1)

                                bashCommand = command + " -frequency " + str(frequency) + " -size " + str(size) + " -resolution " + str(resolution) + " -sigmaPosition " + str(sigmaPosition) + " -sigmaMeasure " + str(sigmaMeasure) + tmpFileID + " -verificationSize " + str(verificationSize) + " -verificationResolution " + str(verificationResolution) + " -studiedFunction " + function + " -DelementType=" + elementType + " -ns"
                                print(bashCommand)
                                process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                                output, error = process.communicate()

                                #DEBUG
                                print(output.decode())

                                killProcess = "killall FreeFem++-mpi"
                                process = subprocess.Popen(killProcess.split(), stdout=subprocess.PIPE)
                                output, error = process.communicate()

                                #DEBUG
                                print(output.decode())
