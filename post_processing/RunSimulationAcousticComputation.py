#!/usr/bin/python3.8

#System packages
import subprocess
import os
import sys
import time
import csv

#Utility packages
import numpy as np

#Mesh packages
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) + "/scripts")
from MeshTools import generateSphericMesh, generateCircularMesh
from MeshToolsUncertainty import generateSphericMeshUncertainty

### Studied function
function = "monopole"
gradient = 0

### Studied parameters
elementType = "P0"

verificationRadius = 0.5
verificationSize = 2*verificationRadius
verificationResolution = np.round(2*np.pi*verificationRadius/100,4)

Frequencies = [100,500,1000,5000]
Radius = [0.1*verificationRadius,0.25*verificationRadius,0.5*verificationRadius]
Resolutions = [0.01*verificationRadius,0.025*verificationRadius,0.05*verificationRadius,0.075*verificationRadius,0.1*verificationRadius] 

DipoleDistances = [0.05*verificationRadius,0.1*verificationRadius,0.15*verificationRadius,0.25*verificationRadius,0.5*verificationRadius,0.75*verificationRadius,0.9*verificationRadius] if function == "dipole" else [0.0]

SigmasPosition = [0.001*verificationRadius,0.0025*verificationRadius,0.005*verificationRadius,0.0075*verificationRadius,0.01*verificationRadius,0.025*verificationRadius]
#SigmasMeasure = [0.01,0.05,0.1]
SigmasMeasure = [0.0]
Nsigma = 20

parametersCombinations = len(Frequencies)*len(Radius)*len(Resolutions)*len(DipoleDistances)*max(1,Nsigma*len(np.nonzero(SigmasPosition)[0]))*max(1,Nsigma*len(np.nonzero(SigmasMeasure)[0]))

if(__name__ == "__main__"):

    #Get computation method
    method = "BEM"
    try:
        method = sys.argv[1]
    except IndexError:
        print("Invalid resolution method, switching to default : " + method)

    if(method == "BEM"):
        command = "ff-mpirun -np 4 " + os.path.dirname(os.path.abspath(__file__)) + "/AcousticComputationBEM.edp -wg"
    elif(method == "SFT"):
        command = "FreeFem++ " + os.path.dirname(os.path.abspath(__file__)) + "/AcousticComputationSFT.edp"
    else:
        method = "BEM"
        command = "ff-mpirun -np 4 " + os.path.dirname(os.path.abspath(__file__)) + "/AcousticComputationBEM.edp -wg"
        print("Invalid resolution method, switching to default : " + method)

    counter = 1

    #Generate verification mesh
    if(gradient == 0):
        if(not os.path.exists(os.path.dirname(os.path.dirname(os.path.abspath(__name__))) + "/config/meshes/circle/P1/" + str(verificationSize) + "_" + str(verificationResolution) + ".mesh")):
                generateCircularMesh(verificationSize,verificationResolution,elementType="P1",saveMesh=True)
    elif(gradient == 1):
        if(not os.path.exists(os.path.dirname(os.path.dirname(os.path.abspath(__name__))) + "/config/meshes/sphere/" + elementType + "/" + str(verificationSize) + "_" + str(verificationResolution) + ".mesh")):
                generateSphericMesh(verificationSize,verificationResolution,elementType=elementType,saveMesh=True)

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
                                try:
                                    generateSphericMeshUncertainty(size,resolution,sigmaPosition,elementType=elementType,saveMesh=True) #Generate a new random mesh !
                                except:
                                    print("Mesh generation failed, skipping computation")
                                    continue

                            if(not os.path.exists(os.path.dirname(os.path.dirname(os.path.abspath(__name__))) + "/config/meshes/sphere/" + elementType + "/" + str(size) + "_" + str(resolution) + ".mesh")):
                                try:
                                    generateSphericMesh(size,resolution,elementType=elementType,saveMesh=True)  #No need to generate a new mesh if it already exists
                                except:
                                    print("Mesh generation failed, skipping computation")
                                    continue

                            for frequency in Frequencies:

                                for dipoleDistance in DipoleDistances:

                                    if(dipoleDistance >= size):
                                        break

                                    print("Combination : " + str(counter) + "/" + str(parametersCombinations))
                                    print("Sigma position : " + str(sigmaPosition) + " m")
                                    print("Sigma measure : " + str(sigmaMeasure) + " Pa")
                                    print("Frequency : " + str(frequency) + " Hz")
                                    print("Size : " + str(size) + " m")
                                    print("Resolution : " + str(resolution) + " m")
                                    if(function == "dipole"):
                                        print("Dipole distance : " + str(dipoleDistance) + " m")

                                    counter += 1

                                    ### Run computation

                                    tmpFileID = " -fileID " + str(i*max(1,Nsigma*len(np.nonzero(SigmasMeasure)[0])) + j + 1)

                                    bashCommand = command + " -realMeasurements 0 -frequency " + str(frequency) + " -size " + str(size) + " -resolution " + str(resolution) + " -dipoleDistance " + str(dipoleDistance) + " -sigmaPosition " + str(sigmaPosition) + " -sigmaMeasure " + str(sigmaMeasure) + tmpFileID + " -verificationSize " + str(verificationSize) + " -verificationResolution " + str(verificationResolution) + " -studiedFunction " + function + " -DelementType=" + elementType + " -Dgradient=" + str(gradient) + " -ns"
                                    print(bashCommand)

                                    t0 = time.time()
                                    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                                    output, error = process.communicate()
                                    t1 = time.time()

                                    with open("computation_time_" + function + "_" + elementType + ".csv","a") as f:
                                        if(function == "dipole"):
                                            data = [str(frequency),str(resolution),str(size),str(dipoleDistance),str(sigmaPosition),str(sigmaMeasure),str(t1-t0)]
                                        else:
                                            data = [str(frequency),str(resolution),str(size),str(sigmaPosition),str(sigmaMeasure),str(t1-t0)]
                                        writer = csv.writer(f)
                                        writer.writerow(data)

                                    #DEBUG
                                    print(output.decode())

                                    killProcess = "killall FreeFem++-mpi"
                                    process = subprocess.Popen(killProcess.split(), stdout=subprocess.PIPE)
                                    output, error = process.communicate()

                                    #DEBUG
                                    print(output.decode())
