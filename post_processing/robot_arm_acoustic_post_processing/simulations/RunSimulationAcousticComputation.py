#!/usr/bin/python3.8

#System packages
import subprocess
import os
import sys
import time
import csv
import cloup

#Utility packages
import numpy as np

#Mesh packages
from robot_arm_acoustic.MeshTools import generateSphericMesh, generateCircularMesh

### Studied function
function = "monopole"
gradient = 0

### Studied parameters
elementType = "P1"

verificationRadius = 0.5
verificationSize = 2*verificationRadius
verificationResolution = np.round(2*np.pi*verificationRadius/100,4)

Frequencies = [100,250,500,750,1000,2500,5000]
Radius = [0.1*verificationRadius,0.25*verificationRadius,0.5*verificationRadius]
Resolutions = [0.01*verificationRadius,0.025*verificationRadius,0.05*verificationRadius,0.075*verificationRadius,0.1*verificationRadius] 

DipoleDistances = [0.05*verificationRadius,0.1*verificationRadius,0.15*verificationRadius,0.25*verificationRadius,0.5*verificationRadius,0.75*verificationRadius,0.9*verificationRadius] if function == "dipole" else [0.0]

SigmasPosition = [0.001*verificationRadius,0.0025*verificationRadius,0.005*verificationRadius,0.0075*verificationRadius,0.01*verificationRadius,0.025*verificationRadius]
SigmasMeasure = [0.01,0.05,0.1]
Nsigma = 20

parametersCombinations = len(Frequencies)*len(Radius)*len(Resolutions)*len(DipoleDistances)*max(1,Nsigma*len(np.nonzero(SigmasPosition)[0]))*max(1,Nsigma*len(np.nonzero(SigmasMeasure)[0]))

@cloup.command()
@cloup.option('--method', type=str, default="BEM", help="Computation method (BEM or SFT)")
@cloup.option('--gradient', is_flag=True, help="Compute the gradient of the acoustic field")
def main(method, gradient):

    #Generate command (bash)
    if(method == "SFT"):
        command = "FreeFem++ ../AcousticComputationSFT.edp"
    else:
        command = "ff-mpirun -np 4 ../AcousticComputationBEM.edp -wg"

    counter = 1

    #Generate verification mesh
    os.makedirs("./meshes/",exist_ok=True)
    if(not gradient):
        generateCircularMesh(verificationSize,verificationResolution,elementType="P1",saveMesh=True,saveFolder="./meshes/")
    else:
        generateSphericMesh(verificationSize,verificationResolution,elementType=elementType,saveMesh=True,saveFolder="./meshes/")

    for radius in Radius:
        for resolution in Resolutions:

            ### Generate computation mesh
            size = 2*radius
            try:
                generateSphericMesh(size,resolution,elementType=elementType,saveMesh=True,saveFolder="./meshes/")  #No need to generate a new mesh if it already exists
            except:
                print("Mesh generation failed, skipping computation")
                continue

            for sigmaMeasure in SigmasMeasure:
                for i in range(Nsigma):
                    if(sigmaMeasure == 0.0 and i > 0):
                        break

                    for sigmaPosition in SigmasPosition:
                        for j in range(Nsigma):
                            if(sigmaPosition == 0.0 and j > 0):
                                break
  
                            ### Generate noisy computation mesh
                            if(sigmaPosition != 0.0):
                                try:
                                    generateSphericMesh(size,resolution,sigmaPosition,elementType=elementType,saveMesh=True,saveFolder="./meshes/") #Generate a new random mesh !
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

                                    tmpFileID = " -fileID " + str((1 + j) + (Nsigma*i))

                                    bashCommand = command + " -realMeasurements 0 -frequency " + str(frequency) + " -size " + str(size) + " -resolution " + str(resolution) + " -dipoleDistance " + str(dipoleDistance) + " -sigmaPosition " + str(sigmaPosition) + " -sigmaMeasure " + str(sigmaMeasure) + tmpFileID + " -verificationSize " + str(verificationSize) + " -verificationResolution " + str(verificationResolution) + " -studiedFunction " + function + " -DelementType=" + elementType + " -Dgradient=" + str(gradient) + " -ns"
                                    print(bashCommand)

                                    t0 = time.time()
                                    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                                    output,_ = process.communicate()
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
                                    output,_ = process.communicate()

                                    #DEBUG
                                    print(output.decode())


if(__name__ == "__main__"):
    main()
    