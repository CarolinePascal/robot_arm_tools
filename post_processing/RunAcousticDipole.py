import subprocess
import os

from numpy.lib.function_base import meshgrid

from ThickSphereMesh import *

import numpy as np
import matplotlib.pyplot as plt
import glob

from AcousticDipoleTools import *

###STUDY PARAMETERS
R0 = 0.2
Rtest = 1.0
dipoleDistance = 0

layers = 1
elementType = "P1"

Vertices = [10,15,20,25,30]
Frequencies = [20,50,200,500,2000,5000,20000]

DeltaRFactors = np.linspace(0.5,5,50)

###CREATE NEW FOLDER

outputFolderList = np.array(glob.glob("Baseline_*_/"))

maxFolder = 0
for folder in outputFolderList:
    folderIndex = int(folder.split("_")[-2])
    if(folderIndex > maxFolder):
        maxFolder = folderIndex

outputFolder = "Baseline_" + str(maxFolder+1) + "_"
os.mkdir(outputFolder)

###LAUNCH COMPUTATIONS

for vertex in Vertices:
    print("Vertices : " + str(vertex))
    Ntheta = vertex
    Nphi = vertex

    for frequency in Frequencies:
        print("Frequency : " + str(frequency))

        for factor in DeltaRFactors:
            l = c/frequency
            deltaR = np.round(l*factor,5)
            print("DeltaR : " + str(deltaR))

            #Create mesh if it does not exist
            mesh = ThickSphericMesh(Ntheta,Nphi,R0,np.round(R0+deltaR,5),layers)
            mesh.write("gmsh")

            outputFilePath = outputFolder + "/output_" + elementType[1] + "_" + str(layers) + "_" + str(vertex) + "_" + str(frequency) + "_" + str(dipoleDistance) + "_" + str(R0) + "_" + str(deltaR) + "_" + str(Rtest) + ".txt"

            bashCommand = "FreeFem++ AcousticDipole.edp -frequency " + str(frequency) + " -dipoleDistance " + str(dipoleDistance) + " -R0 " + str(R0) + " -Rtest " + str(Rtest) + " -element " + str(elementType) + " -mesh " + mesh.path + " -output " + outputFilePath
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

            #DEBUG
            #print(output.decode())
