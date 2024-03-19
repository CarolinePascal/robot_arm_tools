#!/usr/bin/python3.8

#System packages
import subprocess
import os
import glob
import sys
import csv
import time

#Utility packages
import numpy as np

#Mesh packages
import meshio as meshio

if(__name__ == "__main__"):

    #Get computation method
    method = "BEM"
    try:
        method = sys.argv[1]
    except IndexError:
        print("Invalid size and resolution, switching to default : " + method)

    if(method == "BEM"):
        command = "ff-mpirun -np 4 " + os.path.dirname(os.path.abspath(__file__)) + "/AcousticComputationBEM.edp -wg"
    elif(method == "SFT"):
        command = "FreeFem++ " + os.path.dirname(os.path.abspath(__file__)) + "/AcousticComputationSFT.edp"
    else:
        method = "BEM"
        command = "ff-mpirun -np 4 " + os.path.dirname(os.path.abspath(__file__)) + "/AcousticComputationBEM.edp -wg"
        print("Invalid resolution method, switching to default : " + method)

    #Get measurements mesh
    measurementsMeshPath = ""
    measurementsMeshPathDefault = os.path.abspath("../robotMesh.mesh")
    try:
        measurementsMeshPath = sys.argv[2]
    except IndexError:
        pass

    if(not os.path.isfile(measurementsMeshPath)):
        if(not os.path.isfile(measurementsMeshPathDefault)):
            raise ValueError("Invalid path for measurements mesh")
        else:
            print("Invalid path for measurements mesh, defaulting to : " + measurementsMeshPathDefault)
            measurementsMeshPath = measurementsMeshPathDefault

    #Get measurements data folder
    measurementsDataFolder = ""
    measurementsDataFolderDefault = os.path.abspath("./") + "/"
    try:
        measurementsDataFolder = sys.argv[3]
    except IndexError:
        pass

    if(not os.path.isdir(measurementsDataFolder)):
        if(not os.path.isdir(measurementsDataFolderDefault)):
            raise ValueError("Invalid path for measurements data folder")
        else:
            print("Invalid path for measurements data folder, defaulting to : " + measurementsDataFolderDefault)
            measurementsDataFolder = measurementsDataFolderDefault

    #Get verification mesh
    verificationMeshPath = ""
    verificationMeshPathDefault = os.path.abspath("../../../Verification/Verification/robotMesh.mesh")
    try:
        verificationMeshPath = sys.argv[4]
    except IndexError:
        pass

    if(not os.path.isfile(verificationMeshPath)):
        if(not os.path.isfile(verificationMeshPathDefault)):
            raise ValueError("Invalid path for measurements data folder")
        else:
            print("Invalid path for measurements data folder, defaulting to : " + verificationMeshPathDefault)
            verificationMeshPath = verificationMeshPathDefault

    #Get verification data folder
    verificationDataFolder = ""
    verificationDataFolderDefault = os.path.abspath("../../../Verification/Verification/") + "/" + os.path.basename(os.getcwd()) + "/"
    try:
        verificationDataFolder = sys.argv[5]
    except IndexError:
        pass

    if(not os.path.isdir(verificationDataFolder)):
        if(not os.path.isdir(verificationDataFolderDefault)):
            raise ValueError("Invalid path for measurements data folder")
        else:
            print("Invalid path for measurements data folder, defaulting to : " + verificationDataFolderDefault)
            verificationDataFolder = verificationDataFolderDefault

    #Get studied frequencies
    Frequencies = sorted([int(os.path.basename(file).split(".")[0].split("_")[-1]) for file in glob.glob("data_*.csv")])
    try:
        Frequencies = [int(item) for item in sys.argv[6].split(",")]
    except (IndexError, ValueError):
        print("Invalid frequency, defaulting to : " + str(Frequencies) + " Hz")   

    #Compute gradient value ?
    gradient = 0
    try:
        gradient = int(sys.argv[7])
    except (IndexError, ValueError):
        print("Invalid gradient option value, defaulting to no gradient computation")

    #Infering element type
    #TODO on Freefem side (interpolation, extrapolation)
    #elementType = sys.argv[8]
    elementType = None
    if(elementType == None):
        mesh = meshio.read(measurementsMeshPath)
        Vertices, Faces = mesh.points, mesh.get_cells_type("triangle")

        measurementsNumber = 0
        with open("data_" + str(Frequencies[0]) + ".csv", newline='') as csvfile:
            measurementsNumber = sum(1 for _ in csv.reader(csvfile, delimiter=','))

        if(np.abs(len(Faces) - measurementsNumber) > np.abs(len(Vertices) - measurementsNumber)):
            elementType = "P1"
        else:
            elementType = "P0"
        print("Invalid element type, infered element type is : " + elementType)
    
    for frequency in Frequencies:
        print("Frequency : " + str(frequency) + " Hz")

        ### Run computation

        bashCommand = command + " -frequency " + str(frequency) + " -realMeasurements 1 -measurementsMeshPath " + measurementsMeshPath + " -measurementsDataPath " + measurementsDataFolder + "data_" + str(frequency) + ".csv -verificationMeshPath " + verificationMeshPath + " -verificationDataPath " + verificationDataFolder + "data_" + str(frequency) + ".csv -verificationGradientDataFolder " + verificationDataFolder + "gradient/data_" + str(frequency) + ".csv -DelementType=" + elementType + " -Dgradient=" + str(gradient) + " -ns"
        print(bashCommand)

        t0 = time.time()
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        t1 = time.time()

        with open("computation_time_CL_"+ elementType + ".csv","a") as f:
            data = [str(frequency),str(t1-t0)]
            writer = csv.writer(f)
            writer.writerow(data)

        #DEBUG
        print(output.decode())

        killProcess = "killall FreeFem++-mpi"
        process = subprocess.Popen(killProcess.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        #DEBUG
        print(output.decode())