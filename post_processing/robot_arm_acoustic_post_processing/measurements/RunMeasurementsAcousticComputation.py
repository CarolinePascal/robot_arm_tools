#!/usr/bin/python3.8

#System packages
import subprocess
import os
import glob
import sys
import csv
import time
import cloup

#Utility packages
import numpy as np

#Mesh packages
import meshio as meshio

#Custom packages
import robot_arm_acoustic_post_processing

@cloup.command()
@cloup.option('--method', type=str, default="BEM", help="Computation method (BEM or SFT)")
@cloup.option('--measurementsMeshPath', type=str, default="./robotMesh.mesh", help="Path to the measurements mesh")
@cloup.option('--measurementsDataFolder', type=str, default="./", help="Path to the measurements data folder")
@cloup.option('--verificationMeshPath', type=str, default="../../../Verification/Verification/robotMesh.mesh", help="Path to the verification mesh")
@cloup.option('--verificationDataFolder', type=str, default="../../../Verification/Verification/", help="Path to the verification data folder")
@cloup.option('--frequencies', type=str, default="100,250,500,750,1000,2500,5000", help="Studied frequencies")
@cloup.option('--gradient', is_flag=True, help="Compute the gradient of the acoustic field")
def main(method, measurementsMeshPath, measurementsDataFolder, verificationMeshPath, verificationDataFolder, frequencies, gradient):

    #Generate command (bash)
    if(method == "SFT"):
        command = "FreeFem++ " + os.path.dirname(robot_arm_acoustic_post_processing.__file__) + "AcousticComputationSFT.edp"
    else:
        command = "ff-mpirun -np 4 "  + os.path.dirname(robot_arm_acoustic_post_processing.__file__) +  "/AcousticComputationBEM.edp -wg"

    #Get studied frequencies
    Frequencies = sorted([int(os.path.basename(file).split(".")[0].split("_")[-1]) for file in glob.glob("data_*.csv")])
    try:
        Frequencies = [int(item) for item in frequencies.split(",")]
    except (IndexError, ValueError):
        print("Invalid frequency, defaulting to : " + str(Frequencies) + " Hz")   

    #Infering element type
    #TODO on Freefem side (interpolation, extrapolation)
    mesh = meshio.read(measurementsMeshPath)
    Vertices, Faces = mesh.points, mesh.get_cells_type("triangle")

    measurementsNumber = 0
    with open("data_" + str(Frequencies[0]) + ".csv", newline='') as csvfile:
        measurementsNumber = sum(1 for _ in csv.reader(csvfile, delimiter=','))

    elementType = None
    if(np.abs(len(Faces) - measurementsNumber) > np.abs(len(Vertices) - measurementsNumber)):
        elementType = "P1"
    else:
        elementType = "P0"
    print("Invalid element type, infered element type is : " + elementType)

    for frequency in Frequencies:
        print("Frequency : " + str(frequency) + " Hz")

        ### Run computation

        bashCommand = command + " -frequency " + str(frequency) + " -realMeasurements 1 -measurementsMeshPath " + measurementsMeshPath + " -measurementsDataPath " + measurementsDataFolder + "data_" + str(frequency) + ".csv -verificationMeshPath " + verificationMeshPath + " -verificationDataPath " + verificationDataFolder + "data_" + str(frequency) + ".csv -verificationGradientDataPath " + verificationDataFolder + "gradient/data_" + str(frequency) + ".csv -DelementType=" + elementType + " -Dgradient=" + str(int(gradient)) + " -ns"
        print(bashCommand)

        t0 = time.time()
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output,_ = process.communicate()
        t1 = time.time()

        with open("computation_time_"+ elementType + ".csv","a") as f:
            data = [str(frequency),str(t1-t0)]
            writer = csv.writer(f)
            writer.writerow(data)

        #DEBUG
        print(output.decode())

        killProcess = "killall FreeFem++-mpi"
        process = subprocess.Popen(killProcess.split(), stdout=subprocess.PIPE)
        output,_ = process.communicate()

        #DEBUG
        print(output.decode())

if __name__ == "__main__":
    main()