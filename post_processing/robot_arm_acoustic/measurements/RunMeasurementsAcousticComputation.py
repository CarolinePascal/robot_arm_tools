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
import robot_arm_acoustic

@cloup.command()
@cloup.option('--method', type=str, default="BEM", help="Computation method (BEM or SFT)")
@cloup.option('--measurements_mesh_path', type=str, default="./robotMesh.mesh", help="Path to the measurements mesh")
@cloup.option('--measurements_data_folder', type=str, default="./", help="Path to the measurements data folder")
@cloup.option('--verification_mesh_path', type=str, default="../../../Verification/Verification/robotMesh.mesh", help="Path to the verification mesh")
@cloup.option('--verification_data_folder', type=str, default="../../../Verification/Verification/", help="Path to the verification data folder")
@cloup.option('--frequencies', type=str, default="", help="Studied frequencies")
@cloup.option('--gradient', is_flag=True, help="Compute the gradient of the acoustic field")
def main(method, measurements_mesh_path, measurements_data_folder, verification_mesh_path, verification_data_folder, frequencies, gradient):

    #Generate command (bash)
    if(method == "SFT"):
        command = "FreeFem++ " + os.path.dirname(robot_arm_acoustic.__file__) + "AcousticComputationExpansion.edp -method SFT"
    elif(method == "ESM"):
        command = "FreeFem++ " + os.path.dirname(robot_arm_acoustic.__file__) + "AcousticComputationExpansion.edp -method ESM"
    else:
        command = "ff-mpirun -np 4 "  + os.path.dirname(robot_arm_acoustic.__file__) +  "/AcousticComputationBEM.edp -wg"

    #Get studied frequencies
    Frequencies = sorted([int(os.path.basename(file).split(".")[0].split("_")[-1]) for file in glob.glob(measurements_data_folder + "data_[0-9]*.csv")])
    try:
        Frequencies = [int(item) for item in frequencies.split(",")]
    except (IndexError, ValueError):
        print("Invalid frequency, defaulting to : " + str(Frequencies) + " Hz")   

    #Infering element type
    #TODO on Freefem side (interpolation, extrapolation)
    mesh = meshio.read(measurements_mesh_path)
    Vertices, Faces = mesh.points, mesh.get_cells_type("triangle")

    measurementsNumber = 0
    with open(measurements_data_folder + "data_" + str(Frequencies[0]) + ".csv", newline='') as csvfile:
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

        bashCommand = command + " -frequency " + str(frequency) + " -DrealMeasurements=1 -measurementsMeshPath " + measurements_mesh_path + " -measurementsDataPath " + measurements_data_folder + "data_" + str(frequency) + ".csv -verificationMeshPath " + verification_mesh_path + " -verificationDataPath " + verification_data_folder + "data_" + str(frequency) + ".csv -verificationGradientDataPath " + verification_data_folder + "gradient/data_" + str(frequency) + ".csv -DelementType=" + elementType + " -Dgradient=" + str(int(gradient)) + " -ns"
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