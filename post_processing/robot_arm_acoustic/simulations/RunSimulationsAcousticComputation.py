#!/usr/bin/python3.8

#System packages
import subprocess
import os
import time
import csv
import cloup

#Utility packages
import numpy as np

#Mesh packages
from robot_arm_acoustic.MeshTools import generateSphericMesh, generateCircularMesh, addNoiseToMesh, saveMesh
import robot_arm_acoustic

### Studied function
function = "infinitesimalDipole"
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

#SigmasPosition = [0.001*verificationRadius,0.0025*verificationRadius,0.005*verificationRadius,0.0075*verificationRadius,0.01*verificationRadius,0.025*verificationRadius]
SigmasPosition = [0.0]
SigmasMeasure = [0.01,0.05,0.1,0.2,0.3]
#SigmasMeasure = [0.0]
Nsigma = 20

parametersCombinations = len(Frequencies)*len(Radius)*len(Resolutions)*len(DipoleDistances)*max(1,Nsigma*len(np.nonzero(SigmasPosition)[0]))*max(1,Nsigma*len(np.nonzero(SigmasMeasure)[0]))

@cloup.command()
@cloup.option('--method', type=str, default="BEM", help="Computation method (BEM or SFT)")
@cloup.option('--gradient', is_flag=True, help="Compute the gradient of the acoustic field")
@cloup.option('--input_mesh_path', type=str, default=None, help="Path to an eventual custom input mesh")
def main(method, gradient, input_mesh_path):

    #Generate command (bash)
    if(method == "SFT"):
        command = "FreeFem++ " + os.path.dirname(robot_arm_acoustic.__file__) + "AcousticComputationExpansion.edp -method SFT"
    elif(method == "ESM"):
        command = "FreeFem++ " + os.path.dirname(robot_arm_acoustic.__file__) + "AcousticComputationExpansion.edp -method ESM"
    else:
        command = "ff-mpirun -np 4 "  + os.path.dirname(robot_arm_acoustic.__file__) +  "/AcousticComputationBEM.edp -wg"

    #Generate verification mesh
    os.makedirs("./meshes/",exist_ok=True)
    if(not gradient):
        generateCircularMesh(verificationSize,verificationResolution,elementType="P1",save=True,saveFolder="./meshes/")
    else:
        generateSphericMesh(verificationSize,verificationResolution,elementType=elementType,save=True,saveFolder="./meshes/")

    counter = 1

    if(input_mesh_path is None):

        for radius in Radius:
            for resolution in Resolutions:

                ### Generate computation mesh
                size = 2*radius
                try:
                    generateSphericMesh(size,resolution,elementType=elementType,save=True,saveFolder="./meshes/")  #No need to generate a new mesh if it already exists
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
                                        generateSphericMesh(size,resolution,sigmaPosition,elementType=elementType,save=True,saveFolder="./meshes/") #Generate a new random mesh !
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

                                        bashCommand = command + " -realMeasurements 0 -frequency " + str(frequency) + " -size " + str(size) + " -resolution " + str(resolution) + " -dipoleDistance " + str(dipoleDistance) + " -sigmaPosition " + str(sigmaPosition) + " -sigmaMeasure " + str(sigmaMeasure) + tmpFileID + " -verificationSize " + str(verificationSize) + " -verificationResolution " + str(verificationResolution) + " -studiedFunction " + function + " -DelementType=" + elementType + " -Dgradient=" + str(int(gradient)) + " -ns"

                                        # Change GSL random seed
                                        gslPrefix = "GSL_RNG_SEED=" + str(np.random.randint(0,1000000)) + " && "
                                        bashCommand = gslPrefix + bashCommand

                                        print(bashCommand)
                                        t0 = time.time()
                                        output = subprocess.run(bashCommand, capture_output=True, text=True, shell=True)
                                        t1 = time.time()

                                        with open("computation_time_" + function + "_" + elementType + ".csv","a") as f:
                                            if(function == "dipole"):
                                                data = [str(frequency),str(resolution),str(size),str(dipoleDistance),str(sigmaPosition),str(sigmaMeasure),str(t1-t0)]
                                            else:
                                                data = [str(frequency),str(resolution),str(size),str(sigmaPosition),str(sigmaMeasure),str(t1-t0)]
                                            writer = csv.writer(f)
                                            writer.writerow(data)

                                        #DEBUG
                                        print(output.stdout)

                                        killProcess = "killall FreeFem++-mpi"
                                        output = subprocess.run(killProcess, capture_output=True, text=True, shell=True)

                                        #DEBUG
                                        print(output.stdout)
    
    else:

        import trimesh as trimesh
        mesh_name = os.path.basename(input_mesh_path).split(".")[0]
        mesh_folder = "./meshes/" + mesh_name + "/"
        os.makedirs(mesh_folder,exist_ok=True)

        #Translate mesh around the origin and save it in the correct format
        input_mesh = trimesh.load_mesh(input_mesh_path)
        input_mesh.vertices -= np.mean(input_mesh.vertices,axis=0)
        saveMesh(mesh_folder + "/" + mesh_name + ".mesh",input_mesh.vertices,input_mesh.faces)

        mesh_folder_uncertainty = mesh_folder[:-1] + "_uncertainty/"
        if(len(np.nonzero(SigmasPosition)[0]) != 0):
            os.makedirs(mesh_folder_uncertainty,exist_ok=True)

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
                                vertices, faces = addNoiseToMesh(input_mesh.vertices,input_mesh.faces,sigmaPosition)
                                saveMesh(mesh_folder_uncertainty + mesh_name + "_" + str(sigmaPosition) + ".mesh",vertices,faces)
                            except:
                                print("Mesh generation failed, skipping computation")
                                continue

                        for frequency in Frequencies:

                            for dipoleDistance in DipoleDistances:

                                ### Run computation

                                mesh_path = mesh_folder + "/" + mesh_name + ".mesh"
                                noisy_mesh_path = mesh_folder_uncertainty + mesh_name + "_" + str(sigmaPosition) + ".mesh"

                                bashCommand = command + " -realMeasurements 0 -frequency " + str(frequency) + " -measurementsMeshPath " + mesh_path + " -dipoleDistance " + str(dipoleDistance) + " -sigmaPosition " + str(sigmaPosition) + " -noisyMeasurementsMeshPath " + noisy_mesh_path + " -sigmaMeasure " + str(sigmaMeasure) + " -verificationSize " + str(verificationSize) + " -verificationResolution " + str(verificationResolution) + " -studiedFunction " + function + " -DelementType=" + elementType + " -Dgradient=" + str(int(gradient)) + " -ns"

                                # Change GSL random seed
                                gslPrefix = "GSL_RNG_SEED=" + str(np.random.randint(0,1000000)) + " && "
                                bashCommand = gslPrefix + bashCommand

                                print(bashCommand)
                                t0 = time.time()
                                output = subprocess.run(bashCommand, capture_output=True, text=True, shell=True)
                                t1 = time.time()

                                mesh_name = os.path.basename(input_mesh_path).split(".")[0]
                                with open("./computation_time_" + mesh_name + "_" + function + "_" + elementType + ".csv","a") as f:
                                    if(function == "dipole"):
                                        data = [str(frequency),str(dipoleDistance),str(sigmaPosition),str(sigmaMeasure),str(t1-t0)]
                                    else:
                                        data = [str(frequency),str(sigmaPosition),str(sigmaMeasure),str(t1-t0)]
                                    writer = csv.writer(f)
                                    writer.writerow(data)

                                #DEBUG
                                print(output.stdout)

                                killProcess = "killall FreeFem++-mpi"
                                output = subprocess.run(killProcess, capture_output=True, text=True, shell=True)

                                #DEBUG
                                print(output.stdout)


if(__name__ == "__main__"):
    main()
    