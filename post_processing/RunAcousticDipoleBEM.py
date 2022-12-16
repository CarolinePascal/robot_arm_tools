import gmsh as gmsh
import subprocess
import os
import numpy as np

###STUDY PARAMETERS

Frequencies = [100,250,500,750,1000,2500,5000,7500]    #100,250,500,750,1000,2500,5000,7500,10000
Radius = [0.1,0.2,0.4]   #0.1,0.2,0.3,0.4,0.45,0.495
Resolutions = np.round(np.linspace(0.003,0.012,20),4)    #0.005,0.0075,0.01,0.025,0.05,0.075,0.1 #np.round(np.linspace(0.003,0.012,20),4)

parametersCombinations = len(Frequencies)*len(Radius)*len(Resolutions)
counter = 1

###LAUNCH COMPUTATIONS

for radius in Radius:
    for resolution in Resolutions:

        if(not os.path.exists(os.getcwd() + "/meshes/sphere/S_" + str(radius) + "_" + str(resolution) + ".mesh")):
            gmsh.initialize()
            
            gmsh.option.set_number("General.Verbosity",0)
            #DEBUG
            #gmsh.option.set_number("General.Verbosity",1)

            gmsh.model.occ.addSphere(0,0,0,radius)
            gmsh.model.occ.synchronize()

            gmsh.option.setNumber("Mesh.MeshSizeMin",resolution)
            gmsh.option.setNumber("Mesh.MeshSizeMax",resolution)

            gmsh.model.mesh.generate(2)
            gmsh.write(os.getcwd() + "/meshes/sphere/S_" + str(radius) + "_" + str(resolution) + ".mesh")

            #DEBUG
            #gmsh.fltk.run()

            gmsh.finalize()

        for frequency in Frequencies:

            print("Combination : " + str(counter) + "/" + str(parametersCombinations))
            print("Frequency : " + str(frequency) + " Hz")
            print("Radius : " + str(radius) + " m")
            print("Resolution : " + str(resolution) + " m")
            counter += 1

            bashCommand = "ff-mpirun -np 4 AcousticDipoleBEM.edp -wg -frequency " + str(frequency) + " -radius " + str(radius) + " -resolution " + str(resolution)
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

            #DEBUG
            #print(output.decode())
