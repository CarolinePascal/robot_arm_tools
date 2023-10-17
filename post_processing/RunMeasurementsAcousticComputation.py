#!/usr/bin/python3.8

#System packages
import subprocess
import os
import glob
import sys

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

### Studied parameters
Frequencies = sorted([int(os.path.basename(file).split(".")[0].split("_")[-1]) for file in glob.glob("./data*.csv")])

if(__name__ == "__main__"):
    
    for frequency in Frequencies:
        print("Frequency : " + str(frequency) + " Hz")

        ### Run computation

        bashCommand = command + " -frequency " + str(frequency) + " -realMeasurements 1 -ns"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        #DEBUG
        print(output.decode())

        killProcess = "killall FreeFem++-mpi"
        process = subprocess.Popen(killProcess.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        #DEBUG
        print(output.decode())