#!/usr/bin/python3.8

#System packages
import subprocess
import os
import glob

#Utility packages
import numpy as np

### Studied parameters
cutRadius = 0.25
Frequencies = sorted([int(os.path.basename(file).split(".")[0].split("_")[-1]) for file in glob.glob("./data*.csv")])

if(__name__ == "__main__"):
    
    for frequency in Frequencies:
        print("Frequency : " + str(frequency) + " Hz")

        ### Run computation

        bashCommand = "ff-mpirun -np 4 AcousticComputationBEM.edp -frequency " + str(frequency) + " -realMeasurements 1 -ns"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        #DEBUG
        print(output.decode())