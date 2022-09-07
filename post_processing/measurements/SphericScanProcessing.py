from tkinter import E
import measpy as ms

import csv
import glob
import os
import sys

import numpy as np

import pyvista as pv

### MESH GENERATION

if(not os.path.isfile("Mesh.mesh")):

    print("Creating mesh from measurements positions...")

    Points = []
    with open("Positions.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:  
            Points.append(np.array(row[:3],dtype=float))
    Points = np.array(Points)
    Points -= np.mean(Points,axis=0)

    print(np.linalg.norm(Points[0]))

    cloud = pv.PolyData(Points[:82])
    volume = cloud.delaunay_3d(alpha=2.)
    shell = volume.extract_geometry()
    pv.save_meshio("Mesh.mesh",shell)

    print("Done !")

### ACOUSTIC PRESSURE FIELD GENERATION

fmin = 150  #Anechoic room cutting frquency
fmax =  10000   #PU probe upper limit

octBand = 3
c = 341

try : 
    frequency = int(sys.argv[1])
except:
    print("Invalid frequency for acoustic measurements post processing !")
    sys.exit(1)

if(not os.path.isfile("Measurements_" + str(frequency) +".txt")):
    print("Processing measured data...")

    Files = glob.glob("sweep_measurement_1_*.wav")
    Files = sorted(Files, key=lambda file:int(os.path.basename(file).split(".")[0].split("_")[-1]))

    P = []
    Freqs = []

    for i,file in enumerate(Files):

        print("Data processing file : " + file)
        M = ms.Measurement.from_csvwav(file.split(".")[0])

        if(i==0):
            Freqs.append(M.out_sig_freqs[0])
            Freqs.append(M.out_sig_freqs[1])
            Freqs.append(M.fs)

        SP = M.data["In1"] #Pressure
        P.append(SP)

    with open("Measurements_" + str(frequency) + ".txt",'w') as file:

        for i,p in enumerate(P):
            if(sys.argv[2] == "fft"):
                FP = p.rfft()
                label = " (Forurier transform)"
            elif(sys.argv[2] == "tfe"):
                FP = p.tfe_farina(Freqs)
                label = " (transfer function)"

            Pvalue = FP.nth_oct_smooth_to_weight_complex(octBand,frequency,frequency).acomplex[0]
          
            file.write(str(np.real(Pvalue)) + ";" + str(np.imag(Pvalue)) + "\n")

    print("Done !")

