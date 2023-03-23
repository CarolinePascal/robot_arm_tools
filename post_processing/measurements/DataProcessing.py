import measpy as ms

import glob
import os

import csv

import numpy as np

import matplotlib.pyplot as plt
cmap = plt.get_cmap("tab10")

Data = []   
fmin = 150  #Anechoic room cutting frquency
fmax =  10000   #PU probe upper limit
f = 0

octBand = 3

Files = sorted(glob.glob("*.wav"), key=lambda file:int(os.path.basename(file).split(".")[0].split("_")[-1]))

for i,file in enumerate(Files):

        print("Data processing file : " + file)
        M = ms.Measurement.from_csvwav(file.split(".")[0])

        p = M.data["In1"]
        v = M.data["In2"]

        tfe = p.tfe_welch(v)
        f = tfe.freqs[4]
        print(f)

        Data.append(tfe.nth_oct_smooth_to_weight_complex(octBand,fmin=f,fmax=f).acomplex)

X = []
Y = []
Z = []

with open("Positions.csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        X.append(float(row[0]))
        Y.append(float(row[1]))
        Z.append(float(row[2]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X,Y,Z,c=20*np.log10(np.abs(Data)/ms.PREF.v),cmap="jet")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Acoustic pressure field amplitude (dB)\nValue for " + str(f) + "Hz")
plt.colorbar(sc)
plt.show()