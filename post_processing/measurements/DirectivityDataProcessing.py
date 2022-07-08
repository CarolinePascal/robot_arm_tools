import measpy as ms
import ProbeSensitivity as ps

import csv
import glob
import os
import sys

from scipy.spatial.transform import Rotation as R

import numpy as np
import matplotlib.pyplot as plt
cmap = plt.get_cmap("tab20")

if(not os.path.isfile("Directivity.npy")):

    print("Processing data...")

    Angle = []

    with open("Positions.csv") as csvfile:
        reader = csv.reader(csvfile)
        for i,row in enumerate(reader):  
            if(i == 0):
                Z0 = R.from_euler('xyz',np.array(row[3:]).astype(float)).as_matrix()[:,-1]
                Angle.append(0)
            else:
                Z = R.from_euler('xyz',np.array(row[3:]).astype(float)).as_matrix()[:,-1]
                Angle.append(np.arccos(np.dot(Z,Z0)/(np.linalg.norm(Z)*np.linalg.norm(Z0))))

    Angle = np.array(Angle)

    Files = glob.glob("*.wav")
    Files = sorted(Files, key=lambda file:int(os.path.basename(file).split(".")[0].split("_")[-1]))

    P = []
    V = []
    Freqs = []

    for i,file in enumerate(Files):

        print("Data processing file : " + file)

        if(i==0):
            M = ms.Measurement.from_csvwav(file.split(".")[0])
            Freqs.append(M.out_sig_freqs[0])
            Freqs.append(M.out_sig_freqs[1])
            Freqs.append(M.fs)

        SP,SV = ps.dataProcessing(file,"In1","In2")

        P.append(SP)
        V.append(SV)

    P = np.array(P)
    V = np.array(V)

    print("Saving data...")

    with open('Directivity.npy','wb') as f:
        np.save(f,Angle)
        np.save(f,P)
        np.save(f,V)
        np.save(f,Freqs)

Angle = []
P = []
V = []

print("Loading files...")

with open('Directivity.npy','rb') as f:
    Angle = np.load(f,allow_pickle=True)
    P = np.load(f,allow_pickle=True)
    V = np.load(f,allow_pickle=True)
    Freqs = np.load(f,allow_pickle=True)

fmin = 150  #Anechoic room cutting frquency
fmax =  10000   #PU probe upper limit
fs = Freqs[2]

octBand = 6
c = 341

Frequency = np.array([1000])
WaveLength = c/Frequency

PhiP = np.empty((len(Angle),len(Frequency)))
PhiV = np.empty((len(Angle),len(Frequency)))
MP = np.empty((len(Angle),len(Frequency)))
MV = np.empty((len(Angle),len(Frequency)))

label = ""

for i,d in enumerate(Angle):

    if(sys.argv[1] == "fft"):
        FP = P[i].rfft()
        FV = V[i].rfft()
        label = " (Forurier transform)"
    elif(sys.argv[1] == "tfe"):
        FP = P[i].tfe_farina(Freqs)
        FV = V[i].tfe_farina(Freqs)
        label = " (transfer function)"

    if(i == 0):
        ax = FP.plot(color=cmap(1),label="Raw data")
        FP.filterout([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax).plot(ax=ax,color=cmap(0),label="Filtered data")
        for subAx in ax:
            subAx.set_xlim([fmin,fmax])
            subAx.legend()
        ax[0].set_title("Pressure" + label)
        plt.show()

        ax = FV.plot(color=cmap(3),label="Raw data")
        FV.filterout([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax).plot(ax=ax,color=cmap(2),label="Filtered data")
        for subAx in ax:
            subAx.set_xlim([fmin,fmax])
            subAx.legend()
        ax[0].set_title("Velocity" + label)
        plt.show()

    for j,f in enumerate(Frequency):
        Pvalue = FP.nth_oct_smooth_to_weight_complex(octBand,f,f).acomplex[0]
        Vvalue = FV.nth_oct_smooth_to_weight_complex(octBand,f,f).acomplex[0]
        PhiP[i,j] = np.angle(Pvalue)
        PhiV[i,j] = np.angle(Vvalue)
        MP[i,j] = np.abs(Pvalue)
        MV[i,j] = np.abs(Vvalue)

figP = plt.figure()
axP = figP.add_subplot(projection='polar')
axP.plot(Angle,20*np.log10(MP[:,0]/ms.signal.PREF),linestyle="--",color=cmap(0))
scatterP = axP.scatter(Angle,20*np.log10(MP[:,0]/ms.signal.PREF),marker='o',s=20,color=cmap(0))

axP.set_thetamin(Angle[0]*180/np.pi)
axP.set_thetamax(Angle[-1]*180/np.pi)
axP.set_title("Pressure" + label)
axP.set_xlabel("Pressure (dB)")
axP.xaxis.set_label_coords(0.75, 0.15)

figV = plt.figure()
axV = figV.add_subplot(projection='polar')
axV.plot(Angle,20*np.log10(MV[:,0]/ms.signal.VREF),linestyle="--",color=cmap(2))
scatterV = axV.scatter(Angle,20*np.log10(MV[:,0]/ms.signal.VREF),marker='o',s=20,color=cmap(2))

axV.set_thetamin(Angle[0]*180/np.pi)
axV.set_thetamax(Angle[-1]*180/np.pi)
axV.set_title("Velocity" + label)
axV.set_xlabel("Velocity (dB)")
axV.xaxis.set_label_coords(0.75, 0.15)

plt.show()