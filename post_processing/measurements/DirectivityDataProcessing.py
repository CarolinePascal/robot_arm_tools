import measpy as ms
import matplotlib.pyplot as plt

import csv
import glob
import os

import numpy as np

import ProbeSensitivity as ps

from scipy.spatial.transform import Rotation as R

if(not os.path.isfile("Directivity.npy")):

    print("Processing data...")

    Angle = []

    with open("Directivity/Positions.csv") as csvfile:
        reader = csv.reader(csvfile)
        for i,row in enumerate(reader):  
            if(i == 0):
                Z0 = R.from_euler('xyz',np.array(row[3:]).astype(float)).as_matrix[:,-1]
                Angle.append(0)
            else:
                Z = R.from_euler('xyz',np.array(row[3:]).astype(float)).as_matrix[:,-1]
                Angle.append(np.arccos(np.dot(Z,Z0)/(np.linalg.norm(Z)*np.linalg.norm(Z0))))

    Angle = np.array(Angle)

    Files = glob.glob("Directivity/*.wav")
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

    with open('Straight.npy','wb') as f:
        np.save(f,Angle)
        np.save(f,P)
        np.save(f,V)
        np.save(f,Freqs)

def modulo(x):
    try:
        iter(x)
        return(np.array([modulo(item) for item in x]))
    except:
        if(x >= 0 and x < 2*np.pi):
            return(x)
        elif(x < 0):
            return(modulo(x + 2*np.pi))
        else:
            return(modulo(x - 2*np.pi))

Angle = []
P = []
V = []

print("Loading files...")

with open('Directivity.npy','rb') as f:
    Angle = np.load(f,allow_pickle=True)
    P = np.load(f,allow_pickle=True)
    V = np.load(f,allow_pickle=True)
    Freqs = np.load(f,allow_pickle=True)

fmin,fmax = Freqs[0],Freqs[1]
fs = Freqs[2]

c = 341

Frequency = np.array([50,100,200,500,1000,2000,5000])
WaveLength = c/Frequency

PhiP = np.empty((len(Angle),len(Frequency)))
PhiV = np.empty((len(Angle),len(Frequency)))
MP = np.empty((len(Angle),len(Frequency)))
MV = np.empty((len(Angle),len(Frequency)))

for i,d in enumerate(Angle):
    FFTP = P[i].rfft()
    FFTV = V[i].rfft()

    if(i == 0):
        FFTP.plot()
        plt.show()
        FFTP.filterout([Freqs[0],Freqs[1]]).nth_oct_smooth_complex(6,Freqs[0],Freqs[1]).plot()
        plt.show()
        FFTV.plot(dby=False)
        plt.show()
        FFTV.filterout([Freqs[0],Freqs[1]]).nth_oct_smooth_complex(6,Freqs[0],Freqs[1]).plot(dby=False)
        plt.show()

    for j,f in enumerate(Frequency):
        P = FFTP.nth_oct_smooth_to_weight(6,f,f).amp[0]
        V = FFTV.nth_oct_smooth_to_weight(6,f,f).amp[0]
        PhiP[i,j] = np.angle(P)
        PhiV[i,j] = np.angle(V)
        MP[i,j] = np.abs(P)
        MV[i,j] = np.abs(V)

figP = plt.figure()
axP = figP.add_subplot(projection='polar')
scatterP = axP.scatter(Angle,20*np.log10(MP))

axP.set_thetamin(Angle[0]*180/np.pi)
axP.set_thetamax(Angle[-1]*180/np.pi)
axP.set_title("Pressure (dB)")

figV = plt.figure()
axV = figV.add_subplot(projection='polar')
scatterV = axV.scatter(Angle,MV)

axV.set_thetamin(Angle[0]*180/np.pi)
axV.set_thetamax(Angle[-1]*180/np.pi)
axV.set_title("Velocity (m/s)")

plt.show()