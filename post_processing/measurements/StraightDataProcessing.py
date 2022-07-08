import measpy as ms
import ProbeSensitivity as ps
from Tools import *

import csv
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
cmap = plt.get_cmap("tab10")

if(not os.path.isfile("Straight.npy")):

    print("Processing data...")

    Distance = []

    with open("Positions.csv") as csvfile:
        reader = csv.reader(csvfile)
        for i,row in enumerate(reader):  
            if(i == 0):
                X0 = np.array(row[:3]).astype(float)
                Distance.append(0)
            else:
                Distance.append(np.linalg.norm(np.array(row[:3]).astype(float) - X0))

    Distance = np.array(Distance)

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

    with open('Straight.npy','wb') as f:
        np.save(f,Distance)
        np.save(f,P)
        np.save(f,V)
        np.save(f,Freqs)

Distance = []
P = []
V = []

print("Loading files...")

with open('Straight.npy','rb') as f:
    Distance = np.load(f,allow_pickle=True)
    P = np.load(f,allow_pickle=True)
    V = np.load(f,allow_pickle=True)
    Freqs = np.load(f,allow_pickle=True)

fmin = 150  #Anechoic room cutting frquency
fmax =  10000   #PU probe upper limit
fs = Freqs[2]

Distance += 0.32
c = 341
octBand = 48

Frequency = np.array([100,500,1000,2000,5000])
WaveLength = c/Frequency

PhiP = np.empty((len(Distance),len(Frequency)))
PhiV = np.empty((len(Distance),len(Frequency)))
MP = np.empty((len(Distance),len(Frequency)))
MV = np.empty((len(Distance),len(Frequency)))

for i,d in enumerate(Distance):

    if(sys.argv[1] == "fft"):
        FTP = P[i].rfft()
        FTV = V[i].rfft()
        label = " (Forurier transform)"
    elif(sys.argv[1] == "tfe"):
        FTP = P[i].tfe_farina(Freqs)
        FTV = V[i].tfe_farina(Freqs)
        label = " (transfer function)"

    for j,f in enumerate(Frequency):
        Pvalue = FTP.nth_oct_smooth_to_weight_complex(octBand,f,f).acomplex[0]
        Vvalue = FTV.nth_oct_smooth_to_weight_complex(octBand,f,f).acomplex[0]
        PhiP[i,j] = np.angle(Pvalue)
        PhiV[i,j] = np.angle(Vvalue)
        MP[i,j] = np.abs(Pvalue)
        MV[i,j] = np.abs(Vvalue)

fig,ax = plt.subplots()

for i,f in enumerate(Frequency):
    PhiDelta = modulo(np.unwrap(PhiP[:,i]) - np.unwrap(PhiV[:,i]))
    ax.plot(Distance,PhiDelta,label=str(f)+" Hz",color=cmap(i))
    ax.plot(Distance,np.pi/2 - np.arctan((2*np.pi*f/340)*Distance),color=cmap(i),linestyle="dashed")

ax.set_xlabel("Distance (m)")
ax.set_ylabel(r"$\Delta \phi$ (rad)")
ax.set_title("Phase")
plt.legend()
plt.show()

fig,ax = plt.subplots()

for i,f in enumerate(Frequency):
    ax.plot(Distance,np.unwrap(PhiP[:,i]),label=str(f)+" Hz",color=cmap(i))
    ax.plot(Distance,-(2*np.pi*f/340)*Distance,color=cmap(i),linestyle="dashed")

ax.set_xlabel("Distance (m)")
ax.set_ylabel(r"$\phi_P$ (rad)")
plt.legend()
plt.show()

fig,ax = plt.subplots()

for i,f in enumerate(Frequency):
    ax.plot(Distance,np.unwrap(PhiV[:,i]),label=str(f)+" Hz",color=cmap(i))
    ax.plot(Distance,-(2*np.pi*f/340)*Distance -np.pi/2 + np.arctan((2*np.pi*f/340)*Distance),color=cmap(i),linestyle="dashed")

ax.set_xlabel("Distance (m)")
ax.set_ylabel(r"$\phi_V$ (rad)")
plt.legend()
plt.show()

fig,ax = plt.subplots()

LinearRegression = np.linalg.pinv(np.concatenate((np.array([np.log10(Distance)]),np.ones((1,len(Distance)))),axis=0).T)

for i,f in enumerate(Frequency):
    ax.plot(np.log10(Distance),np.log10(MP[:,i]),label=str(f)+" Hz",color=cmap(i))
    C = np.dot(LinearRegression,np.log10(MP[:,i]))
    ax.plot(np.log10(Distance),C[0]*np.log10(Distance)+C[1],color=cmap(i),linestyle="dashed")
    ax.annotate(str(np.round(C[0],2)) + "log(r) + " + str(np.round(C[1],2)),(np.average(np.log10(Distance)),0.01 + C[0]*np.average(np.log10(Distance))+C[1]),color=cmap(i))

ax.set_xlabel("log(Distance)")
ax.set_ylabel("log(|P|)")
ax.set_title("Pressure")
plt.legend()
plt.show()

fig,ax = plt.subplots()

for i,f in enumerate(Frequency):
    ax.plot(np.log10(Distance),np.log10(MV[:,i]),label=str(f)+" Hz",color=cmap(i))
    C = np.dot(LinearRegression,np.log10(MV[:,i]))
    ax.plot(np.log10(Distance),C[0]*np.log10(Distance)+C[1],color=cmap(i),linestyle="dashed")
    ax.annotate(str(np.round(C[0],2)) + "log(r) + " + str(np.round(C[1],2)),(np.average(np.log10(Distance)),0.01 + C[0]*np.average(np.log10(Distance))+C[1]),color=cmap(i))

ax.set_xlabel("log(Distance)")
ax.set_ylabel("log(|V|)")
ax.set_title("Velocity")
plt.legend()
plt.show()

fig,ax = plt.subplots()

for i,f in enumerate(Frequency):
    ax.plot(Distance,MP[:,i]/MV[:,i],label=str(f)+" Hz",color=cmap(i))
    ax.plot(Distance,(2*np.pi*f*Distance)/np.sqrt((Distance*2*np.pi*f/c)**2 + 1),color=cmap(i),linestyle="dashed")

ax.set_xlabel("Distance")
ax.set_ylabel("|P|/|V|")
ax.set_title("Modulus")
plt.legend()
plt.show()




