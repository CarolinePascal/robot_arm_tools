import measpy as ms
import matplotlib.pyplot as plt
cmap = plt.get_cmap("tab10")

from scipy.signal import butter, lfilter

import csv
import glob
import os

import numpy as np

import ProbeSensitivity as ps

if(not os.path.isfile("Straight.npy")):

    print("Processing data...")

    Distance = []

    with open("Straight/Positions.csv") as csvfile:
        reader = csv.reader(csvfile)
        for i,row in enumerate(reader):  
            if(i == 0):
                X0 = np.array(row[:3]).astype(float)
                Distance.append(0)
            else:
                Distance.append(np.linalg.norm(np.array(row[:3]).astype(float) - X0))

    Distance = np.array(Distance)

    Files = glob.glob("Straight/*.wav")
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

Distance = []
P = []
V = []

print("Loading files...")

with open('Straight.npy','rb') as f:
    Distance = np.load(f,allow_pickle=True)
    P = np.load(f,allow_pickle=True)
    V = np.load(f,allow_pickle=True)
    Freqs = np.load(f,allow_pickle=True)

fmin,fmax = Freqs[0],Freqs[1]
fs = Freqs[2]

Distance += 0.32

c = 341

Frequency = np.array([50,100,200,500,1000,2000,5000])
WaveLength = c/Frequency
PhiP = np.empty((len(Distance),len(Frequency)))
PhiV = np.empty((len(Distance),len(Frequency)))

MP = np.empty((len(Distance),len(Frequency)))
MV = np.empty((len(Distance),len(Frequency)))

for i,d in enumerate(Distance):
    smoothFFTP = P[i].rfft()
    smoothFFTV = V[i].rfft()
    for j,f in enumerate(Frequency):
        index = np.where(np.round(smoothFFTP.freqs - f,1) == 0)[0][0]

        PhiP[i,j] = np.angle(smoothFFTP.values[index])
        PhiV[i,j] = np.angle(smoothFFTV.values[index])

        MP[i,j] = np.abs(smoothFFTP.values[index])
        MV[i,j] = np.abs(smoothFFTV.values[index])

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

"""
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
"""

fig,ax = plt.subplots()

for i,f in enumerate(Frequency):
    ax.plot(Distance,MP[:,i]/MV[:,i],label=str(f)+" Hz",color=cmap(i))
    ax.plot(Distance,(0.001*2*np.pi*f*Distance)/np.sqrt((Distance*2*np.pi*f/c)**2 + 1),color=cmap(i),linestyle="dashed")

ax.set_xlabel("Distance")
ax.set_ylabel("|P|/|V|")
ax.set_title("Modulus")
plt.legend()
plt.show()

PhiVteps = []
filteredDistance = []

for indexDistance,d in enumerate(Distance):
    smoothFFTP = P[indexDistance].rfft()
    smoothFFTV = V[indexDistance].rfft()

    #fig0,ax0 = plt.subplots(2)
    #smoothFFTP.plot(ax0)
    #smoothFFTV.plot(ax0)
    #plt.show()

    indexMin = np.where(np.round(smoothFFTP.freqs - fmin,1) == 0)[0][0]
    indexMax = np.where(np.round(smoothFFTP.freqs - fmax,1) == 0)[0][0]

    PhiDelta1 = np.unwrap(np.angle(smoothFFTP.values[indexMin:indexMax])) - np.unwrap(np.angle(smoothFFTV.values[indexMin:indexMax]))

    fs = 1/(smoothFFTP.freqs[1] - smoothFFTP.freqs[0])
    b,a = butter(1, fs/50, fs = fs, btype="low", analog=False)

    filteredPhiDelta1 = lfilter(b, a, PhiDelta1)
    dPhiDelta1 = (np.roll(filteredPhiDelta1,-1)[:-1] - filteredPhiDelta1[:-1])/(np.roll(smoothFFTP.freqs[indexMin:indexMax],-1)[:-1] - smoothFFTP.freqs[indexMin:indexMax][:-1])
    filtereddPhiDelta1 = lfilter(b, a, dPhiDelta1)

    indexZero = np.where(np.round(filtereddPhiDelta1)==0)[0]

    indexSteps = [indexZero[0]]
    N = 0.1
    for i,item in enumerate(indexZero[:-1]):
        if(indexZero[i+1] != item+1):
            delta = np.log10(smoothFFTP.freqs[indexMin:indexMax-1][item]) - np.log10(smoothFFTP.freqs[indexMin:indexMax-1][indexSteps[-1]])
            if(delta < N):
                indexSteps.pop()
                indexSteps.append(indexZero[i+1])
            else:
                indexSteps.append(item)
                indexSteps.append(indexZero[i+1])

    delta = np.log10(smoothFFTP.freqs[indexMin:indexMax-1][indexZero[-1]]) - np.log10(smoothFFTP.freqs[indexMin:indexMax-1][indexSteps[-1]])
    if(delta > N):
        indexSteps.append(len(indexZero)-1)
    else:
        indexSteps.pop()

    indexSteps = np.reshape(indexSteps,(-1,2))
    if(len(indexSteps)>2):
        continue
    else:
        filteredDistance.append(d)

    PhiVteps.append([np.average(PhiDelta1[step[0]:step[1]]) for step in indexSteps])

    if(indexDistance%1 == 0):
        fig1,ax1 = plt.subplots()
        ax1.plot(smoothFFTP.freqs[indexMin:indexMax],PhiDelta1,color=cmap(0))
        ax1.plot(smoothFFTP.freqs[indexMin:indexMax][:-1],dPhiDelta1,color=cmap(0),linestyle="dashed")
        for i,step in enumerate(indexSteps):
            ax1.plot(smoothFFTP.freqs[indexMin:indexMax][step],PhiDelta1[step],color=cmap(1+i))
            ax1.annotate(str(np.round(modulo(PhiVteps[-1][i]),2)) + " rad",(smoothFFTP.freqs[indexMin:indexMax][step[0]],PhiVteps[-1][i] + 50),color=cmap(1+i))
        ax1.set_xscale("log")

        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel(r"$\Delta \phi$ (rad)")
        ax1.set_title("Distance = " + str(np.round(d,2)) + " m")

        plt.legend()
        plt.show()

PhiVteps = modulo(np.array(PhiVteps))
print(PhiVteps)

fig,ax = plt.subplots()
ax.plot(filteredDistance,PhiVteps[:,0])
ax.plot(filteredDistance,PhiVteps[:,1])
plt.show()




