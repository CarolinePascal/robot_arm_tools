import measpy as ms
import ProbeSensitivity as ps

import glob
import os

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
cmap = plt.get_cmap("tab10")

if(not os.path.isfile("Robot.npy")):

    print("Processing data...")

    FilesWith = sorted(glob.glob("WithRobot/*.wav"), key=lambda file:int(os.path.basename(file).split(".")[0].split("_")[-1]))
    FilesWithout = sorted(glob.glob("WithoutRobot/*.wav"), key=lambda file:int(os.path.basename(file).split(".")[0].split("_")[-1]))

    PWith = []
    VWith = []
    PWithout = []
    VWithout = []

    Freqs = []

    for i,file in enumerate(FilesWith):

        print("Data processing file : " + file)

        if(i==0):
            M = ms.Measurement.from_csvwav(file.split(".")[0])
            Freqs.append(M.out_sig_freqs[0])
            Freqs.append(M.out_sig_freqs[1])
            Freqs.append(M.fs)

        SP,SV = ps.dataProcessing(file,"In1","In2")

        PWith.append(SP)
        VWith.append(SV)

    PWith = np.array(PWith)
    VWith = np.array(VWith)

    for i,file in enumerate(FilesWithout):

        print("Data processing file : " + file)

        SP,SV = ps.dataProcessing(file,"In1","In2")

        PWithout.append(SP)
        VWithout.append(SV)

    PWithout = np.array(PWithout)
    VWithout = np.array(VWithout)

    print("Saving data...")

    with open('Robot.npy','wb') as f:
        np.save(f,PWith)
        np.save(f,VWith)
        np.save(f,PWithout)
        np.save(f,VWithout)

PWith = []
VWith = []
PWithout = []
VWithout = []

fmin = 150  #Anechoic room cutting frquency
fmax =  10000   #PU probe upper limit

octBand = 12

print("Loading files...")

with open('Robot.npy','rb') as f:
    PWith = np.load(f,allow_pickle=True)
    VWith = np.load(f,allow_pickle=True)
    PWithout = np.load(f,allow_pickle=True)
    VWithout = np.load(f,allow_pickle=True)

"""
figAllP,axAllP = plt.subplots(2)
figAllP.canvas.manager.set_window_title("All pressures without robot")

figAllV,axAllV = plt.subplots(2)
figAllV.canvas.manager.set_window_title("All velocities without robot")

from scipy.fftpack import fft

for i,P in enumerate(PWithout):
    P.rfft().filterout([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax).plot(axAllP,label=str(i+1))

for i,V in enumerate(VWithout):
    V.rfft().filterout([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax).plot(axAllV,label=str(i+1))

axAllP[0].set_xlim([fmin,fmax])
axAllP[1].set_xlim([fmin,fmax])
axAllV[0].set_xlim([fmin,fmax])
axAllV[1].set_xlim([fmin,fmax])
axAllP[0].legend()
axAllV[0].legend()

figAllPR,axAllPR = plt.subplots(2)
figAllPR.canvas.manager.set_window_title("All pressures with robot")

figAllVR,axAllVR = plt.subplots(2)
figAllVR.canvas.manager.set_window_title("All velocities with robot")

for i,P in enumerate(PWith):
    P.rfft().filterout([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax).plot(axAllPR,label=str(i+1))

for i,V in enumerate(VWith):
    V.rfft().filterout([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax).plot(axAllVR,label=str(i+1))

axAllPR[0].set_xlim([fmin,fmax])
axAllPR[1].set_xlim([fmin,fmax])
axAllVR[0].set_xlim([fmin,fmax])
axAllVR[1].set_xlim([fmin,fmax])
axAllPR[0].legend()
axAllVR[0].legend()

plt.show()
"""

index = 2

figP,axP = plt.subplots(2)
figP.canvas.manager.set_window_title("Pressure")

figV,axV = plt.subplots(2)
figV.canvas.manager.set_window_title("Velocity")

figD,axD = plt.subplots(2)
figD.canvas.manager.set_window_title("Delta")

PWith[index].rfft(norm="forward").filterout([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax).plot(axP,label="With robot")
PWithout[index].rfft(norm="forward").filterout([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax).plot(axP,label="Without robot")
axP[0].set_xlim([fmin,fmax])
axP[1].set_xlim([fmin,fmax])
VWith[index].rfft().filterout([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax).plot(axV,label="With robot")
VWithout[index].rfft().filterout([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax).plot(axV,label="Without robot")
axV[0].set_xlim([fmin,fmax])
axV[1].set_xlim([fmin,fmax])

DeltaV = (VWith[index].rfft().filterout([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax)/VWithout[index].rfft().filterout([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax))
DeltaP = (PWith[index].rfft().filterout([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax)/PWithout[index].rfft().filterout([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax))
DeltaV.plot(axD,label="Velocity",color=cmap(0))
DeltaP.plot(axD,label="Pressure",color=cmap(1))

maxDeltaV = max(20*np.log10(np.abs(DeltaV.values))[(DeltaV.freqs > fmin) & (DeltaV.freqs < fmax)])
minDeltaV = min(20*np.log10(np.abs(DeltaV.values))[(DeltaV.freqs > fmin) & (DeltaV.freqs < fmax)])
axD[0].plot(DeltaV.freqs,np.ones(len(DeltaV.freqs))*maxDeltaV,linestyle="--",color=cmap(0))
axD[0].plot(DeltaV.freqs,np.ones(len(DeltaV.freqs))*minDeltaV,linestyle="--",color=cmap(0))

maxDeltaP = max(20*np.log10(np.abs(DeltaP.values))[(DeltaP.freqs > fmin) & (DeltaP.freqs < fmax)])
minDeltaP = min(20*np.log10(np.abs(DeltaP.values))[(DeltaP.freqs > fmin) & (DeltaP.freqs < fmax)])
axD[0].plot(DeltaP.freqs,np.ones(len(DeltaP.freqs))*maxDeltaP,linestyle="--",color=cmap(1))
axD[0].plot(DeltaP.freqs,np.ones(len(DeltaP.freqs))*minDeltaP,linestyle="--",color=cmap(1))

axD[0].set_xlim([fmin,fmax])
axD[0].set_ylim([min(minDeltaV*1.1,minDeltaP*1.1),max(maxDeltaV*1.1,maxDeltaP*1.1)])

maxDeltaV = max(np.unwrap(np.angle(DeltaV.values))[(DeltaV.freqs > fmin) & (DeltaV.freqs < fmax)])
minDeltaV = min(np.unwrap(np.angle(DeltaV.values))[(DeltaV.freqs > fmin) & (DeltaV.freqs < fmax)])
axD[1].plot(DeltaV.freqs,np.ones(len(DeltaV.freqs))*maxDeltaV,linestyle="--",color=cmap(0))
axD[1].plot(DeltaV.freqs,np.ones(len(DeltaV.freqs))*minDeltaV,linestyle="--",color=cmap(0))

maxDeltaP = max(np.unwrap(np.angle(DeltaP.values))[(DeltaP.freqs > fmin) & (DeltaP.freqs < fmax)])
minDeltaP = min(np.unwrap(np.angle(DeltaP.values))[(DeltaP.freqs > fmin) & (DeltaP.freqs < fmax)])
axD[1].plot(DeltaP.freqs,np.ones(len(DeltaP.freqs))*maxDeltaP,linestyle="--",color=cmap(1))
axD[1].plot(DeltaP.freqs,np.ones(len(DeltaP.freqs))*minDeltaP,linestyle="--",color=cmap(1))

axD[1].set_xlim([fmin,fmax])
axD[1].set_ylim([min(minDeltaV*1.1,minDeltaP*1.1),max(maxDeltaV*1.1,maxDeltaP*1.1)])

axP[0].legend()
axV[0].legend()
axD[0].legend()

plt.show()

"""
correlationP = sp.signal.correlate(PWithout[index].values,PWith[index].values,mode = "same",method = "fft")
lagsP = sp.signal.correlation_lags(len(PWithout[index].values),len(PWith[index].values),mode = "same")
lagP = lagsP[np.argmax(correlationP)]

correlationV = sp.signal.correlate(VWithout[index].values,VWith[index].values,mode = "same",method = "fft")
lagsV = sp.signal.correlation_lags(len(VWithout[index].values),len(VWith[index].values),mode = "same")
lagV = lagsV[np.argmax(correlationV)]

def shiftValues(list,lag):
    #Right shift
    if(lag >= 0):
        newList = np.concatenate((np.zeros(np.abs(lag)),list[:-np.abs(lag)]))
        return(newList)
    #Left shift
    else:
        newList = np.concatenate((list[np.abs(lag):],np.zeros(np.abs(lag))))
        return(newList)

shiftedP = PWith[index].similar(values=shiftValues(PWith[index].values,lagP))
shiftedV = VWith[index].similar(values=shiftValues(VWith[index].values,lagV))

figP,axP = plt.subplots(2)
figP.canvas.manager.set_window_title("Pressure")

figV,axV = plt.subplots(2)
figV.canvas.manager.set_window_title("Velocity")

figD,axD = plt.subplots(2)
figD.canvas.manager.set_window_title("Delta")

PWith[index].rfft().plot(axP,label="With robot")
PWithout[index].rfft().plot(axP,label="Without robot")
axP[0].set_xlim([fmin,fmax])
axP[1].set_xlim([fmin,fmax])
VWith[index].rfft().plot(axV,label="With robot")
VWithout[index].rfft().plot(axV,label="Without robot")
axV[0].set_xlim([fmin,fmax])
axV[1].set_xlim([fmin,fmax])

(VWith[index].rfft()/VWithout[index].rfft()).plot(axD,label="Velocity")
(PWith[index].rfft()/PWithout[index].rfft()).plot(axD,label="Pressure")
axD[0].set_xlim([fmin,fmax])
axD[1].set_xlim([fmin,fmax])

axP[0].legend()
axV[0].legend()
axD[0].legend()

plt.show()
"""