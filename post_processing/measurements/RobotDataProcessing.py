import measpy as ms
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp

import glob
import os

import ProbeSensitivity as ps

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

def averageSpectral(spectralList):
    L = len(spectralList)
    modulus = None
    phase = None
    for i,item in enumerate(spectralList):
        fft = item.rfft().values
        if(i==0):
            modulus = np.abs(fft)/L
            phase = np.unwrap(np.angle(fft))/L
        else:
            modulus += np.abs(fft)/L
            phase += np.unwrap(np.angle(fft))/L
    return(spectralList[0].rfft().similar(values=modulus*np.exp(1j*phase)))

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

print("Loading files...")

with open('Robot.npy','rb') as f:
    PWith = np.load(f,allow_pickle=True)
    VWith = np.load(f,allow_pickle=True)
    PWithout = np.load(f,allow_pickle=True)
    VWithout = np.load(f,allow_pickle=True)

figAllP,axAllP = plt.subplots(2)
figAllP.canvas.manager.set_window_title("All pressures without robot")

figAllV,axAllV = plt.subplots(2)
figAllV.canvas.manager.set_window_title("All velocities without robot")

for i,P in enumerate(PWithout):
    P.rfft().plot(axAllP,label=str(i+1))

for i,V in enumerate(VWithout):
    V.rfft().plot(axAllV,label=str(i+1))

axAllP[0].set_xlim([10,10000])
axAllP[1].set_xlim([10,10000])
axAllV[0].set_xlim([10,10000])
axAllV[1].set_xlim([10,10000])
axAllP[0].legend()
axAllV[0].legend()

figAllPR,axAllPR = plt.subplots(2)
figAllPR.canvas.manager.set_window_title("All pressures with robot")

figAllVR,axAllVR = plt.subplots(2)
figAllVR.canvas.manager.set_window_title("All velocities with robot")

for i,P in enumerate(PWith):
    P.rfft().plot(axAllPR,label=str(i+1))

for i,V in enumerate(VWith):
    V.rfft().plot(axAllVR,label=str(i+1))

axAllPR[0].set_xlim([10,10000])
axAllPR[1].set_xlim([10,10000])
axAllVR[0].set_xlim([10,10000])
axAllVR[1].set_xlim([10,10000])
axAllPR[0].legend()
axAllVR[0].legend()

plt.show()

index = 2

figP,axP = plt.subplots(2)
figP.canvas.manager.set_window_title("Pressure")

figV,axV = plt.subplots(2)
figV.canvas.manager.set_window_title("Velocity")

figD,axD = plt.subplots(2)
figD.canvas.manager.set_window_title("Delta")

PWith[index].rfft().plot(axP,label="With robot")
PWithout[index].rfft().plot(axP,label="Without robot")
axP[0].set_xlim([10,10000])
axP[1].set_xlim([10,10000])
VWith[index].rfft().plot(axV,label="With robot")
VWithout[index].rfft().plot(axV,label="Without robot")
axV[0].set_xlim([10,10000])
axV[1].set_xlim([10,10000])

(VWith[index].rfft()/VWithout[index].rfft()).plot(axD,label="Velocity")
(PWith[index].rfft()/PWithout[index].rfft()).plot(axD,label="Pressure")
axD[0].set_xlim([10,10000])
axD[1].set_xlim([10,10000])

axP[0].legend()
axV[0].legend()
axD[0].legend()

plt.show()

figPAvg,axPAvg = plt.subplots(2)
figPAvg.canvas.manager.set_window_title("Pressure")

figVAvg,axVAvg = plt.subplots(2)
figVAvg.canvas.manager.set_window_title("Velocity")

figDAvg,axDAvg = plt.subplots(2)
figDAvg.canvas.manager.set_window_title("Delta")

PWithAvg = averageSpectral(PWith[1:])
VWithAvg = averageSpectral(VWith[1:])
PWithoutAvg = averageSpectral(PWithout[1:])
VWithoutAvg = averageSpectral(VWithout[1:])

PWithAvg.plot(axPAvg,label="With robot")
PWithoutAvg.plot(axPAvg,label="Without robot")
axPAvg[0].set_xlim([10,10000])
axPAvg[1].set_xlim([10,10000])
VWithAvg.plot(axVAvg,label="With robot")
VWithoutAvg.plot(axVAvg,label="Without robot")
axVAvg[0].set_xlim([10,10000])
axVAvg[1].set_xlim([10,10000])

(VWithAvg/VWithoutAvg).plot(axDAvg,label="Velocity")
(PWithAvg/PWithoutAvg).plot(axDAvg,label="Pressure")
axDAvg[0].set_xlim([10,10000])
axDAvg[1].set_xlim([10,10000])

axPAvg[0].legend()
axVAvg[0].legend()
axDAvg[0].legend()

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
axP[0].set_xlim([10,10000])
axP[1].set_xlim([10,10000])
VWith[index].rfft().plot(axV,label="With robot")
VWithout[index].rfft().plot(axV,label="Without robot")
axV[0].set_xlim([10,10000])
axV[1].set_xlim([10,10000])

(VWith[index].rfft()/VWithout[index].rfft()).plot(axD,label="Velocity")
(PWith[index].rfft()/PWithout[index].rfft()).plot(axD,label="Pressure")
axD[0].set_xlim([10,10000])
axD[1].set_xlim([10,10000])

axP[0].legend()
axV[0].legend()
axD[0].legend()

plt.show()
"""