import measpy as ms
import ProbeSensitivity as ps

import glob
import os

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
cmap = plt.get_cmap("tab10")

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
    M = ms.Measurement.from_csvwav(file.split(".")[0])

    if(i==0):
        Freqs.append(M.out_sig_freqs[0])
        Freqs.append(M.out_sig_freqs[1])
        Freqs.append(M.fs)

    SP,SV = M.data["In1"],M.data["In2"]

    PWith.append(SP)
    VWith.append(SV)

for i,file in enumerate(FilesWithout):

    print("Data processing file : " + file)
    M = ms.Measurement.from_csvwav(file.split(".")[0])

    SP,SV = M.data["In1"],M.data["In2"]

    PWithout.append(SP)
    VWithout.append(SV)

fmin = 150  #Anechoic room cutting frquency
fmax =  10000   #PU probe upper limit

octBand = 12
index = 1

"""
figAllP,axAllP = plt.subplots(2)
figAllP.canvas.manager.set_window_title("All pressures without robot")

for i,(P,V) in enumerate(zip(PWithout,VWithout)):
    P.tfe_welch(V).nth_oct_smooth_complex(octBand,fmin,fmax).filterout([fmin,fmax]).plot(axAllP,label=str(i+1))

axAllP[0].set_title("Pressure without robot - 1/" + str(octBand) + " octave smoothing")
axAllP[0].legend()

figAllPR,axAllPR = plt.subplots(2)
figAllPR.canvas.manager.set_window_title("All pressures with robot")

for i,(P,V) in enumerate(zip(PWith,VWith)):
    P.tfe_welch(V).nth_oct_smooth_complex(octBand,fmin,fmax).filterout([fmin,fmax]).plot(axAllPR,label=str(i+1))

axAllPR[0].set_title("Pressure with robot - 1/" + str(octBand) + " octave smoothing")
axAllPR[0].legend()

plt.show()
"""

figP,axP = plt.subplots(2)
figP.canvas.manager.set_window_title("Pressure")

figD,axD = plt.subplots(2)
figD.canvas.manager.set_window_title("Delta")

PWith[index].tfe_welch(VWith[index]).nth_oct_smooth_complex(octBand,fmin,fmax).filterout([fmin,fmax]).plot(axP,label="With robot")
PWithout[index].tfe_welch(VWithout[index]).nth_oct_smooth_complex(octBand,fmin,fmax).filterout([fmin,fmax]).plot(axP,label="Without robot")

DeltaP = (PWith[index].tfe_welch(VWith[index]).nth_oct_smooth_complex(octBand,fmin,fmax).filterout([fmin,fmax]) - PWithout[index].tfe_welch(VWithout[index]).nth_oct_smooth_complex(octBand,fmin,fmax).filterout([fmin,fmax]))/PWithout[index].tfe_welch(VWithout[index]).nth_oct_smooth_complex(octBand,fmin,fmax).filterout([fmin,fmax])
DeltaP.plot(axD,label="Pressure",color=cmap(1))

for ax in axP:
    ax.grid(which="major")
    ax.grid(linestyle = '--',which="minor")
for ax in axD:
    ax.grid(which="major")
    ax.grid(linestyle = '--',which="minor")

axP[0].set_title("Pressure/Voltage TFE - 1/" + str(octBand) + " octave smoothing")
axP[0].legend()
axD[0].set_title("Pressure/Voltage TFE relative delta - 1/" + str(octBand) + " octave smoothing")
axD[0].legend()

figP.tight_layout()
figD.tight_layout()

figP.savefig("./Pressure.png",dpi=300,bbox_inches='tight')
figD.savefig("./Delta.png",dpi=300,bbox_inches='tight')

"""
figC,axC = plt.subplots()
figC.canvas.manager.set_window_title("Coherence")

PWith[index].coh(VWith[index]).filterout([fmin,fmax]).plot(axC,label="Coherence",dby=False,plot_phase=False)

axC.set_title("Pressure/Voltage coherence")
axC.legend()
figC.tight_layout()
figC.savefig("./Coherence.png",dpi=300,bbox_inches='tight')
"""
