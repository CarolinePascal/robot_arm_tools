import measpy as ms
import ProbeSensitivity as ps

import glob
import os

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
cmap = plt.get_cmap("tab10")
plt.rc('font', **{'size': 12, 'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

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

    PWith.append(M.data["In1"])
    VWith.append(M.data["Out1"])

for i,file in enumerate(FilesWithout):

    print("Data processing file : " + file)
    M = ms.Measurement.from_csvwav(file.split(".")[0])

    PWithout.append(M.data["In1"])
    VWithout.append(M.data["Out1"])

#fmin = 150  #Anechoic room cutting frquency
#fmax =  10000   #PU probe upper limit
fmin = 20
fmax = 20000

octBand = 30
index = 1

figAllP,axAllP = plt.subplots(2)

for i,P in enumerate(PWithout):
    P.tfe_farina([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax).filterout([fmin,fmax]).plot(axAllP,label=str(i+1))

for ax in axAllP:
    ax.grid(which="major")
    ax.grid(linestyle = '--',which="minor")
axAllP[0].set_title("Pressure without robot - 1/" + str(octBand) + " octave smoothing")
axAllP[0].legend()

figAllP.tight_layout()
figAllP.savefig("./AllPressuresWithout.pdf",dpi=300,bbox_inches='tight')

figAllPR,axAllPR = plt.subplots(2)

for i,P in enumerate(PWith):
    P.tfe_farina([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax).filterout([fmin,fmax]).plot(axAllPR,label=str(i+1))

for ax in axAllPR:
    ax.grid(which="major")
    ax.grid(linestyle = '--',which="minor")
axAllPR[0].set_title("Pressure with robot - 1/" + str(octBand) + " octave smoothing")
axAllPR[0].legend()

figAllPR.tight_layout()
figAllPR.savefig("./AllPressuresWith.pdf",dpi=300,bbox_inches='tight')

figP,axP = plt.subplots(2)
figD,axD = plt.subplots(2)

PWith[index].tfe_farina([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax).filterout([fmin,fmax]).plot(axP,label="With robot")
PWithout[index].tfe_farina([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax).filterout([fmin,fmax]).plot(axP,label="Without robot")

DeltaP = (PWith[index].tfe_farina([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax).filterout([fmin,fmax]) - PWithout[index].tfe_farina([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax).filterout([fmin,fmax]))/PWithout[index].tfe_farina([fmin,fmax]).nth_oct_smooth_complex(octBand,fmin,fmax).filterout([fmin,fmax])
DeltaP.plot(axD,label="Pressure",color=cmap(1))

for ax in axP:
    ax.grid(which="major")
    ax.grid(linestyle = '--',which="minor")
for ax in axD:
    ax.grid(which="major")
    ax.grid(linestyle = '--',which="minor")

axP[0].set_title("Pressure/Input signal TFE - 1/" + str(octBand) + " octave smoothing")
axP[0].legend()
axD[0].set_title("Pressure/Input signal TFE relative delta - 1/" + str(octBand) + " octave smoothing")
axD[0].legend()

figP.tight_layout()
figD.tight_layout()

figP.savefig("./Pressure.pdf",dpi=300,bbox_inches='tight')
figD.savefig("./Delta.pdf",dpi=300,bbox_inches='tight')

figC,axC = plt.subplots()
figC.canvas.manager.set_window_title("Coherence")

PWith[index].coh(VWith[index],nperseg=2**(np.ceil(np.log2(VWith[index].fs)))).filterout([fmin,fmax]).plot(axC,label="Coherence",dby=False,plot_phase=False)

axC.set_title("Pressure/Input signal coherence")
axC.legend()
figC.tight_layout()
figC.savefig("./Coherence.pdf",dpi=300,bbox_inches='tight')

