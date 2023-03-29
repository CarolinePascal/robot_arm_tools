import measpy as ms
import ProbeSensitivity as ps

import glob
import os

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
cmap = plt.get_cmap("tab10")

P = []
V = []
Freqs = []

Files = sorted(glob.glob("*.wav"), key=lambda file:int(os.path.basename(file).split(".")[0].split("_")[-2]))

for i,file in enumerate(Files):

        print("Data processing file : " + file)
        M = ms.Measurement.from_csvwav(file.split(".")[0])

        if(i==0):
            Freqs.append(M.out_sig_freqs[0])
            Freqs.append(M.out_sig_freqs[1])
            Freqs.append(M.fs)

        P.append(M.data["In1"])
        V.append(M.data["In2"])

fmin = 150  #Anechoic room cutting frquency
fmax =  10000   #PU probe upper limit

octBand = 12

figAllP,axAllP = plt.subplots(2)
figAllP.canvas.manager.set_window_title("All pressures")

for i,(p,v) in enumerate(zip(P,V)):
    p.tfe_welch(v,fs=Freqs[2],nperseg=2**12,noverlap=None).nth_oct_smooth_complex(octBand,fmin,fmax).filterout([fmin,fmax]).plot(axAllP,label=str(i+1))

for ax in axAllP:
    ax.grid(which="major")
    ax.grid(linestyle = '--',which="minor")

axAllP[0].legend()

figAllP.tight_layout()
figAllP.savefig("./Pressure.png",dpi=300,bbox_inches='tight')

