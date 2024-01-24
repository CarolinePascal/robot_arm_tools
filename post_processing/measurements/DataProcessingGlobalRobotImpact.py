#!/usr/bin/python3

#Acoustics package
import measpy as ms

#Utility packages
import glob
import os
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

#Plot packages
import matplotlib.pyplot as plt

#Data processing tools
from DataProcessingTools import plot_absolute_error, plot_relative_error, plot_relative_separated_error, compute_l2_errors, save_fig, set_title, cmap, markers, figsize, fmin, fmax, fminValidity, fmaxValidity, octBand

#Reference signal index
index = 0

if __name__ == "__main__":

    #Get processing method 
    processingMethod = "welch"
    try:
        processingMethod = os.sys.argv[1].lower()
        if(processingMethod not in ["welch","farina"]):
            raise ValueError("Invalid processing method")
    except IndexError:
        print("Invalid processing method, defaulting to " + processingMethod + " method")

    #Get transfer function input and output signals names
    inputSignal = "Out1" #Voltage
    outputSignal = "In1"   #Pressure
    try:
        inputSignal = sys.argv[2]
        outputSignal = sys.argv[3]
    except IndexError:
        print("Invalid input/output signals, defaulting to input : " + inputSignal + " and output : " + outputSignal)

    print("Processing input " + inputSignal + " and output " + outputSignal + " with " + processingMethod + " method")

    ControlPointsFolders = sorted(glob.glob("*/"), key=lambda folder:int(os.path.dirname(folder)))

    WWith = []
    WWithout = []

    for i,folder in enumerate(ControlPointsFolders):

        FilesWith = sorted(glob.glob(folder + "WithRobot/*.wav"), key=lambda file:int(os.path.basename(file).split(".")[0].split("_")[-1]))
        FilesWithout = sorted(glob.glob(folder + "WithoutRobot/*.wav"), key=lambda file:int(os.path.basename(file).split(".")[0].split("_")[-1]))

        file = FilesWith[index]

        print("Data processing file : " + file)
        M = ms.Measurement.from_csvwav(file.split(".")[0])

        #Check processing method compatibility
        if(processingMethod == "farina" and M.out_sig != "logsweep"):
            raise ValueError("Farina method cannot be used with non log sweep signals")

        P = M.data[outputSignal]
        V = M.data[inputSignal]
        
        if(processingMethod == "farina"):
            TFE = P.tfe_farina([fmin,fmax])
        else:
            TFE = P.tfe_welch(V) #Also possible for dB values : (P*V.rms).tfe_welch(V)

        #Remark : always the same !
        if(i == 0):
            Freqs = TFE.freqs[(TFE.freqs > fmin) & (TFE.freqs < fmax)]
            unit = TFE.unit

        w = TFE.nth_oct_smooth_to_weight_complex(octBand,fmin,fmax)
        WWith.append(w)

        file = FilesWithout[index]

        print("Data processing file : " + file)
        M = ms.Measurement.from_csvwav(file.split(".")[0])
        
        #Check processing method compatibility
        if(processingMethod == "farina" and M.out_sig != "logsweep"):
            raise ValueError("Farina method cannot be used with non log sweep signals")

        P = M.data[outputSignal]
        V = M.data[inputSignal]
        
        if(processingMethod == "farina"):
            TFE = P.tfe_farina([fmin,fmax])
        else:
            TFE = P.tfe_welch(V) #Also possible for dB values : (P*V.rms).tfe_welch(V)

        w = TFE.nth_oct_smooth_to_weight_complex(octBand,fmin,fmax)
        WWithout.append(w)
        
    figAllAbs,axAllAbs = plt.subplots(2,figsize=figsize)
    figAllRel,axAllRel = plt.subplots(1,figsize=figsize)
    figAllRelSep,axAllRelSep = plt.subplots(2,figsize=figsize)

    for i,(wWith, wWithout) in enumerate(zip(WWith,WWithout)):

        plot_absolute_error(wWith, wWithout, Freqs, ax=axAllAbs, validity_range=[fminValidity,fmaxValidity], marker=markers[i], color=cmap(i), label=str(i+1))

        plot_relative_error(wWith, wWithout, Freqs, ax=axAllRel, validity_range=[fminValidity,fmaxValidity], marker=markers[i], color=cmap(i), label=str(i+1))

        plot_relative_separated_error(wWith, wWithout, Freqs, ax=axAllRelSep, validity_range=[fminValidity,fmaxValidity], marker=markers[i], color=cmap(i), label=str(i+1))

        errorAbs, errorRel = compute_l2_errors(wWith, wWithout, frequencyRange=[fminValidity,fmaxValidity])
        print("Absolute L2 error " + str(i+1) + " : " + str(errorAbs) + " Pa/V")
        print("Relative L2 error " + str(i+1) + " : " + str(100*errorRel) + " %")

    #set_title(axAllAbs, "Pressure/Input signal TFE absolute error - 1/" + str(octBand) + " octave smoothing")
    #set_title(axAllRel, "Pressure/Input signal TFE relative error - 1/" + str(octBand) + " octave smoothing")
    #set_title(axAllRelSep, "Pressure/Input signal TFE modulus and phase relative error - 1/" + str(octBand) + " octave smoothing")

    save_fig(figAllAbs, "./" + processingMethod + "_AbsoluteError.pdf")
    save_fig(figAllRel, "./" + processingMethod + "_RelativeError.pdf")
    save_fig(figAllRelSep, "./" + processingMethod + "_RelativeErrorSeparate.pdf")
    plt.close("all")

    #plt.show()
