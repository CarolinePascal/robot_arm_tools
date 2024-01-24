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
from matplotlib.ticker import FuncFormatter

#Data processing tools
from DataProcessingTools import plot_weighting, log_formatter, plot_absolute_error, plot_relative_error, plot_relative_separated_error, compute_l2_errors, save_fig, set_title, cmap, markers, figsize, fmin, fmax, fminValidity, fmaxValidity, octBand

#Reference signal index
index = 0

if __name__ == "__main__":

    #Get processing method 
    processingMethod = "welch"
    try:
        processingMethod = sys.argv[1].lower()
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

    FilesWith = sorted(glob.glob("WithRobot/*.wav"), key=lambda file:int(os.path.basename(file).split(".")[0].split("_")[-1]))
    FilesWithout = sorted(glob.glob("WithoutRobot/*.wav"), key=lambda file:int(os.path.basename(file).split(".")[0].split("_")[-1]))
    
    WWith = []
    WWithout = []
    PWith = []
    VWith = []

    Freqs = []
    unit = None

    for i,file in enumerate(FilesWith):

        print("Data processing file : " + file)
        M = ms.Measurement.from_csvwav(file.split(".")[0])

        #Check processing method compatibility
        if(processingMethod == "farina" and M.out_sig != "logsweep"):
            raise ValueError("Farina method cannot be used with non log sweep signals")

        P = M.data[outputSignal]
        PWith.append(P)
        V = M.data[inputSignal]
        VWith.append(V)
        
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

    for i,file in enumerate(FilesWithout):

        print("Data processing file : " + file)
        M = ms.Measurement.from_csvwav(file.split(".")[0])

        P = M.data[outputSignal]
        V = M.data[inputSignal]
        
        if(processingMethod == "farina"):
            TFE = P.tfe_farina([fmin,fmax])
        else:
            TFE = P.tfe_welch(V) #Also possible for dB values : (P*V.rms).tfe_welch(V)

        w = TFE.nth_oct_smooth_to_weight_complex(octBand,fmin,fmax)
        WWithout.append(w)

    figAllWith,axAllWith = plt.subplots(2,figsize=figsize)
    figAllWithAbs,axAllWithAbs = plt.subplots(2,figsize=figsize)
    figAllWithRel,axAllWithRel = plt.subplots(1,figsize=figsize)
    figAllWithRelSep,axAllWithRelSep = plt.subplots(2,figsize=figsize)

    for i,w in enumerate(WWith):

        plot_weighting(w, Freqs, unit=unit, ax=axAllWith, marker=markers[i], color=cmap(i), label=str(i+1))

        if(i != index):
            plot_absolute_error(w, WWith[index], Freqs, ax=axAllWithAbs, validity_range=[fminValidity,fmaxValidity], marker=markers[i], color=cmap(i), label="Absolute error " + str(i+1) + "/" + str(index+1))
            
            plot_relative_error(w, WWith[index], Freqs, ax=axAllWithRel, validity_range=[fminValidity,fmaxValidity], marker=markers[i], color=cmap(i), label="Relative error " + str(i+1) + "/" + str(index+1))

            plot_relative_separated_error(w, WWith[index], Freqs, ax=axAllWithRelSep, validity_range=[fminValidity,fmaxValidity], marker=markers[i], color=cmap(i), label="Relative error " + str(i+1) + "/" + str(index+1))

            errorAllWithAbs, errorAllWithRel = compute_l2_errors(w, WWith[index], frequencyRange=[fminValidity,fmaxValidity])
            print("Absolute L2 repetability error with robot " + str(i+1) + "/" + str(index+1) + " : " + str(errorAllWithAbs) + " Pa/V")
            print("Relative L2 repetability error with robot " + str(i+1) + "/" + str(index+1) + " : " + str(100*errorAllWithRel) + " %")
        
    #set_title(axAllWith,"Pressure/Input signal TFE with robot\n1/" + str(octBand) + " octave smoothing")
    #set_title(axAllWithAbs,"Pressure/Input signal TFE repetability absolute error with robot\n1/" + str(octBand) + " octave smoothing")
    #set_title(axAllWithRel,"Pressure/Input signal TFE repetability relative error with robot\n1/" + str(octBand) + " octave smoothing")
    #set_title(axAllWithRelSep,"Pressure/Input Signal TFE repetability modulus and phase\nrelative errors with robot - 1/" + str(octBand) + " octave smoothing")

    save_fig(figAllWith,"./" + processingMethod + "_AllPressuresWith.pdf")
    save_fig(figAllWithAbs,"./" + processingMethod + "_AbsoluteErrorAllPressuresWith.pdf")
    save_fig(figAllWithRel,"./" + processingMethod + "_RelativeErrorAllPressuresWith.pdf")
    save_fig(figAllWithRelSep,"./" + processingMethod + "_RelativeErrorSeparateAllPressuresWith.pdf")
    plt.close("all")

    figAllWithout,axAllWithout = plt.subplots(2,figsize=figsize)
    figAllWithoutAbs,axAllWithoutAbs = plt.subplots(2,figsize=figsize)
    figAllWithoutRel,axAllWithoutRel = plt.subplots(1,figsize=figsize)
    figAllWithoutRelSep,axAllWithoutRelSep = plt.subplots(2,figsize=figsize)

    for i,w in enumerate(WWithout):

        plot_weighting(w, Freqs, unit=unit, ax=axAllWithout, marker=markers[i], color=cmap(i), label=str(i+1))

        if(i != index):
            plot_absolute_error(w, WWithout[index], Freqs, ax=axAllWithoutAbs, validity_range=[fminValidity,fmaxValidity], marker=markers[i], color=cmap(i), label="Absolute error " + str(i+1) + "/" + str(index+1))
            
            plot_relative_error(w, WWithout[index], Freqs, ax=axAllWithoutRel, validity_range=[fminValidity,fmaxValidity], marker=markers[i], color=cmap(i), label="Relative error " + str(i+1) + "/" + str(index+1))

            plot_relative_separated_error(w, WWithout[index], Freqs, ax=axAllWithoutRelSep, validity_range=[fminValidity,fmaxValidity], marker=markers[i], color=cmap(i), label="Relative error " + str(i+1) + "/" + str(index+1))

            errorAllWithAbs, errorAllWithRel = compute_l2_errors(w, WWithout[index], frequencyRange=[fminValidity,fmaxValidity])
            print("Absolute L2 repetability error without robot " + str(i+1) + "/" + str(index+1) + " : " + str(errorAllWithAbs) + " Pa/V")
            print("Relative L2 repetability error without robot " + str(i+1) + "/" + str(index+1) + " : " + str(100*errorAllWithRel) + " %")

    #set_title(axAllWithout,"Pressure/Input signal TFE without robot - 1/" + str(octBand) + " octave smoothing")
    #set_title(axAllWithoutAbs,"Pressure/Input signal TFE repetability absolute error without robot\n1/" + str(octBand) + " octave smoothing")
    #set_title(axAllWithoutRel,"Pressure/Input signal TFE repetability relative error without robot\n1/" + str(octBand) + " octave smoothing")
    #set_title(axAllWithoutRelSep,"Pressure/Input Signal TFE repetability modulus and phase\nrelative errors without robot - 1/" + str(octBand) + " octave smoothing")

    save_fig(figAllWithout,"./" + processingMethod + "_AllPressuresWithout.pdf")
    save_fig(figAllWithoutAbs,"./" + processingMethod + "_AbsoluteErrorAllPressuresWithout.pdf")
    save_fig(figAllWithoutRel,"./" + processingMethod + "_RelativeErrorAllPressuresWithout.pdf")
    save_fig(figAllWithoutRelSep,"./" + processingMethod + "_RelativeErrorSeparateAllPressuresWithout.pdf")
    plt.close("all")

    figBoth,axBoth = plt.subplots(2,figsize=figsize)
    figAbs,axAbs = plt.subplots(2,figsize=figsize)
    figRel,axRel = plt.subplots(1,figsize=figsize)
    figRelSep,axRelSep = plt.subplots(2,figsize=figsize)
    
    plot_weighting(WWithout[index], Freqs, unit=unit, ax=axBoth, marker=markers[0], color=cmap(0), label="Without robot")
    plot_weighting(WWith[index], Freqs, unit=unit, ax=axBoth, marker=markers[1], color=cmap(1), label="With robot")

    plot_absolute_error(WWith[index], WWithout[index], Freqs, ax=axAbs, validity_range=[fminValidity,fmaxValidity], marker=markers[0], color=cmap(0), label="Absolute error")

    plot_relative_error(WWith[index], WWithout[index], Freqs, ax=axRel, validity_range=[fminValidity,fmaxValidity], marker=markers[0], color=cmap(0), label="Relative error")

    plot_relative_separated_error(WWith[index], WWithout[index], Freqs, ax=axRelSep, validity_range=[fminValidity,fmaxValidity], marker=markers[0], color=cmap(0), label="Relative error")

    errorAbs, errorRel = compute_l2_errors(WWith[index], WWithout[index], frequencyRange=[fminValidity,fmaxValidity])
    print("Absolute L2 error : " + str(errorAbs) + " Pa/V")
    print("Relative L2 error : " + str(100*errorRel) + " %")

    #set_title(axBoth[0],"Pressure/Input signal TFE - 1/" + str(octBand) + " octave smoothing")
    #set_title(axAbs[0],"Pressure/Input signal TFE absolute error - 1/" + str(octBand) + " octave smoothing")
    #set_title(axRel,"Pressure/Input signal TFE relative error - 1/" + str(octBand) + " octave smoothing")  
    #set_title(axRelSep[0],"Pressure/Input Signal TFE modulus and phase relative errors\n1/" + str(octBand) + " octave smoothing")

    save_fig(figBoth,"./" + processingMethod + "_Pressure.pdf")
    save_fig(figAbs,"./" + processingMethod + "_AbsoluteError.pdf")
    save_fig(figRel,"./" + processingMethod + "_RelativeError.pdf")
    save_fig(figRelSep,"./" + processingMethod + "_RelativeErrorSeparate.pdf")
    plt.close("all")

    figC,axC = plt.subplots(1,figsize=figsize)

    PWith[index].coh(VWith[index], nperseg=2**(np.ceil(np.log2(VWith[index].fs)))).filterout([fmin,fmax]).plot(axC,label="Coherence",dby=False,plot_phase=False)

    axC.set_title("Pressure/Input Signal coherence", pad=30)
    axC.axvspan(fminValidity,fmaxValidity,color="gray",alpha=0.175,label="Valid frequency range")
    axC.grid(which="major")
    axC.grid(linestyle = '--',which="minor")
    axC.xaxis.set_minor_formatter(FuncFormatter(log_formatter))
    axC.legend(bbox_to_anchor=(0.5,1.0), loc='lower center', ncol=5, borderaxespad=0.25)
    figC.tight_layout()
    figC.savefig("./" + processingMethod + "_Coherence.pdf",dpi=300,bbox_inches='tight')
    plt.close("all")

    #plt.show()