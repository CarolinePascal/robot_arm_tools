#!/usr/bin/python3

#System packages
import sys
import csv

#Utility packages
import numpy as np

#Acoustic packages
from measpy._tools import wrap

#Custom tools packages
from AcousticTools import *
from PlotTools import *

from matplotlib.ticker import ScalarFormatter,FormatStrFormatter
from matplotlib.ticker import MaxNLocator

figsize = (10,9)

## Function plotting the error between computed values and analytical values for an output files folder on the z=0 plane for a given parameters configuration
#  @param postProcessingID ID of the post-processing function (c.f. PlotTools.py)
#  @param analyticalFunctionID ID of the analytical function (c.f. AcousticTools.py)
#  @param errorID ID of the error function (c.f. PlotTools.py)
def plotSphericCut(postProcessingID,analyticalFunctionID,errorID,figureName="output.pdf",**kwargs):

    #Get post-processing and analytical functions
    postProcessingFunction = postProcessingFunctions[postProcessingID]
    analyticalFunction = analyticalFunctions[analyticalFunctionID]
    errorFunction = errorFunctions[errorID]

    P,parametersList,parametersUnits,fileList = getParametersConfigurations()

    #Get indices frequency (mandatory parameter !) and dipole distance (optional parameter, default is 0)
    frequencyIndex = np.where(parametersList=="frequency")[0][0]
    try:
        dipoleDistanceIndex = np.where(parametersList=="dipoleDistance")[0][0]
    except IndexError:
        dipoleDistanceIndex = None

    #Registering the desired parameters configurations
    configurations = P

    for i,parameter in enumerate(parametersList):
        parameterValues = np.unique(configurations[:,i])
        fixedParameterValues = []

        if(parameter in kwargs and (np.isin(kwargs[parameter],parameterValues).all() or kwargs[parameter] == "*")):
            if(kwargs[parameter] == "*"):
                fixedParameterValues = parameterValues
            else:
                fixedParameterValues = np.intersect1d(parameterValues,kwargs[parameter]).flatten()
            
        else:
            if(len(parameterValues) == 1):
                print("Only possible value for " + parameter +" is " + str(configurations[0,i]) +  " (" + parametersUnits[i] + ")")
                continue
            else:
                fixedParameterValues = list(input("What value for " + parameter + " ? (default : *) " + str(np.sort(parameterValues)) + " (" + parametersUnits[i] + ")").split(" "))

                try:
                    fixedParameterValues = [float(item) for item in fixedParameterValues]
                except ValueError:
                    fixedParameterValues = parameterValues

        interestIndices = np.hstack([np.where(configurations[:,i] == value)[0] for value in fixedParameterValues])
        
        configurations = configurations[interestIndices]

    if(postProcessingID == "re/im" or postProcessingID == "id"):
        fig,ax = plt.subplots(1,2,figsize=figsize,subplot_kw=dict(projection='polar'))
    elif(postProcessingID == "mod" or postProcessingID == "phase"):
        fig,ax = plt.subplots(1,figsize=figsize,subplot_kw=dict(projection='polar'))
        ax = [ax]

    title = "Computed acoustic pressure field"
    noTitle = True
    titleCounter = 0

    for j,name in enumerate(parametersList):
        if(name == "iteration"):
            continue
        
        if(len(np.unique(configurations[:,j])) == 1):
            if(noTitle):
                noTitle = False
                title = "Acoustic pressure field computed for : \n"

            if(titleCounter < 3):
                title += name + " = " + str(configurations[0,j]) + " " + parametersUnits[j] + " - "
                titleCounter += 1
            else:
                title = title[:-3]
                title += "\n" + name + " = " + str(configurations[0,j]) + " " + parametersUnits[j] + " - "
                titleCounter = 0

        else:
            continue

    if(not noTitle):        
        title = title[:-3]

    #fig.suptitle(title)

    for i,configuration in enumerate(configurations):
        #Find the corresponding file names
        file = fileList[(P==configuration).all(1)][0]

        #Get configuration frequency and dipole distance parameters
        f = configuration[frequencyIndex]
        k = 2*np.pi*f/c
        halfDipoleDistance = configuration[dipoleDistanceIndex]/2 if dipoleDistanceIndex is not None else 0

        #Create empty arrays
        numericValuesA = []
        numericValuesN = []

        if(analyticalFunction is not None):
            analyticalValues = []

        R = []
        Theta = []
        Phi = []

        #Fill arrays from output file and analytical function
        with open(file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar='|')

            for row in reader:
                x = float(row[0])
                y = float(row[1])
                z = float(row[2])
    
                R.append(np.sqrt(x*x + y*y + z*z))
                Theta.append(np.arctan2(np.sqrt(x*x + y*y),z))
                Phi.append(np.arctan2(y,x))

                if(analyticalFunction is not None):
                    analyticalValues.append(analyticalFunction(f,np.sqrt(x*x + y*y + z*z),np.arctan2(np.sqrt(x*x + y*y),z),np.arctan2(y,x),halfDipoleDistance))
                numericValuesA.append(complex(float(row[3]),float(row[4])))
                numericValuesN.append(complex(float(row[5]),float(row[6])))

        R = np.array(R)
        Theta = np.array(Theta)
        Phi = np.array(Phi)

        #Remove outliers
        outliers = []
        if(analyticalFunction is not None):
            outliers = np.concatenate((np.where(np.isinf(np.abs(analyticalValues)))[0],np.where(np.abs(analyticalValues) < np.mean(np.abs(analyticalValues)) - 2*np.std(np.abs(analyticalValues)))[0]))
            analyticalValues = postProcessingFunction(np.delete(analyticalValues,outliers))

        R = np.delete(R,outliers)
        Theta = np.delete(Theta,outliers)
        Phi = np.delete(Phi,outliers)
        numericValuesN = postProcessingFunction(np.delete(numericValuesN,outliers))
        numericValuesA = postProcessingFunction(np.delete(numericValuesA,outliers))

        #Compute error over the z=0 plane
        tmpUnit = "Pa" if postProcessingID != "phase" else "rad"
        absoluteError = 0
        relativeError = 0
        if(analyticalFunction is not None):
            absoluteError = np.round(errorFunction(numericValuesN - analyticalValues),6)
            relativeError = np.round(100*errorFunction(numericValuesN - analyticalValues)/errorFunction(analyticalValues),6)
            #print("Absolute error FreeFem analytical solution = " + str(np.round(errorFunction(numericValuesA - analyticalValues),6)) + " (" + tmpUnit + ")")
            print("Absolute error numerical solution = " +  str(absoluteError) + " (" + tmpUnit + ")")
            #print("Relative error FreeFem analytical solution = " + str(np.round(100*errorFunction((numericValuesA - analyticalValues)/analyticalValues),6)) + " %")
            print("Relative error numerical solution = " +  str(relativeError) + " %")
        else:
            absoluteError = np.round(errorFunction(numericValuesN - numericValuesA),6)
            relativeError = np.round(100*errorFunction(numericValuesN - numericValuesA)/errorFunction(numericValuesA),6)
            print("Absolute error = " +  str(absoluteError) + " (" + tmpUnit + ")")
            print("Relative error = " + str(relativeError) + " %")

        #if(len(configurations) == 1):
            #fig.suptitle(title + "\n Relative error = " + str(np.round(relativeError,3)) + " \%")

        #Create plot
        Phi = np.append(Phi,Phi[0])
        if(len(np.shape(numericValuesA)) == 1):
            if(analyticalFunction is not None):
                analyticalValues = np.append(analyticalValues,analyticalValues[0])
            numericValuesA = np.append(numericValuesA,numericValuesA[0])
            numericValuesN = np.append(numericValuesN,numericValuesN[0])
        else:
            if(analyticalFunction is not None):
                analyticalValues = np.vstack((analyticalValues.T,analyticalValues.T[0])).T
            numericValuesA = np.vstack((numericValuesA.T,numericValuesA.T[0])).T
            numericValuesN = np.vstack((numericValuesN.T,numericValuesN.T[0])).T

        legend = ""
        noLegend = True

        for j,name in enumerate(parametersList):
            if(name == "iteration"):
                continue
            if(len(np.unique(configurations[:,j])) != 1):
                if(noLegend):
                    legend = " with "
                    noLegend = False
                legend += name + " = " + str(configuration[j]) + " " + parametersUnits[j] + " - "
            else:
                continue
        if(not noLegend):
            legend = legend[:-3]

        def plotSphericFunction(function,functionName,unit,ax=None,addLegend=True):

            if(ax is None):
                fig,ax = plt.subplots(1,figsize=figsize,subplot_kw=dict(projection='polar'))

            if(analyticalFunction is not None):
                if(addLegend): 
                    ax.plot(Phi,function(analyticalValues), label="Analytical solution", color=cmap(0), linestyle="dashed",linewidth=2.5)
                    #ax.plot(Phi,function(numericValuesA), label="FreeFem analytical solution - " + functionName + " (" + unit + ")" + legend, color='g')
                    ax.plot(Phi,function(numericValuesN), label="Numerical solution", color=cmap(1), alpha=0.25,linewidth=2.5)
                    return(max(max(function(analyticalValues)),max(function(numericValuesN))),min(min(function(analyticalValues)),min(function(numericValuesN))))
                else:
                    ax.plot(Phi,function(numericValuesN),color=cmap(1), alpha=0.25,linewidth=2.5)
                    return(max(function(numericValuesN)),min(function(numericValuesN)))
            else:
                if(addLegend):
                    ax.plot(Phi,function(numericValuesA),label="Measured data - " + functionName + " (" + unit + ")" + legend, color=cmap(i), linestyle="dashed",linewidth=2.5)
                    ax.plot(Phi,function(numericValuesN),label="Computed solution - " + functionName + " (" + unit + ")" + legend, color=cmap(i),linewidth=2.5)
                else:
                    ax.plot(Phi,function(numericValuesA), color=cmap(i), linestyle="dashed",linewidth=2.5)
                    ax.plot(Phi,function(numericValuesN), color=cmap(i),linewidth=2.5)
                return(max(max(function(numericValuesA)),max(function(numericValuesN))),min(min(function(numericValuesA)),min(function(numericValuesN))))

        legendFlag = True if(len(configurations) >= 1 and i == 0) else False
        maxPlot = -np.ones(len(ax))*10**10
        minPlot = np.ones(len(ax))*10**10
        if(postProcessingID == "re/im"):
            maxPlot_tmp, minPlot_tmp = plotSphericFunction(lambda x : x[0],"real part","Pa",ax[0],legendFlag)
            if(maxPlot_tmp > maxPlot[0]):
                maxPlot[0] = maxPlot_tmp
            if(minPlot_tmp < minPlot[0]):
                minPlot[0] = minPlot_tmp
            ax[0].set_title("Real part", y=-0.325)

            maxPlot_tmp, minPlot_tmp = plotSphericFunction(lambda x : x[1],"imaginary part","Pa",ax[1],legendFlag)
            if(maxPlot_tmp > maxPlot[1]):
                maxPlot[1] = maxPlot_tmp
            if(minPlot_tmp < minPlot[1]):
                minPlot[1] = minPlot_tmp
            ax[1].set_title("Imaginary part", y=-0.325)

        elif(postProcessingID == "id"):
            maxPlot_tmp, minPlot_tmp = plotSphericFunction(np.abs,"modulus","Pa",ax[0],legendFlag)
            if(maxPlot_tmp > maxPlot[0]):
                maxPlot[0] = maxPlot_tmp
            if(minPlot_tmp < minPlot[0]):
                minPlot[0] = minPlot_tmp
            ax[0].set_title("Modulus (Pa)", y=-0.325)

            maxPlot_tmp, minPlot_tmp = plotSphericFunction(lambda x :np.angle(x),"phase","rad",ax[1],legendFlag)
            if(maxPlot_tmp > maxPlot[1]):
                maxPlot[1] = maxPlot_tmp
            if(minPlot_tmp < minPlot[1]):
                minPlot[1] = minPlot_tmp
            ax[1].set_title("Phase (rad)", y=-0.325)

        elif(postProcessingID == "mod"):
            maxPlot_tmp, minPlot_tmp = plotSphericFunction(lambda x:x,"modulus","Pa",ax[0],legendFlag)
            if(maxPlot_tmp > maxPlot[0]):
                maxPlot[0] = maxPlot_tmp
            if(minPlot_tmp < minPlot[0]):
                minPlot[0] = minPlot_tmp
            ax[0].set_title("Modulus (Pa)", y=-0.325)

        elif(postProcessingID == "phase"):
            maxPlot_tmp, minPlot_tmp = plotSphericFunction(lambda x:x,"phase","rad",ax[0],legendFlag)
            if(maxPlot_tmp > maxPlot[0]):
                maxPlot[0] = maxPlot_tmp
            if(minPlot_tmp < minPlot[0]):
                minPlot[0] = minPlot_tmp
            ax[0].set_title("Phase (rad)", y=-0.325)

    for i,subax in enumerate(ax):

        if((postProcessingID == "id" and i == 1) or postProcessingID == "phase"):
            subax.set_rmin(-np.pi)
            subax.set_rmax(np.pi)
        #else:
            #subax.set_rmin((maxPlot[i] + minPlot[i])/2 - 1.5*(maxPlot[i] - minPlot[i])/2)
            #subax.set_rmax((maxPlot[i] + minPlot[i])/2 + 1.5*(maxPlot[i] - minPlot[i])/2)
        
        subax.set_thetagrids(np.arange(0,360,45),['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$'])

        arrow = dict(arrowstyle='<-',color="gray")
        subax.annotate("",xy=(0.5,0.5),xytext=(1.0,0.5),xycoords="axes fraction",arrowprops=arrow)
        subax.annotate('x',xy=(0.5,0.5),xytext=(0.95,0.44),xycoords="axes fraction",color="gray")
        subax.annotate("",xy=(0.5,0.5),xytext=(0.5,1.0),xycoords="axes fraction",arrowprops=arrow)
        subax.annotate('y',xy=(0.5,0.5),xytext=(0.44,0.95),xycoords="axes fraction",color="gray")

        subax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
        subax.yaxis.set_major_locator(MaxNLocator(4))
        #subax.yaxis.set_minor_locator(MaxNLocator(2))
        subax.grid(linestyle= '-', which="major")
        #subax.grid(linestyle = '--', which="minor")
        subax.tick_params(axis='y', which='major', pad=10, labelsize=15)
        subax.tick_params(axis='x', which='major', pad=5, labelsize=20)

    if(len(ax) == 1):
        ax[0].legend(bbox_to_anchor=(0.5,1.0), loc='lower center', ncol=4, borderaxespad=2, reverse=False,  columnspacing=0.5)
    else:
        ax[0].legend(bbox_to_anchor=(-0.2, 1.0, 2.7, .1), loc='lower left', ncol=2, borderaxespad=2, reverse=False, mode="expand",  columnspacing=0.5)

    if(not figureName is None):
        fig.savefig(figureName, dpi = 300, bbox_inches = 'tight')
    else:
        plt.show()

if __name__ == "__main__": 

    #postProcessing = input("Post processing function ? (default : id) " + str(list((postProcessingFunctions.keys()))))
    postProcessing = "id"
    if(postProcessing not in list((postProcessingFunctions.keys()))):
        postProcessing = "id"

    #error = input("Error function ? (default : l2) " + str(list((errorFunctions.keys()))))
    error = "l2"
    if(error not in list((errorFunctions.keys()))):
        error = "l2"

    analytical = "infinitesimalDipole"
    try:
        analytical = os.getcwd().split("/")[-1].split("_")[0]
    except IndexError:
        analytical = input("Analytical function ? (default : infinitesimalDipole) " + str(list((analyticalFunctions.keys()))))
    if(analytical not in list((analyticalFunctions.keys()))):
        analytical = None

    kwargs = {}
    kwargs["resolution"] = 0.005
    kwargs["size"] = 0.5
    kwargs["dipoleDistance"] = 0.45
    kwargs["frequency"] = 5000
    kwargs["sigmaPosition"] = [0.0]

    plotSphericCut(postProcessing,analytical,error,"SphericPlaneCut1.pdf",**kwargs)

    kwargs["resolution"] = 0.25

    plotSphericCut(postProcessing,analytical,error,"SphericPlaneCut2.pdf",**kwargs)

    kwargs["resolution"] = 0.005
    kwargs["sigmaPosition"] = [0.0005]
    kwargs["iteration"] = "*"
    print(kwargs)

    plotSphericCut(postProcessing,analytical,error,"SphericPlaneCutSigma1.pdf",**kwargs)

    kwargs["sigmaPosition"] = [0.0025]
    kwargs["iteration"] = "*"

    plotSphericCut(postProcessing,analytical,error,"SphericPlaneCutSigma2.pdf",**kwargs)

    kwargs["sigmaPosition"] = [0.005]
    kwargs["iteration"] = "*"

    plotSphericCut(postProcessing,analytical,error,"SphericPlaneCutSigma3.pdf",**kwargs)

    kwargs["sigmaPosition"] = [0.0125]
    kwargs["iteration"] = "*"

    plotSphericCut(postProcessing,analytical,error,"SphericPlaneCutSigma4.pdf",**kwargs)


