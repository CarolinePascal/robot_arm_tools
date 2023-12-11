#!/usr/bin/python3

#System packages
import sys
import csv

#Utility packages
import numpy as np

#Custom tools packages
from acousticTools import *
from plotTools import *

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

## Function plotting the error between computed values and analytical values for an output files folder on the z=0 plane for a given parameters configuration
#  @param postProcessingID ID of the post-processing function (c.f. plotTools.py)
#  @param analyticalFunctionID ID of the analytical function (c.f. acousticTools.py)
#  @param errorID ID of the error function (c.f. plotTools.py)
def plotSphericCut(postProcessingID,analyticalFunctionID,errorID,**kwargs):

    #Get post-processing and analytical functions
    postProcessingFunction = postProcessingFunctions[postProcessingID]
    analyticalFunction = analyticalFunctions[analyticalFunctionID]
    errorFunction = errorFunctions[errorID]

    P,parametersList,parametersUnits,fileList = getParametersConfigurations()

    #Get indices frequency (mandatory parameter !) and dipole distance (optional parameter, default is 0)
    frequencyIndex = np.where(parametersList=="frequency")[0][0]
    try:
        dipoleDistanceIndex = np.where(parametersList=="dipoleDistance")[0][0]
    except:
        dipoleDistanceIndex = None

    #Registering the desired parameters configurations
    configurations = P

    for i,parameter in enumerate(parametersList):
        parameterValues = np.unique(configurations[:,i])
        fixedParameterValues = []

        if(parameter in kwargs and kwargs[parameter] in parameterValues):
            fixedParameterValues = [kwargs[parameter]]
            
        else:
            if(len(parameterValues) == 1):
                print("Only possible value for " + parameter +" is " + str(configurations[0,i]) +  " (" + parametersUnits[i] + ")")
                continue
            else:
                fixedParameterValues = list(input("What value for " + parameter + " ? (default : *) " + str(np.sort(parameterValues)) + " (" + parametersUnits[i] + ")").split(" "))

                try:
                    fixedParameterValues = [float(item) for item in fixedParameterValues]
                except:
                    fixedParameterValues = parameterValues

        interestIndices = np.hstack([np.where(configurations[:,i] == value)[0] for value in fixedParameterValues])
        
        configurations = configurations[interestIndices]

    if(postProcessingID == "re/im" or postProcessingID == "id"):
        fig1 = plt.figure()
        fig2 = plt.figure()
        fig = [fig1,fig2]
        ax = [fig[0].add_subplot(111, polar=True),fig[1].add_subplot(111, polar=True)]
    elif(postProcessingID == "mod" or postProcessingID == "phase"):
        fig = [plt.figure()]
        ax = [fig[0].add_subplot(111, polar=True)]

    title = "Computed acoustic pressure field"
    noTitle = True

    for j,name in enumerate(parametersList):
        if(name == "iteration"):
            continue
        
        if(len(np.unique(configurations[:,j])) == 1):
            if(noTitle):
                noTitle = False
                title = "Acoustic pressure field computed for : \n"

            title += name + " = " + str(configurations[0,j]) + " " + parametersUnits[j] + " - "
        else:
            continue

    if(not noTitle):        
        title = title[:-3]
    
    for subax in ax:
        subax.set_title(title)

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
            print("Absolute error FreeFem numerical solution = " +  str(absoluteError) + " (" + tmpUnit + ")")
            #print("Relative error FreeFem analytical solution = " + str(np.round(100*errorFunction((numericValuesA - analyticalValues)/analyticalValues),6)) + " %")
            print("Relative error FreeFem numerical solution = " +  str(relativeError) + " %")
        else:
            absoluteError = np.round(errorFunction(numericValuesN - numericValuesA),6)
            relativeError = np.round(100*errorFunction(numericValuesN - numericValuesA)/errorFunction(numericValuesA),6)
            print("Absolute error = " +  str(absoluteError) + " (" + tmpUnit + ")")
            print("Relative error = " + str(relativeError) + " %")

        if(len(configurations) == 1):
            for subax in ax:
                subax.set_title(title + "\n Relative error = " + str(np.round(relativeError,3)) + " \%")

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
                fig = plt.figure()
                ax = fig.add_subplot(111, polar=True)

            if(analyticalFunction is not None):
                if(addLegend): 
                    ax.plot(Phi,function(analyticalValues), label="Analytical solution - " + functionName + " (" + unit + ")" + legend, color='r', linestyle="dashed")
                    #ax.plot(Phi,function(numericValuesA), label="FreeFem analytical solution - " + functionName + " (" + unit + ")" + legend, color='g')
                    ax.plot(Phi,function(numericValuesN), label="FreeFem numerical solution - " + functionName + " (" + unit + ")" + legend, color='b', alpha=1)
                    return(max(max(function(analyticalValues)),max(function(numericValuesN))),min(min(function(analyticalValues)),min(function(numericValuesN))))
                else:
                    ax.plot(Phi,function(numericValuesN),color='g',alpha=1)
                    return(max(function(numericValuesN)),min(function(numericValuesN)))
            else:
                if(addLegend):
                    ax.plot(Phi,function(numericValuesA),label="Measured data - " + functionName + " (" + unit + ")" + legend, color=cmap(i), linestyle="dashed")
                    ax.plot(Phi,function(numericValuesN),label="Computed solution - " + functionName + " (" + unit + ")" + legend, color=cmap(i))
                else:
                    ax.plot(Phi,function(numericValuesA), color=cmap(i), linestyle="dashed")
                    ax.plot(Phi,function(numericValuesN), color=cmap(i))
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

            maxPlot_tmp, minPlot_tmp = plotSphericFunction(lambda x : x[1],"imaginary part","Pa",ax[1],legendFlag)
            if(maxPlot_tmp > maxPlot[1]):
                maxPlot[1] = maxPlot_tmp
            if(minPlot_tmp < minPlot[1]):
                minPlot[1] = minPlot_tmp
        elif(postProcessingID == "id"):
            maxPlot_tmp, minPlot_tmp = plotSphericFunction(np.abs,"modulus","Pa",ax[0],legendFlag)
            if(maxPlot_tmp > maxPlot[0]):
                maxPlot[0] = maxPlot_tmp
            if(minPlot_tmp < minPlot[0]):
                minPlot[0] = minPlot_tmp

            maxPlot_tmp, minPlot_tmp = plotSphericFunction(np.angle,"phase","rad",ax[1],legendFlag)
            if(maxPlot_tmp > maxPlot[1]):
                maxPlot[1] = maxPlot_tmp
            if(minPlot_tmp < minPlot[1]):
                minPlot[1] = minPlot_tmp
        elif(postProcessingID == "mod"):
            maxPlot_tmp, minPlot_tmp = plotSphericFunction(lambda x:x,"modulus","Pa",ax[0],legendFlag)
            if(maxPlot_tmp > maxPlot[0]):
                maxPlot[0] = maxPlot_tmp
            if(minPlot_tmp < minPlot[0]):
                minPlot[0] = minPlot_tmp
        elif(postProcessingID == "phase"):
            maxPlot_tmp, minPlot_tmp = plotSphericFunction(lambda x:x,"phase","rad",ax[0],legendFlag)
            if(maxPlot_tmp > maxPlot[0]):
                maxPlot[0] = maxPlot_tmp
            if(minPlot_tmp < minPlot[0]):
                minPlot[0] = minPlot_tmp

    for i,subax in enumerate(ax):

        Amp = [minPlot[i],maxPlot[i]]   
        AmpDelta = maxPlot[i] - minPlot[i]

        subax.set_rmin(Amp[0]*0.9)
        subax.set_rmax(Amp[1]*1.1)
        
        subax.set_thetagrids(np.arange(0,360,45),['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$'])

        Amp = list(subax.get_ylim())
        arrow = dict(arrowstyle='<-')
        subax.annotate("",xy=(0,Amp[0]),xytext=(0,Amp[1]),xycoords="data",arrowprops=arrow,va='center')
        subax.annotate('x',xy=(0,Amp[0]),xytext=(-0.025,Amp[1] - 0.1*AmpDelta),xycoords="data",va='top',ha='right')
        subax.annotate("",xy=(np.pi/2,Amp[0]),xytext=(np.pi/2,Amp[1]),xycoords="data",arrowprops=arrow,va='center')
        subax.annotate('y',xy=(np.pi/2,Amp[0]),xytext=(np.pi/2+0.025,Amp[1] - 0.1*AmpDelta),xycoords="data",va='top',ha='right')

        subax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)
        subax.grid(linestyle = '--')

        subax.yaxis.set_major_formatter(ScalarFormatter())
        subax.yaxis.get_major_formatter().set_useOffset(False)
        subax.yaxis.set_major_locator(plt.MaxNLocator(4))

    plt.show()

if __name__ == "__main__": 

    postProcessing = input("Post processing function ? (default : id) " + str(list((postProcessingFunctions.keys()))))
    if(postProcessing not in list((postProcessingFunctions.keys()))):
        postProcessing = "id"

    error = input("Error function ? (default : l2) " + str(list((errorFunctions.keys()))))
    if(error not in list((errorFunctions.keys()))):
        error = "l2"

    analytical = "infinitesimalDipole"
    try:
        analytical = os.getcwd().split("/")[-1].split("_")[0]
    except:
        analytical = input("Analytical function ? (default : infinitesimalDipole) " + str(list((analyticalFunctions.keys()))))
    if(analytical not in list((analyticalFunctions.keys()))):
        analytical = None

    plotSphericCut(postProcessing,analytical,error)


