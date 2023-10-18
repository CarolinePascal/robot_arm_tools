#!/usr/bin/python3

#System packages
import sys
import csv

#Utility packages
import numpy as np

#Custom tools packages
from acousticTools import *
from plotTools import *

## Function plotting the error between computed values and analytical values for an output files folder on the z=0 plane for a given parameters configuration
#  @param postProcessingID ID of the post-processing function (c.f. plotTools.py)
#  @param analyticalFunctionID ID of the analytical function (c.f. acousticTools.py)
#  @param errorID ID of the error function (c.f. plotTools.py)
def plotSphericCut(postProcessingID,analyticalFunctionID,errorID):

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
        if(len(parameterValues) == 1):
            print("Only possible value for " + parameter +" is " + str(configurations[0,i]) +  " (" + parametersUnits[i] + ")")
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
        ax = [fig1.add_subplot(111, polar=True),fig2.add_subplot(111, polar=True)]
    elif(postProcessingID == "mod" or postProcessingID == "phase"):
        fig = plt.figure()
        ax = [fig.add_subplot(111, polar=True)]

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
        if(analyticalFunction is not None):
            print("Absolute error FreeFem analytical solution = " + str(np.round(errorFunction(numericValuesA - analyticalValues),3)) + " (" + tmpUnit + ")")
            print("Absolute error FreeFem numerical solution = " +  str(np.round(errorFunction(numericValuesN - analyticalValues),3)) + " (" + tmpUnit + ")")
            print("Relative error FreeFem analytical solution = " + str(np.round(100*errorFunction((numericValuesA - analyticalValues)/analyticalValues),3)) + " %")
            print("Relative error FreeFem numerical solution = " +  str(np.round(100*errorFunction((numericValuesN - analyticalValues)/analyticalValues),3)) + " %")
        else:
            print("Absolute error = " +  str(np.round(errorFunction(numericValuesN - numericValuesA),3)) + " (" + tmpUnit + ")")
            print("Relative error = " + str(np.round(100*errorFunction((numericValuesN - numericValuesA)/numericValuesA),3)) + " %")

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
                    ax.plot(Phi,function(analyticalValues),label="Analytical solution - " + functionName + " (" + unit + ")" + legend,color='r')
                    ax.plot(Phi,function(numericValuesA),label="FreeFem analytical solution - " + functionName + " (" + unit + ")" + legend,color='b')
                    ax.plot(Phi,function(numericValuesN),label="FreeFem numerical solution - " + functionName + " (" + unit + ")" + legend,color='g',alpha=0.25)
                else:
                    ax.plot(Phi,function(numericValuesN),color='g',alpha=0.25)
            else:
                if(addLegend):
                    ax.plot(Phi,function(numericValuesA),label="Measured data - " + functionName + " (" + unit + ")" + legend, color=cmap(i), linestyle="dashed")
                    ax.plot(Phi,function(numericValuesN),label="Computed solution - " + functionName + " (" + unit + ")" + legend, color=cmap(i))
                else:
                    ax.plot(Phi,function(numericValuesA), color=cmap(i), linestyle="dashed")
                    ax.plot(Phi,function(numericValuesN), color=cmap(i))

        legendFlag = True if(len(configurations) > 1 and i == 0) else False
        if(postProcessingID == "re/im"):
            plotSphericFunction(lambda x : x[0],"real part","Pa",ax[0],legendFlag)
            plotSphericFunction(lambda x : x[1],"imaginary part","Pa",ax[1],legendFlag)
        elif(postProcessingID == "id"):
            plotSphericFunction(np.abs,"modulus","Pa",ax[0],legendFlag)
            plotSphericFunction(np.angle,"phase","rad",ax[1],legendFlag)
        elif(postProcessingID == "mod"):
            plotSphericFunction(lambda x:x,"modulus","Pa",ax[0],legendFlag)
        elif(postProcessingID == "phase"):
            plotSphericFunction(lambda x:x,"phase","rad",ax[0],legendFlag)

    for subax in ax:
        Amp = list(subax.get_ylim())

        if(Amp[0] < 0):
            delta = np.abs(Amp[1] - Amp[0])
            Amp[1] = Amp[1] + 0.5*delta 
            Amp[0] = Amp[0] - 0.5*delta 

        subax.set_rmin(Amp[0])
        subax.set_rmax(Amp[1])
        
        subax.set_thetagrids(np.arange(0,360,45),['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$'])

        arrow = dict(arrowstyle='<-')
        subax.annotate("",xy=(0,Amp[0]),xytext=(0,Amp[1]),xycoords="data",arrowprops=arrow,va='center')
        subax.annotate('x',xy=(0,Amp[0]),xytext=(-0.01,Amp[1] - np.abs(Amp[1])*0.03),xycoords="data",va='top',ha='right')
        subax.annotate("",xy=(np.pi/2,Amp[0]),xytext=(np.pi/2,Amp[1]),xycoords="data",arrowprops=arrow,va='center')
        subax.annotate('y',xy=(np.pi/2,Amp[0]),xytext=(np.pi/2+0.01,Amp[1] - np.abs(Amp[1])*0.03),xycoords="data",va='top',ha='right')

        subax.legend()
        plt.grid(linestyle = '--')

    plt.show()

if __name__ == "__main__": 
    postProcessing = input("Post processing function ? (default : id) " + str(list((postProcessingFunctions.keys()))))
    if(postProcessing not in list((postProcessingFunctions.keys()))):
        postProcessing = "id"
    analytical = input("Analytical function ? (default : infinitesimalDipole) " + str(list((analyticalFunctions.keys()))))
    if(analytical not in list((analyticalFunctions.keys()))):
        analytical = "infinitesimalDipole"
    error = input("Error function ? (default : l2) " + str(list((errorFunctions.keys()))))
    if(error not in list((errorFunctions.keys()))):
        error = "l2"

    plotSphericCut(postProcessing,analytical,error)


