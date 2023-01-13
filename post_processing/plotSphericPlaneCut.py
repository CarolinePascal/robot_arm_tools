#!/usr/bin/python3

#System packages
import sys
import csv

#Utility packages
import matplotlib.pyplot as plt
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
    configuration = P

    for i,parameter in enumerate(parametersList):
        if(len(np.unique(configuration[:,i])) == 1):
            print("Only possible value for " + parameter +" is " + str(configuration[0,i]) +  " (" + parametersUnits[i] + ")")
        else:
            value = 0
            interestIndex = []
            value = float(input("What value for " + parameter + " ? " + str(np.unique(configuration[:,i])) + " (" + parametersUnits[i] + ")"))
            interestIndex = np.where(configuration[:,i] == value)[0]
            
            configuration = configuration[interestIndex]

    #Find the corresponding file name
    configuration = configuration[0]
    file = fileList[(P==configuration).all(1)][0]

    #Get configuration frequency and dipole distance parameters
    f = configuration[frequencyIndex]
    k = 2*np.pi*f/c
    demid = configuration[dipoleDistanceIndex]/2 if dipoleDistanceIndex is not None else 0

    #Create empty arrays
    numericValuesA = []
    numericValuesN = []
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

            analyticalValues.append(analyticalFunction(f,np.sqrt(x*x + y*y + z*z),np.arctan2(np.sqrt(x*x + y*y),z),np.arctan2(y,x),demid))
            numericValuesA.append(complex(float(row[3]),float(row[4])))
            numericValuesN.append(complex(float(row[5]),float(row[6])))

    R = np.array(R)
    Theta = np.array(Theta)
    Phi = np.array(Phi)

    #Remove outliers
    outliers = np.concatenate((np.where(np.isinf(np.abs(analyticalValues)))[0],np.where(np.abs(analyticalValues) < np.mean(np.abs(analyticalValues)) - 2*np.std(np.abs(analyticalValues)))[0]))

    R = np.delete(R,outliers)
    Theta = np.delete(Theta,outliers)
    Phi = np.delete(Phi,outliers)
    analyticalValues = postProcessingFunction(np.delete(analyticalValues,outliers))
    numericValuesN = postProcessingFunction(np.delete(numericValuesN,outliers))
    numericValuesA = postProcessingFunction(np.delete(numericValuesA,outliers))

    #Compute error over the z=0 plane
    print("Absolute error FreeFem analytical solution = " + str(errorFunction(numericValuesA - analyticalValues)))
    print("Absolute error FreeFem numerical solution = " +  str(errorFunction(numericValuesN - analyticalValues)))
    print("Relative error FreeFem analytical solution = " + str(errorFunction((numericValuesA - analyticalValues)/analyticalValues)))
    print("Relative error FreeFem numerical solution = " +  str(errorFunction((numericValuesN - analyticalValues)/analyticalValues)))

    #Create plot
    Phi = np.append(Phi,Phi[0])
    if(len(np.shape(analyticalValues)) == 1):
        analyticalValues = np.append(analyticalValues,analyticalValues[0])
        numericValuesA = np.append(numericValuesA,numericValuesA[0])
        numericValuesN = np.append(numericValuesN,numericValuesN[0])
    else:
        analyticalValues = np.vstack((analyticalValues.T,analyticalValues.T[0])).T
        numericValuesA = np.vstack((numericValuesA.T,numericValuesA.T[0])).T
        numericValuesN = np.vstack((numericValuesN.T,numericValuesN.T[0])).T

    label = "Acoustic pressure field computed for : \n"
    for j,name in enumerate(parametersList):
        label += name + " = " + str(configuration[j]) + " " + parametersUnits[j] + " - "
    label = label[:-2]

    def plotSphericFunction(function,functionName,unit):
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)

        ax.plot(Phi,function(analyticalValues),label="Analytical solution " + functionName + " (" + unit + ")",color='r')
        ax.plot(Phi,function(numericValuesA),label="FreeFem analytical solution " + functionName + " (" + unit + ")",color='b')
        ax.plot(Phi,function(numericValuesN),label="FreeFem numerical solution " + functionName + " (" + unit + ")",color='g')
        maxAmp = max(max(function(analyticalValues)),max(function(numericValuesA)),max(function(numericValuesN)))
        minAmp = min(min(function(analyticalValues)),min(function(numericValuesA)),min(function(numericValuesN)))
        delta = np.abs(maxAmp - minAmp)

        maxAmp = maxAmp + 0.1*delta 
        minAmp = minAmp - 0.1*delta 

        ax.set_title(label + "(" + functionName + ")")

        ax.set_rmin(minAmp)
        ax.set_rmax(maxAmp)
        ax.set_thetagrids(np.arange(0,360,45),['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$'])

        arrow = dict(arrowstyle='<-')
        ax.annotate("",xy=(0,minAmp),xytext=(0,maxAmp),xycoords="data",arrowprops=arrow,va='center')
        ax.annotate('x',xy=(0,minAmp),xytext=(-0.01,maxAmp - np.abs(maxAmp)*0.03),xycoords="data",va='top',ha='right')
        ax.annotate("",xy=(np.pi/2,minAmp),xytext=(np.pi/2,maxAmp),xycoords="data",arrowprops=arrow,va='center')
        ax.annotate('y',xy=(np.pi/2,minAmp),xytext=(np.pi/2+0.01,maxAmp - np.abs(maxAmp)*0.03),xycoords="data",va='top',ha='right')

        plt.grid(linestyle = '--')
        plt.legend()

    if(postProcessingID == "re/im"):
        plotSphericFunction(lambda x : x[0],"real part","Pa")
        plotSphericFunction(lambda x : x[1],"imaginary part","Pa")
    elif(postProcessingID == "id"):
        plotSphericFunction(np.abs,"modulus","Pa")
        plotSphericFunction(np.angle,"phase","rad")
    elif(postProcessingID == "mod"):
        plotSphericFunction(lambda x:x,"modulus","Pa")
    elif(postProcessingID == "phase"):
        plotSphericFunction(lambda x:x,"phase","rad")

    plt.show()

if __name__ == "__main__": 
    postProcessing = input("Post processing function ? (default : id) " + str(list((postProcessingFunctions.keys()))))
    if(postProcessing not in list((postProcessingFunctions.keys()))):
        postProcessing = "id"
    analytical = input("Analytical function ? (default : monopole) " + str(list((analyticalFunctions.keys()))))
    if(analytical not in list((analyticalFunctions.keys()))):
        analytical = "monopole"
    error = input("Error function ? (default : l2) " + str(list((errorFunctions.keys()))))
    if(error not in list((errorFunctions.keys()))):
        error = "l2"

    plotSphericCut(postProcessing,analytical,error)


