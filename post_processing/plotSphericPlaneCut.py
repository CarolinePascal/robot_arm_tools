import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

import sys

from AcousticDipoleTools import *
from plotTools import *

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

            #analyticalValues.append(analyticalFunction(f,demid,np.sqrt(x*x + y*y + z*z),np.arctan2(np.sqrt(x*x + y*y),z),np.arctan2(y,x)))
            analyticalValues.append(analyticalFunction(f,np.sqrt(x*x + y*y + z*z)))
            numericValuesA.append(np.complex(float(row[3]),float(row[4])))
            numericValuesN.append(np.complex(float(row[5]),float(row[6])))

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
    analyticalValues = np.append(analyticalValues,analyticalValues[0])
    numericValuesA = np.append(numericValuesA,numericValuesA[0])
    numericValuesN = np.append(numericValuesN,numericValuesN[0])

    label = "Acoustic pressure field computed for : \n"
    for j,name in enumerate(parametersList):
        label += name + " = " + str(configuration[j]) + " (" + parametersUnits[j] + ") "
    label = label[:-1]

    if(postProcessingID != "re/im"):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        ax.plot(Phi,analyticalValues,label="Analytical solution (" + postProcessingID+")",color='r')
        ax.plot(Phi,numericValuesA,label="FreeFem analytical solution (" + postProcessingID+")",color='b')
        ax.plot(Phi,numericValuesN,label="FreeFem numerical solution (" + postProcessingID+")",color='g')
        maxAmp = max(max(analyticalValues),max(numericValuesA),max(numericValuesN))
        minAmp = min(min(analyticalValues),min(numericValuesA),min(numericValuesN))

        ax.set_title(label)
        ax.annotate('x', xy=(np.pi/40,maxAmp), xycoords='data', annotation_clip=False, size = 12)
        ax.annotate('y', xy=(np.pi/2 - np.pi/40,maxAmp), xycoords='data', annotation_clip=False, size = 12)
        ax.set_ylim([minAmp*0.9,maxAmp*1.1])

    else:
        figR, axR = plt.subplots(subplot_kw={'projection': 'polar'})
        figI, axI = plt.subplots(subplot_kw={'projection': 'polar'})

        axR.plot(Phi,np.real(analyticalValues),label="Analytical solution (Re)",color='r')
        axR.plot(Phi,np.real(numericValuesA),label="FreeFem analytical solution (Re)",color='b')
        axR.plot(Phi,np.real(numericValuesN),label="FreeFem numerical solution (Re)",color='g')
        axI.plot(Phi,np.imag(analyticalValues),label="Analytical solution (Im)",color='r')
        axI.plot(Phi,np.imag(numericValuesA),label="FreeFem analytical solution (Im)",color='b')
        axI.plot(Phi,np.imag(numericValuesN),label="FreeFem numerical solution (Im)",color='g')
        maxAmpR = max(max(np.real(analyticalValues)),max(np.real(numericValuesA)),max(np.real(numericValuesN)))
        minAmpR = min(min(np.real(analyticalValues)),min(np.real(numericValuesA)),min(np.real(numericValuesN)))
        maxAmpI = max(max(np.imag(analyticalValues)),max(np.imag(numericValuesA)),max(np.imag(numericValuesN)))
        minAmpI = min(min(np.imag(analyticalValues)),min(np.imag(numericValuesA)),min(np.imag(numericValuesN)))

        axR.set_title(label + "\n[Real part]")
        axI.set_title(label + "\n[Imaginary part]")
        axR.annotate('x', xy=(np.pi/40,maxAmpR), xycoords='data', annotation_clip=False, size = 12)
        axR.annotate('y', xy=(np.pi/2 - np.pi/40,maxAmpR), xycoords='data', annotation_clip=False, size = 12)
        axI.annotate('x', xy=(np.pi/40,maxAmpI), xycoords='data', annotation_clip=False, size = 12)
        axI.annotate('y', xy=(np.pi/2 - np.pi/40,maxAmpI), xycoords='data', annotation_clip=False, size = 12)
        axR.set_ylim([minAmpR*0.9,maxAmpR*1.1])
        axI.set_ylim([minAmpI*0.9,maxAmpI*1.1])

    plt.legend()
    plt.show()

### MAIN ###

postProcessing = input("Post processing function ? " + str(list((postProcessingFunctions.keys()))))
analytical = input("Analytical function ? " + str(list((analyticalFunctions.keys()))))
error = input("Error function ? " + str(list((errorFunctions.keys()))))
plotSphericCut(postProcessing,analytical,error)

