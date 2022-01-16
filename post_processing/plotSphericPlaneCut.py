import matplotlib.pyplot as plt
import numpy as np

import sys

from AcousticDipoleTools import *
from plotTools import *


def plotSphericCut(postProcessingID,analyticalFunctionID):

    #Get post-processing and analytical functions
    postProcessingFunction = postProcessingFunctions[postProcessingID]
    analyticalFunction = analyticalFunctions[analyticalFunctionID]

    M,parametersList,parametersUnits,fileList = getParametersConfigurations()

    #Registering the desired parameters configurations
    configuration = []
    for i,parameter in enumerate(parametersList):
        if(len(np.unique(M[:,i])) == 1):
            configuration.append(M[0,i])
            print("Only possible value for " + parameter +" is " + str(M[0,i]) +  " (" + parametersUnits[i] + ")")
        else:
            value = input("What value for " + parameter + " ? " + str(np.unique(M[:,i])) + " (" + parametersUnits[i] + ")")
            configuration.append(float(value))

    #Find the corresponding file name
    file = fileList[np.where((M==configuration).all(axis=1))[0][0]]

    numericValuesA = []
    numericValuesB = []
    analyticalValues = []

    R = []
    Theta = []
    Phi = []

    frequencyIndex = np.where(parametersList=="frequency")[0][0]
    dipoleDistanceIndex = np.where(parametersList=="dipoleDistance")[0][0]
    verticesIndex = np.where(parametersList=="vertices")[0][0]

    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')

        f = configuration[frequencyIndex]
        demid = configuration[dipoleDistanceIndex]/2        

        for row in reader:
            x = float(row[0])
            y = float(row[1])
            z = float(row[2])

            R.append(np.sqrt(x*x + y*y + z*z))
            Theta.append(np.arctan2(np.sqrt(x*x + y*y),z)+np.pi)
            Phi.append(np.arctan2(y,x)+np.pi)
    
            analyticalValues.append(postProcessingFunction(analyticalFunction(f,demid,np.sqrt(x*x + y*y + z*z),np.arctan2(np.sqrt(x*x + y*y),z),np.arctan2(y,x))))
            numericValuesA.append(postProcessingFunction(np.complex(float(row[3]),float(row[4]))))
            numericValuesB.append(postProcessingFunction(np.complex(float(row[5]),float(row[6]))))

    #NEW TEST
    PhiI = []
    testInterpolation = []
    for i in range(int(configuration[verticesIndex])*2):
        PhiI.append(0 + i*np.pi/configuration[verticesIndex])

    deltar = configuration[np.where(parametersList=="Rmax")[0][0]] - configuration[np.where(parametersList=="Rmin")[0][0]]

    for i,phi in enumerate(Phi):
        testInterpolation.append(postProcessingFunction(interpolationPhi(phi,PhiI,R[i],Theta[i],f,demid,deltar)))

    Phi = np.array(Phi)
    analyticalValues = np.array(analyticalValues)
    numericValuesA = np.array(numericValuesA)
    numericValuesB = np.array(numericValuesB)
    testInterpolation = np.array(testInterpolation)
    
    #Removing infinite values
    IndexInf = np.argwhere(np.isinf(np.abs(analyticalValues)))
    Phi = np.delete(Phi,IndexInf)
    analyticalValues = np.delete(analyticalValues,IndexInf)
    numericValuesA = np.delete(numericValuesA,IndexInf)
    numericValuesB = np.delete(numericValuesB,IndexInf)
    testInterpolation = np.delete(testInterpolation,IndexInf)

    #Creating plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    if(postProcessingID != "id"):
        ax.plot(Phi,analyticalValues,label="Analytical solution (" + postProcessingID+")",color='r')
        #ax.plot(Phi,numericValuesA,label="FreeFem Numerical solution A (" + postProcessingID+")",color='b')
        ax.plot(Phi,numericValuesB,label="FreeFem Numerical solution (" + postProcessingID+")",color='g')
        #ax.plot(Phi,testInterpolation,label="Finite differences (" + postProcessingID+")",color='k')
    else:
        ax.plot(Phi,np.real(analyticalValues),label="Analytical solution (Re)",color='r')
        ax.plot(Phi,np.real(numericValuesA),label="FreeFem Numerical solution A (Re)",color='b')
        ax.plot(Phi,np.real(numericValuesB),label="FreeFem Numerical solution B (Re)",color='g')
        ax.plot(Phi,np.imag(analyticalValues),label="Analytical solution (Im)",color='r',linestyle='dashed')
        ax.plot(Phi,np.imag(numericValuesA),label="FreeFem Numerical solution A (Im)",color='b',linestyle='dashed')
        ax.plot(Phi,np.imag(numericValuesB),label="FreeFem Numerical solution B (Im)",color='g',linestyle='dashed')

    print("ErrorA = " + str(np.sqrt(np.average(np.abs(numericValuesA - analyticalValues)**2))))
    print("ErrorB = " + str(np.sqrt(np.average(np.abs(numericValuesB - analyticalValues)**2))))

    print("ErrorTest = " + str(np.sqrt(np.average(np.abs(testInterpolation - analyticalValues)**2))))

    label = "Acoustic pressure field computed for : \n"
    for j,name in enumerate(parametersList):
        label += name + " = " + str(configuration[j]) + " (" + parametersUnits[j] + ") "
    label = label[:-1]

    ax.set_title(label)

    maxAmp = max(max(np.abs(analyticalValues)),max(np.abs(numericValuesA)),max(np.abs(numericValuesB)))*1.1

    ax.annotate('x', xy=(np.pi/40,maxAmp), xycoords='data', annotation_clip=False, size = 12)
    ax.annotate('y', xy=(np.pi/2 - np.pi/40,maxAmp), xycoords='data', annotation_clip=False, size = 12)

    plt.legend()
    plt.show()

### MAIN ###

postProcessing = input("Post processing function ? " + str(list((postProcessingFunctions.keys()))))
analytical = input("Analytical function ? " + str(list((analyticalFunctions.keys()))))
plotSphericCut(postProcessing,analytical)

