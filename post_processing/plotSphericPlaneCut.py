import matplotlib.pyplot as plt
import numpy as np

import sys

from AcousticDipoleTools import *
from plotTools import *

np.set_printoptions(threshold=sys.maxsize)

def plotSphericCut(postProcessingID,analyticalFunctionID):

    #Get post-processing and analytical functions
    postProcessingFunction = postProcessingFunctions[postProcessingID]
    analyticalFunction = analyticalFunctions[analyticalFunctionID]

    M,parametersList,parametersUnits,fileList = getParametersConfigurations()

    #Get the scaling factors in case of normalisation
    normalisation = input("Normalise results ? y/n")
    scalingFactors = np.ones(np.shape(M)[0])
    if(normalisation == "y"):
        scalingFactors = M[:,np.where(parametersList=="frequency")[0][0]]/c

    #Registering the desired parameters configurations
    configuration = M

    for i,parameter in enumerate(parametersList):
        if(len(np.unique(configuration[:,i])) == 1):
            if(parametersUnits[i] == "m" and normalisation == "y"):
                print("Only possible value for " + parameter +" is " + str(np.round(scalingFactors[0]*configuration[0,i],2)) +  " (" + parametersUnits[i] + "/lambda)")
            else:
                print("Only possible value for " + parameter +" is " + str(configuration[0,i]) +  " (" + parametersUnits[i] + ")")
        else:
            value = 0
            interestIndex = []
            if(parametersUnits[i] == "m" and normalisation == "y"):
                value = float(input("What value for " + parameter + " ? " + str(np.unique(np.round(scalingFactors*configuration[:,i],2))) + " (" + parametersUnits[i] + "/lambda)"))
                interestIndex = np.where(np.round(scalingFactors*configuration[:,i],2) == value)[0]
            else:
                value = float(input("What value for " + parameter + " ? " + str(np.unique(configuration[:,i])) + " (" + parametersUnits[i] + ")"))
                interestIndex = np.where(configuration[:,i] == value)[0]
            
            configuration = configuration[interestIndex]
            scalingFactors = scalingFactors[interestIndex]

    #Find the corresponding file name
    configuration = configuration[0]
    file = fileList[(M==configuration).all(1)][0]

    numericValuesA = []
    numericValuesN = []
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

            if(np.abs(analyticalFunction(f,demid,np.sqrt(x*x + y*y + z*z),np.arctan2(np.sqrt(x*x + y*y),z),np.arctan2(y,x))) >= 1e-10 and not np.isinf(np.abs(analyticalFunction(f,demid,np.sqrt(x*x + y*y + z*z),np.arctan2(np.sqrt(x*x + y*y),z),np.arctan2(y,x))))):
    
                R.append(np.sqrt(x*x + y*y + z*z))
                Theta.append(np.arctan2(np.sqrt(x*x + y*y),z)+np.pi)
                Phi.append(np.arctan2(y,x)+np.pi)

                analyticalValues.append(postProcessingFunction(analyticalFunction(f,demid,np.sqrt(x*x + y*y + z*z),np.arctan2(np.sqrt(x*x + y*y),z),np.arctan2(y,x))))
                numericValuesN.append(postProcessingFunction(np.complex(float(row[3]),float(row[4]))))
                numericValuesA.append(postProcessingFunction(np.complex(float(row[5]),float(row[6]))))

                print(np.complex(float(row[3]),float(row[4])))
                print(np.complex(float(row[5]),float(row[6])))
                print(analyticalFunction(f,demid,np.sqrt(x*x + y*y + z*z),np.arctan2(np.sqrt(x*x + y*y),z),np.arctan2(y,x)))
                print(abs(analyticalFunction(f,demid,np.sqrt(x*x + y*y + z*z),np.arctan2(np.sqrt(x*x + y*y),z),np.arctan2(y,x))))

    analyticalValues = np.array(analyticalValues)
    numericValuesA = np.array(numericValuesA)
    numericValuesN = np.array(numericValuesN)

    #NEW TEST
    if(analyticalFunctionID == "dPn"):
        PhiI = []
        testInterpolation = []
        for i in range(int(configuration[verticesIndex])*2):
            PhiI.append(0 + i*np.pi/configuration[verticesIndex])

        deltar = configuration[np.where(parametersList=="deltaR")[0][0]]
        layers = configuration[np.where(parametersList=="layers")[0][0]]

        for i,phi in enumerate(Phi):
            testInterpolation.append(postProcessingFunction(interpolationPhi(phi,PhiI,R[i],Theta[i],f,demid,deltar,layers)))

        testInterpolation = np.array(testInterpolation)
    
    #Creating plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    if(postProcessingID != "id"):
        ax.plot(Phi,analyticalValues,label="Analytical solution (" + postProcessingID+")",color='r')
        ax.plot(Phi,numericValuesA,label="FreeFem analytical solution (" + postProcessingID+")",color='b')
        ax.plot(Phi,numericValuesN,label="FreeFem numerical solution (" + postProcessingID+")",color='g')
        if(analyticalFunctionID == "dPn"):
            ax.plot(Phi,testInterpolation,label="Finite differences (" + postProcessingID+")",color='k')
        maxAmp = max(max(analyticalValues),max(numericValuesA),max(numericValuesN))
        minAmp = min(min(analyticalValues),min(numericValuesA),min(numericValuesN))
    else:
        ax.plot(Phi,np.real(analyticalValues),label="Analytical solution (Re)",color='r')
        ax.plot(Phi,np.real(numericValuesA),label="FreeFem analytical solution (Re)",color='b')
        ax.plot(Phi,np.real(numericValuesN),label="FreeFem numerical solution (Re)",color='g')
        ax.plot(Phi,np.imag(analyticalValues),label="Analytical solution (Im)",color='r',linestyle='dashed')
        ax.plot(Phi,np.imag(numericValuesA),label="FreeFem analytical solution (Im)",color='b',linestyle='dashed')
        ax.plot(Phi,np.imag(numericValuesN),label="FreeFem numerical solution (Im)",color='g',linestyle='dashed')
        maxAmp = max(max(np.real(analyticalValues),np.imag(analyticalValues)),max(np.real(numericValuesA),np.imag(numericValuesA)),max(np.real(numericValuesN),np.imag(numericValuesN)))
        minAmp = min(min(np.real(analyticalValues),np.imag(analyticalValues)),min(np.real(numericValuesA),np.imag(numericValuesA)),min(np.real(numericValuesN),np.imag(numericValuesN)))

    print("Absolute error FreeFem analytical solution = " + str(np.sqrt(np.average(np.abs((numericValuesA - analyticalValues))**2))))
    print("Absolute error FreeFem numerical solution = " + str(np.sqrt(np.average(np.abs((numericValuesN - analyticalValues))**2))))
    print("Relative error FreeFem analytical solution = " + str(np.sqrt(np.average(np.abs((numericValuesA - analyticalValues)/analyticalValues)**2))))
    print("Relative error FreeFem numerical solution = " + str(np.sqrt(np.average(np.abs((numericValuesN - analyticalValues)/analyticalValues)**2))))

    if(analyticalFunctionID == "dPn"):
        print("Absolute error finite differences = " + str(np.sqrt(np.average(np.abs((testInterpolation - analyticalValues))**2))))
        print("Relative error finite differences = " + str(np.sqrt(np.average(np.abs((testInterpolation - analyticalValues)/analyticalValues)**2))))

    label = "Acoustic pressure field computed for : \n"
    for j,name in enumerate(parametersList):
        label += name + " = " + str(configuration[j]) + " (" + parametersUnits[j] + ") "
    label = label[:-1]

    ax.set_title(label)



    ax.annotate('x', xy=(np.pi/40,maxAmp), xycoords='data', annotation_clip=False, size = 12)
    ax.annotate('y', xy=(np.pi/2 - np.pi/40,maxAmp), xycoords='data', annotation_clip=False, size = 12)

    ax.set_ylim([minAmp,maxAmp])

    plt.legend()
    plt.show()

### MAIN ###

postProcessing = input("Post processing function ? " + str(list((postProcessingFunctions.keys()))))
analytical = input("Analytical function ? " + str(list((analyticalFunctions.keys()))))
plotSphericCut(postProcessing,analytical)

