import matplotlib.pyplot as plt
import numpy as np

import sys

from AcousticDipoleTools import *
from plotTools import *

def plotError(postProcessingID,analyticalFunctionID):

    #Get post-processing and analytical functions
    postProcessingFunction = postProcessingFunctions[postProcessingID]
    analyticalFunction = analyticalFunctions[analyticalFunctionID]

    P,parametersList,parametersUnits,fileList = getParametersConfigurations()

    #Studied parameter ?
    try:
        parameter = sys.argv[1]
    except:
        parameter = input("Abscissa parameter ? " + str(parametersList))

    #Put the studied parameter in the first position in each parameters configurations
    parameterIndex = np.where(parametersList == parameter)[0][0]
    P[:,[0,parameterIndex]] = P[:,[parameterIndex,0]]
    parametersList[[0,parameterIndex]] = parametersList[[parameterIndex,0]]
    parametersUnits[[0,parameterIndex]] = parametersUnits[[parameterIndex,0]]

    #Sort the output files according to the studied parameter values
    sortedIndices = np.argsort(P[:, 0])
    P = P[sortedIndices]
    fileList = fileList[sortedIndices]

    #Get all possible values for the studied parameter
    parameterValues = np.unique(P[:,0])
    lastIndex = []
    for value in parameterValues[:-1]:  #The last index is the last one of the list !
        lastIndex.append(np.where(P[:,0] == value)[0][-1] + 1)

    #Split the parameters configurations according to the studied parameter values
    splittedP = np.split(P,lastIndex) 

    #Get all possible parameters configurations
    interestConfigurations = np.unique(P[:,1:],axis=0)

    #Select only the parameters configurations which are computed for each studied parameter value
    for configurations in splittedP:
        configurations = configurations[:,1:]
        interestConfigurations = interestConfigurations[(interestConfigurations[:, None] == configurations).all(-1).any(1)]

    #Fixed parameter ? 
    flag = input("Any fixed parameter ? y/n")
    tmpParametersList = parametersList[1:]

    while(flag != "n"):
        tmpParameter = input("What parameter ? " + str(tmpParametersList))
        tmpParameterIndex = np.where(parametersList[1:] == tmpParameter)[0][0]
        tmpValue = list(input("what values ? " + str(np.unique(interestConfigurations[:,tmpParameterIndex])) + " (" + parametersUnits[1:][tmpParameterIndex] + ") ").split(' '))
        tmpValue = [float(item) for item in tmpValue]

        #Select only the parameters configurations containing the fixed parameters
        interestConfigurations = interestConfigurations[np.where(np.isin(interestConfigurations[:,tmpParameterIndex], tmpValue))[0]]
        tmpParametersList = np.delete(tmpParametersList,np.where(tmpParametersList == tmpParameter)[0][0])
        flag = input("Any fixed parameter ? y/n")

    #Create the interest configurations / plot values matrix
    plotListA = np.empty((len(interestConfigurations),len(parameterValues)))
    plotListB = np.empty((len(interestConfigurations),len(parameterValues)))
    plotListTest = np.empty((len(interestConfigurations),len(parameterValues)))

    #Get indices for frequency and dipole distance
    frequencyIndex = np.where(parametersList=="frequency")[0][0]
    dipoleDistanceIndex = np.where(parametersList=="dipoleDistance")[0][0]
    verticesIndex = np.where(parametersList=="vertices")[0][0]

    for i,file in enumerate(fileList):

        #Get interest parameters configuration index
        try:
            configurationIndex = np.where((interestConfigurations==P[i,1:]).all(axis=1))[0][0]
        except:
            continue

        #Get studied parameter value index
        parameterValueIndex = np.where(parameterValues == P[i,0])[0][0]

        #Create empty arrays
        numericValuesA = []
        numericValuesB = []
        analyticalValues = []

        R = []
        Theta = []
        Phi = []

        #Filling arrays from output file and analytical function
        with open(file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar='|')

            f = P[i,frequencyIndex]
            demid = P[i,dipoleDistanceIndex]/2

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
        for j in range(int(P[i,verticesIndex])*2):
            PhiI.append(0 + j*np.pi/P[i,verticesIndex])

        deltar = P[i,np.where(parametersList=="Rmax")[0][0]] - P[i,np.where(parametersList=="Rmin")[0][0]]

        for j,phi in enumerate(Phi):
            testInterpolation.append(postProcessingFunction(interpolationPhi(phi,PhiI,R[j],Theta[j],f,demid,deltar)))

        analyticalValues = np.array(analyticalValues)
        numericValuesA = np.array(numericValuesA)
        numericValuesB = np.array(numericValuesB)
        testInterpolation = np.array(testInterpolation)
        
        #Removing infinite values
        IndexInf = np.argwhere(np.isinf(np.abs(analyticalValues)))
        analyticalValues = np.delete(analyticalValues,IndexInf)
        numericValuesA = np.delete(numericValuesA,IndexInf)
        numericValuesB = np.delete(numericValuesB,IndexInf)
        testInterpolation = np.delete(testInterpolation,IndexInf)

        #Computing error over the z=0 planeA
        plotListA[configurationIndex][parameterValueIndex] = np.sqrt(np.average(np.abs(numericValuesA - analyticalValues)**2))
        plotListB[configurationIndex][parameterValueIndex] = np.sqrt(np.average(np.abs(numericValuesB - analyticalValues)**2))

        plotListTest[configurationIndex][parameterValueIndex] = np.sqrt(np.average(np.abs(testInterpolation - analyticalValues)**2))
        
    #Creating plots
    _, axA = plt.subplots()
    _, axB = plt.subplots()

    cmap = plt.cm.get_cmap('gist_rainbow', len(interestConfigurations))

    title = "Error (" + postProcessingID + ") computed with : \n" 

    for j,name in enumerate(parametersList[1:]): 
        if(len(np.unique(interestConfigurations[:,j])) == 1):
            title += name + " = " + str(interestConfigurations[0,j]) 
            if(parametersUnits[1:][j] != " "):
                title += " (" + parametersUnits[1:][j] + ")"
            title += " "
    title = title[:-1]

    for i,configuration in enumerate(interestConfigurations):
        label = ""
        for j,name in enumerate(parametersList[1:]):
            if(len(np.unique(interestConfigurations[:,j])) > 1):
                label += name + " = " + str(configuration[j]) 
                if(parametersUnits[1:][j] != " "):
                    label += " (" + parametersUnits[1:][j] + ")"
                label += "\n"
        label = label[:-1]
        
        axA.plot(parameterValues,np.log10(plotListA[i]),label=label)
        axB.plot(parameterValues,np.log10(plotListB[i]),label=label,color=cmap(i))

        #axA.plot(parameterValues,np.log10(plotListTest[i]),label=label,color=cmap(i),linestyle='dashed')

    axA.set_xlabel(parameter + " (" + parametersUnits[0] + ")")
    axA.set_ylabel("log(Average error)")
    axA.set_xscale('log')  
    axA.set_title(title)    

    axA.legend()

    axB.set_xlabel(parameter + " (" + parametersUnits[0] + ")")
    axB.set_ylabel("log(Average error)")
    axB.set_xscale('log')  
    axB.set_title(title)    

    axB.legend()

    plt.show()

### MAIN ###

postProcessing = input("Post processing function ? " + str(list((postProcessingFunctions.keys()))))
analytical = input("Analytical function ? " + str(list((analyticalFunctions.keys()))))
plotError(postProcessing,analytical)