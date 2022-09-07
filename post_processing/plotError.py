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
    parameter = input("Abscissa parameter ? " + str(parametersList))
    normalisation = input("Normalise results ? y/n")

    #Put the studied parameter in the first position in each parameters configurations
    parameterIndex = np.where(parametersList == parameter)[0][0]
    P[:,[0,parameterIndex]] = P[:,[parameterIndex,0]]
    parametersList[[0,parameterIndex]] = parametersList[[parameterIndex,0]]
    parametersUnits[[0,parameterIndex]] = parametersUnits[[parameterIndex,0]]

    #Get the scaling factors in case of normalisation
    scalingFactors = np.ones(np.shape(P)[0])

    if(normalisation == "y"):
        #scalingFactor = 1/lambda = f/c
        scalingFactors = P[:,np.where(parametersList=="frequency")[0][0]]/c

    #Sort the output files according to the studied parameter values
    sortedIndices = np.argsort(np.round(P[:,0]*scalingFactors,2)) if np.any(scalingFactors != 1) else np.argsort(P[:,0])
    P = P[sortedIndices]
    fileList = fileList[sortedIndices]
    scalingFactors = scalingFactors[sortedIndices]

    #Get all possible values for the studied parameter
    parameterValues = np.unique(np.round(P[:,0]*scalingFactors,2)) if np.any(scalingFactors != 1) else np.unique(P[:,0])

    lastIndex = []
    for value in parameterValues[:-1]:  #The last index is the last one of the list !
        index = np.where(np.round(P[:,0]*scalingFactors,2) == value)[0][-1] + 1 if np.any(scalingFactors != 1) else np.where(P[:,0] == value)[0][-1] + 1
        lastIndex.append(index)

    #Split the parameters configurations according to the studied parameter values
    splittedP = np.split(P,lastIndex) 

    #Get all possible parameters configurations
    interestConfigurations = np.unique(P[:,1:],axis=0)

    #Select only the parameters configurations which are computed for each studied parameter value
    for configurations in splittedP:
        configurations = configurations[:,1:]
        interestConfigurations = interestConfigurations[(interestConfigurations[:, None] == configurations).all(-1).any(1)]

    #TODO Fix ?
    if(len(interestConfigurations) == 0):
        interestConfigurations = np.unique(P[:,1:],axis=0)

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
    plotListA = np.zeros((len(interestConfigurations),len(parameterValues)))
    plotListN = np.zeros((len(interestConfigurations),len(parameterValues)))
    plotListTest = np.zeros((len(interestConfigurations),len(parameterValues)))

    #Get indices for frequency and dipole distance
    frequencyIndex = np.where(parametersList=="frequency")[0][0]
    dipoleDistanceIndex = np.where(parametersList=="dipoleDistance")[0][0]
    verticesIndex = np.where(parametersList=="vertices")[0][0]

    #Relative or absolute error ?
    errorType = "absolute"
    relativeError = input("Relative error ? y/n")
    if(relativeError == "y"):
        errorType = "relative"

    plotCounter = 0

    for i,file in enumerate(fileList):
        
        #Get interest parameters configuration index
        try:
            configurationIndex = np.where((interestConfigurations==P[i,1:]).all(axis=1))[0][0]
            plotCounter += 1
            print("Plot : " + str(plotCounter) + " on " + str(len(parameterValues)*len(interestConfigurations)))
        except:
            continue

        #Get studied parameter value index
        parameterValueIndex = np.where(parameterValues == np.round(P[i,0]*scalingFactors[i],2))[0][0] if np.any(scalingFactors != 1) else np.where(parameterValues == P[i,0])[0][0]

        #Create empty arrays
        numericValuesA = []
        numericValuesN = []
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

                if(np.abs(analyticalFunction(f,demid,np.sqrt(x*x + y*y + z*z),np.arctan2(np.sqrt(x*x + y*y),z),np.arctan2(y,x))) >= 1e-10 and not np.isinf(np.abs(analyticalFunction(f,demid,np.sqrt(x*x + y*y + z*z),np.arctan2(np.sqrt(x*x + y*y),z),np.arctan2(y,x))))):
                    
                    R.append(np.sqrt(x*x + y*y + z*z))
                    Theta.append(np.arctan2(np.sqrt(x*x + y*y),z)+np.pi)
                    Phi.append(np.arctan2(y,x)+np.pi)

                    analyticalValues.append(postProcessingFunction(analyticalFunction(f,demid,np.sqrt(x*x + y*y + z*z),np.arctan2(np.sqrt(x*x + y*y),z),np.arctan2(y,x))))
                    numericValuesN.append(postProcessingFunction(np.complex(float(row[3]),float(row[4]))))
                    numericValuesA.append(postProcessingFunction(np.complex(float(row[5]),float(row[6]))))

        analyticalValues = np.array(analyticalValues)
        numericValuesA = np.array(numericValuesA)
        numericValuesN = np.array(numericValuesN)

        #NEW TEST
        if(analyticalFunctionID == "dPn"):
            PhiI = []
            testInterpolation = []
            for j in range(int(P[i,verticesIndex])*2):
                PhiI.append(0 + j*np.pi/P[i,verticesIndex])

            deltar = P[i,np.where(parametersList=="deltaR")[0][0]]
            layers = P[i,np.where(parametersList=="layers")[0][0]]

            for j,phi in enumerate(Phi):
                testInterpolation.append(postProcessingFunction(interpolationPhi(phi,PhiI,R[j],Theta[j],f,demid,deltar,layers)))

            testInterpolation = np.array(testInterpolation)
            plotListTest[configurationIndex][parameterValueIndex] = np.sqrt(np.average(np.abs(testInterpolation - analyticalValues)**2))

        #Computing error over the z=0 planeA
        if(relativeError == "y"):
            plotListA[configurationIndex][parameterValueIndex] = np.sqrt(np.average(np.abs((numericValuesA - analyticalValues)/analyticalValues)**2))
            plotListN[configurationIndex][parameterValueIndex] = np.sqrt(np.average(np.abs((numericValuesN - analyticalValues)/analyticalValues)**2))
        else:
            plotListA[configurationIndex][parameterValueIndex] = np.sqrt(np.average(np.abs((numericValuesA - analyticalValues))**2))   
            plotListN[configurationIndex][parameterValueIndex] = np.sqrt(np.average(np.abs((numericValuesN - analyticalValues))**2))

    #Creating plots
    figA, axA = plt.subplots()
    figA.canvas.manager.set_window_title('Analytical results comparaison')
    figN, axN = plt.subplots()
    figN.canvas.manager.set_window_title('Numerical results comparaison')

    cmap = plt.cm.get_cmap('gist_rainbow', len(interestConfigurations))

    title = errorType + " error (" + postProcessingID + ") computed with : \n" 

    for j,name in enumerate(parametersList[1:]): 
        if(len(np.unique(interestConfigurations[:,j])) == 1):
            title += name + " = " + str(interestConfigurations[0,j]) 
            if(parametersUnits[1:][j] != " "):
                title += " (" + parametersUnits[1:][j] + ")"
            title += " "
    title = title[:-1]

    scalingFunction = lambda x: x
    log = input("Log scale ? y/n")
    if(log == "y"):
        scalingFunction = lambda x: np.log10(x)
    linearRegression = input("Linear regression ? y/n")

    for i,configuration in enumerate(interestConfigurations):

        label = ""
        for j,name in enumerate(parametersList[1:]):
            if(len(np.unique(interestConfigurations[:,j])) > 1):
                label += name + " = " + str(configuration[j]) 
                if(parametersUnits[1:][j] != " "):
                    label += " (" + parametersUnits[1:][j] + ")"
                label += "\n"
        label = label[:-1]

        plotIndex = np.where(plotListA[i] != 0)

        axA.plot(parameterValues[plotIndex],scalingFunction(plotListA[i][plotIndex]),label=label,color=cmap(i))
        axN.plot(parameterValues[plotIndex],scalingFunction(plotListN[i][plotIndex]),label=label,color=cmap(i))

        if(linearRegression == "y"):
            M = np.vstack((scalingFunction(parameterValues[plotIndex]),np.ones(len(parameterValues[plotIndex])))).T
            VA = np.dot(np.linalg.pinv(M),scalingFunction(plotListA[i][plotIndex]))
            VN = np.dot(np.linalg.pinv(M),scalingFunction(plotListN[i][plotIndex]))

            axA.plot(parameterValues[plotIndex],VA[0]*scalingFunction(parameterValues[plotIndex])+VA[1],label=label,color=cmap(i),linestyle='dashed')
            axA.annotate(str(np.round(VA[0],2)) + "log(x) + " + str(np.round(VA[1],2)),(np.average(parameterValues[plotIndex]),0.01 + VA[0]*np.average(scalingFunction(parameterValues[plotIndex]))+VA[1]),color=cmap(i))
            axN.plot(parameterValues[plotIndex],VN[0]*scalingFunction(parameterValues[plotIndex])+VN[1],label=label,color=cmap(i),linestyle='dashed')
            axN.annotate(str(np.round(VN[0],2)) + "log(x) + " + str(np.round(VN[1],2)),(np.average(parameterValues[plotIndex]),0.01 + VN[0]*np.average(scalingFunction(parameterValues[plotIndex]))+VN[1]),color=cmap(i))

        if(analyticalFunctionID == "dPn"):
            axN.plot(parameterValues[plotIndex],scalingFunction(plotListTest[i][plotIndex]),label=label,color=cmap(i),linestyle='dashed')

    if(normalisation == "y"):
        axA.set_xlabel(parameter + r"/$\lambda$")
        axN.set_xlabel(parameter + r"/$\lambda$")
    else:
        axA.set_xlabel(parameter + " (" + parametersUnits[0] + ")")
        axN.set_xlabel(parameter + " (" + parametersUnits[0] + ")")

    if(log=="y"):
        axA.set_ylabel("log(Average " + errorType + " error)")
        axA.set_xscale('log')  
        axN.set_ylabel("log(Average " + errorType + " error)")
        axN.set_xscale('log')  
    else:
        axA.set_ylabel("Average " + errorType + " error")
        axN.set_ylabel("Average " + errorType + " error")
    
    axA.set_title(title)
    axN.set_title(title)    
    axA.legend()
    axN.legend()

    plt.show()

### MAIN ###

postProcessing = input("Post processing function ? " + str(list((postProcessingFunctions.keys()))))
analytical = input("Analytical function ? " + str(list((analyticalFunctions.keys()))))
plotError(postProcessing,analytical)