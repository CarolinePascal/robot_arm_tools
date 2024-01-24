#!/usr/bin/python3

#TODO
# DATA PROCESSING
#-> Mesures moyennées sur tous les points de controle, sur plusieurs acquisitions
#-> Bonne métrique pour le calcul des erreurs (relative, absolue)
#-> Simuler une mesure avec résolution plus faible (échantillonnage des points mesurés)
#-> Haute fréquences : regarder longueure d'onde, taille de la sphère, résolution, etc.

# UNCERTAINTIES
#-> Moyenne ecart type => Tracer les deux en même temps ?
#-> Plus de sigmas, explosion quand sigma = resolution => A voir sur les prochains tracés
#-> Trace en fonction des valeurs normalisées r1/lambda, r2/lambda, resolution/lambda
#-> Sigma sur la mesure => Estimer des valeurs de sigma, peut-être en fonction de la fréquence ?
#-> Introduire distance caractéristique source avec dipole


#System packages
import sys
import os
import csv
import copy

#Utility packages
import numpy as np

#Mesh packages
import meshio as meshio
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) + "/scripts")
from MeshTools import generateSphericMesh

#Custom tools packages
from acousticTools import *
from plotTools import *
figsize = (10,9)

from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter

## Function plotting the error between computed values and analytical values for an output files folder according to a given parameter
#  @param postProcessingID ID of the post-processing function (c.f. plotTools.py)
#  @param analyticalFunctionID ID of the analytical function (c.f. acousticTools.py)
#  @param errorID ID of the error function (c.f. plotTools.py)
def plotError(postProcessingID,analyticalFunctionID,errorID,**kwargs):

    #Get post-processing and analytical functions
    postProcessingFunction = postProcessingFunctions[postProcessingID]
    analyticalFunction = analyticalFunctions[analyticalFunctionID]
    errorFunction = errorFunctions[errorID]

    #Get parameters names, values, units and corresponding output files
    P,parametersList,parametersUnits,fileList = getParametersConfigurations()

    #Studied parameter ?
    if("abscissaParameter" in kwargs and kwargs["abscissaParameter"] in parametersList):
        parameter = kwargs["abscissaParameter"]
    else:
        parameter = input("Abscissa parameter ? (default : resolution)" + str(parametersList))
        if(parameter not in parametersList):
            parameter = "resolution"

    try:
        parameterIndex = np.where(parametersList == parameter)[0][0]
    except IndexError:
        print("INVALID PARAMETER")
        sys.exit(-1)      

    #Alternative for resolution parameter
    verticesNumber = False
    if(parameter == "resolution"):
        if("verticesNumber" in kwargs and kwargs["verticesNumber"] == True):
            verticesNumber = True
            parameter = "vertices"
        elif("verticesNumber" in kwargs and kwargs["verticesNumber"] == False):
            verticesNumber = False
        else:
            flag = input("Number of vertices instead of resolution ? y/n")
            if(flag == "y"):  
                verticesNumber = True    
                parameter = "vertices"

    #Put the studied parameter in the first position in each parameters configurations
    P[:,[0,parameterIndex]] = P[:,[parameterIndex,0]]
    parametersList[[0,parameterIndex]] = parametersList[[parameterIndex,0]]
    parametersUnits[[0,parameterIndex]] = parametersUnits[[parameterIndex,0]]

    #Get all possible values for the studied parameter
    parameterValues = np.unique(P[:,0])

    #Any fixed values for the studied parameter ?
    if("abscissaValues" in kwargs):
        fixedParameterValues = kwargs["abscissaValues"]
    else:
        fixedParameterValues = list(input("Abscissa parameter values ? (default : *) " + str(np.sort(parameterValues)) + " (" + parametersUnits[0] + ") ").split(' '))

    try:
        fixedParameterValues = [float(item) for item in fixedParameterValues]
    except ValueError:
        fixedParameterValues = parameterValues

    #Delete useless parameters configurations
    fixedParameterValuesIndices = [item in fixedParameterValues for item in P[:,0]]
    P = P[fixedParameterValuesIndices]
    fileList = fileList[fixedParameterValuesIndices]

    #Get all possible values for the studied parameter
    parameterValues = np.unique(P[:,0])

    #Get the scaling factors in case of normalisation
    normalisation = False
    if(parametersUnits[0] == "m"):
        if("normalisation" in kwargs and kwargs["normalisation"] == True):
            normalisation = True
        elif("normalisation" in kwargs and kwargs["normalisation"] == False):
            normalisation = False
        else:
            flag = input("Normalise abscissa according to wavelength ? y/n")
            if(flag == "y"):  
                normalisation = True    

    scalingFactors = np.ones(np.shape(P)[0])

    if(normalisation):
        if(parameter == "resolution" or parameter == "size"):
            #scalingFactor = 1/lambda = f/c
            scalingFactors = P[:,np.where(parametersList=="frequency")[0][0]]/c
            #scalingFactor = k = 2*pi*f/c
            scalingFactors *= 2*np.pi
        elif(parameter == "sigmaPosition"):
            scalingFactors = 1/P[:,np.where(parametersList=="resolution")[0][0]]

    #Sort the output files according to the studied parameter values
    sortedIndices = np.argsort(P[:,0])
    P = P[sortedIndices]
    fileList = fileList[sortedIndices]
    scalingFactors = scalingFactors[sortedIndices]

    lastIndex = []
    for value in parameterValues[:-1]:  #The last index is the last one of the list !
        index = np.where(P[:,0] == value)[0][-1] + 1
        lastIndex.append(index)

    #Split the parameters configurations according to the studied parameter values
    splittedP = np.split(P,lastIndex) 

    #Get all possible parameters configurations
    interestConfigurations = np.unique(P[:,1:],axis=0)

    #Select only the parameters configurations which are computed for each studied parameter value
    #for configurations in splittedP:
        #configurations = configurations[:,1:]
        #interestConfigurations = interestConfigurations[(interestConfigurations[:, None] == configurations).all(-1).any(1)]

    if(len(interestConfigurations) == 0):
        print("[WARNING]: Inconsistent data, missing outputs for all parameters configurations.")
        interestConfigurations = np.unique(P[:,1:],axis=0)

    #Fixed parameter ? 
    fixedParameters = []
    fixedValues = []

    if("fixedParameters" in kwargs):
        fixedParameters = kwargs["fixedParameters"]
        if(not isinstance(fixedParameters, list)):
            fixedParameters = [fixedParameters]

        if("fixedValues" in kwargs):
            if(not isinstance(kwargs["fixedValues"], list) and len(fixedParameters) == 1):
                try:
                    fixedValues = [float(kwargs["fixedValues"])]
                except ValueError:
                    fixedValues = [None]

            elif(isinstance(kwargs["fixedValues"], list) and len(fixedParameters) == len(kwargs["fixedValues"])):
                fixedValues = kwargs["fixedValues"]
                for i,itemList in enumerate(fixedValues):
                    try:
                        fixedValues[i] = [float(item) for item in itemList]
                    except ValueError:
                        fixedValues[i] = None

        else:
            fixedParameters = []

    variableParametersList = parametersList[1:]
    if(len(fixedParameters) == 0):
        flag = input("Any fixed parameter ? y/n")

        while(flag == "y"):
            fixedParameter = input("What parameter ? " + str(variableParametersList))
            fixedParameterIndex = np.where(parametersList[1:] == fixedParameter)[0][0]
            tmpValue = list(input("what values ? " + str(np.unique(interestConfigurations[:,fixedParameterIndex])) + " (" + parametersUnits[1:][fixedParameterIndex] + ") ").split(' '))
            tmpValue = [float(item) for item in tmpValue]

            #Select only the parameters configurations containing the fixed parameters
            interestConfigurations = interestConfigurations[np.where(np.isin(interestConfigurations[:,fixedParameterIndex], tmpValue))[0]]
            variableParametersList = np.delete(variableParametersList,np.where(variableParametersList == fixedParameter)[0][0])
            flag = input("Any fixed parameter ? y/n")

    else:
        for fixedParameter,fixedValue in zip(fixedParameters,fixedValues):
            print(str(fixedParameter) + " : " + str(fixedValue))

            try:
                fixedParameterIndex = np.where(parametersList[1:] == fixedParameter)[0][0]
            except:
                print("Skipping fixed parameter " + fixedParameter + " : invalid parameter")
                continue

            if(fixedValue is None):
                fixedValue = np.unique(interestConfigurations[:,fixedParameterIndex])

            interestConfigurations = interestConfigurations[np.where(np.isin(interestConfigurations[:,fixedParameterIndex],fixedValue))[0]]
            variableParametersList = np.delete(variableParametersList,np.where(variableParametersList == fixedParameter)[0][0])

    #Plot reference values too !
    referenceInterestConfigurations = np.empty((0,np.shape(interestConfigurations)[1]))
    if(parameter != "sigmaPosition" and "iteration" in variableParametersList):
        sigmaPositionIndex = np.where(parametersList[1:] == "sigmaPosition")[0][0]
        iterationIndex = np.where(parametersList[1:] == "iteration")[0][0]

        for interestConfiguration in interestConfigurations:
            if(interestConfiguration[sigmaPositionIndex] != 0):
                tmpConfiguration = copy.deepcopy(interestConfiguration)
                tmpConfiguration[sigmaPositionIndex] = 0
                tmpConfiguration[iterationIndex] = 1

                if(not (tmpConfiguration == referenceInterestConfigurations).all(axis=1).any()):
                    referenceInterestConfigurations = np.vstack((referenceInterestConfigurations,tmpConfiguration))

    #Get interest files, scaling factors and complete configurations (i.e. with the studied parameter)
    interestConfigurationsIndices = np.hstack([np.where((P[:,1:] == configuration).all(axis=1))[0] for configuration in interestConfigurations])

    parameterValues = np.unique(P[:,0])

    if(len(referenceInterestConfigurations) != 0):
        referenceInterestConfigurationsNumber = len(referenceInterestConfigurations)
        referenceInterestConfigurationsIndices = np.hstack([np.where((P[:,1:] == configuration).all(axis=1))[0] for configuration in referenceInterestConfigurations])
        referenceFileList = fileList[referenceInterestConfigurationsIndices]
        referenceP = P[referenceInterestConfigurationsIndices]
        referenceScalingFactors = scalingFactors[referenceInterestConfigurationsIndices]

    interestConfigurationsNumber = len(interestConfigurations)
    if("iteration" in variableParametersList):
        iterationIndex = np.where(parametersList[1:] == "iteration")[0][0]
        iterationNumber = len(np.unique(interestConfigurations[:,iterationIndex]))
        interestConfigurationsNumber = len(np.unique(np.delete(interestConfigurations,iterationIndex,1),axis=0))

        scalingFactors = scalingFactors[interestConfigurationsIndices][::len(parameterValues)*iterationNumber]
    else:
        scalingFactors = scalingFactors[interestConfigurationsIndices][::len(parameterValues)]
                
    #Create the interest configurations / plot values matrix
    if(postProcessingID == "re/im"):
        if("iteration" in variableParametersList):
            #plotListA = np.zeros((2,iterationNumber,interestConfigurationsNumber,len(parameterValues)))
            plotListN = np.zeros((2,iterationNumber,interestConfigurationsNumber,len(parameterValues)))
        else:
            #plotListA = np.zeros((2,interestConfigurationsNumber,len(parameterValues)))
            plotListN = np.zeros((2,interestConfigurationsNumber,len(parameterValues)))

        if(len(referenceInterestConfigurations) != 0):
            referencePlotListN = np.zeros((2,referenceInterestConfigurationsNumber,len(parameterValues)))
    else:
        if("iteration" in variableParametersList):
            #plotListA = np.zeros((1,iterationNumber,interestConfigurationsNumber,len(parameterValues)))
            plotListN = np.zeros((1,iterationNumber,interestConfigurationsNumber,len(parameterValues)))
        else:
            #plotListA = np.zeros((1,interestConfigurationsNumber,len(parameterValues)))
            plotListN = np.zeros((1,interestConfigurationsNumber,len(parameterValues)))

        if(len(referenceInterestConfigurations) != 0):
            referencePlotListN = np.zeros((1,referenceInterestConfigurationsNumber,len(parameterValues)))

    plotListP = np.zeros((interestConfigurationsNumber,len(parameterValues)))
    if(len(referenceInterestConfigurations) != 0):
        referencePlotListP = np.zeros((referenceInterestConfigurationsNumber,len(parameterValues)))

    #Get indices frequency (mandatory parameter !) and dipole distance (optional parameter, default is 0)
    frequencyIndex = np.where(parametersList=="frequency")[0][0]
    try:
        dipoleDistanceIndex = np.where(parametersList=="dipoleDistance")[0][0]
    except IndexError:
        dipoleDistanceIndex = None

    #Relative or absolute error ?
    errorType = "absolute"
    if("errorType" in kwargs and kwargs["errorType"] in ["absolute","relative"]):
        errorType = kwargs["errorType"]
    else:
        flag = input("Relative error ? y/n")
        if(flag == "y"):
            errorType = "relative"

    for interestConfigurationIndex, interestConfiguration in enumerate(np.unique(np.delete(interestConfigurations,iterationIndex,1),axis=0)):

        print("Plot : " + str(interestConfigurationIndex+1) + " on " + str(interestConfigurationsNumber))
        
        print("Scaling factor : " + str(scalingFactors[interestConfigurationIndex]))

        for parameterValueIndex,parameterValue in enumerate(parameterValues):
                    
            configuration = np.concatenate(([parameterValue],interestConfiguration))

            if("iteration" in variableParametersList):
                fileIndices = np.where((np.delete(P,iterationIndex+1,1) == configuration).all(axis=1))[0]
            else:
                fileIndices = np.where((P == configuration).all(axis=1))[0]

            #Get configuration frequency and dipole distance parameters
            f = configuration[frequencyIndex]
            k = 2*np.pi*f/c
            halfDipoleDistance = configuration[dipoleDistanceIndex]/2 if dipoleDistanceIndex is not None else 0

            #Create empty arrays
            #numericValuesA = []
            numericValuesN = []
            analyticalValues = []

            for l,fileIndex in enumerate(fileIndices):

                #Create empty arrays
                #numericValuesA.append([])
                numericValuesN.append([])
                analyticalValues.append([])
                R = []
                Theta = []
                Phi = []

                #Fill arrays from output file and analytical function
                with open(fileList[fileIndex], newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=';', quotechar='|')

                    for row in reader:
                        x = float(row[0])
                        y = float(row[1])
                        z = float(row[2])

                        R.append(np.sqrt(x*x + y*y + z*z))
                        Theta.append(np.arctan2(np.sqrt(x*x + y*y),z))
                        Phi.append(np.arctan2(y,x))

                        analyticalValues[l].append(analyticalFunction(f,np.sqrt(x*x + y*y + z*z),np.arctan2(np.sqrt(x*x + y*y),z),np.arctan2(y,x),halfDipoleDistance))
                        #numericValuesA[l].append(complex(float(row[3]),float(row[4])))
                        numericValuesN[l].append(complex(float(row[5]),float(row[6])))               

                R = np.array(R)
                Theta = np.array(Theta)
                Phi = np.array(Phi)

                #Remove outliers
                outliers = np.concatenate((np.where(np.isinf(np.abs(analyticalValues[l])))[0],np.where(np.abs(analyticalValues[l]) < np.mean(np.abs(analyticalValues[l])) - 2*np.std(np.abs(analyticalValues[l])))[0]))

                R = np.delete(R,outliers)
                Theta = np.delete(Theta,outliers)
                Phi = np.delete(Phi,outliers)
                analyticalValues[l] = postProcessingFunction(np.delete(analyticalValues[l],outliers))
                numericValuesN[l] = postProcessingFunction(np.delete(numericValuesN[l],outliers))
                #numericValuesA[l] = postProcessingFunction(np.delete(numericValuesA[l],outliers))

            #Compute error over the z=0 plane
            if(verticesNumber):
                #TODO Non spheric mesh ?
                sizeIndex = np.where(parametersList=="size")[0][0]
                #In this case, resolutionIndex = 0
                try:
                    mesh = meshio.read(os.path.dirname(os.path.abspath(__file__)) + "/config/meshes/sphere/" + str(np.round(configuration[sizeIndex],4)) + "_" + str(np.round(configuration[0],4)) + ".mesh")
                    plotListP[interestConfigurationIndex][parameterValueIndex] = len(mesh.faces)*scalingFactors[interestConfigurationIndex]
                except meshio._exceptions.ReadError:
                    points,_ = generateSphericMesh(np.round(configuration[sizeIndex],4),np.round(configuration[0],4))
                    plotListP[interestConfigurationIndex][parameterValueIndex] = len(points)*scalingFactors[interestConfigurationIndex]
            else:
                plotListP[interestConfigurationIndex][parameterValueIndex] = parameterValue*scalingFactors[interestConfigurationIndex]

            #numericValuesA = np.array(numericValuesA)
            numericValuesN = np.array(numericValuesN)
            analyticalValues = np.array(analyticalValues)

            if("iteration" in variableParametersList):
                for l,(analytical, numN) in enumerate(zip(analyticalValues,numericValuesN)):
                    if(errorType == "relative"):
                        #plotListA[:,l,interestConfigurationIndex,parameterValueIndex] = errorFunction(numA - analytical)/errorFunction(analytical)
                        plotListN[:,l,interestConfigurationIndex,parameterValueIndex] = errorFunction(numN - analytical)/errorFunction(analytical)
                    else:
                        #plotListA[:,l,interestConfigurationIndex,parameterValueIndex] = errorFunction(numA - analytical)
                        plotListN[:,l,interestConfigurationIndex,parameterValueIndex] = errorFunction(numN - analytical)
            else:
                if(errorType == "relative"):
                    #plotListA[:,interestConfigurationIndex,parameterValueIndex] = errorFunction(numericValuesA - analyticalValues)/errorFunction(analyticalValues)
                    plotListN[:,interestConfigurationIndex,parameterValueIndex] = errorFunction(numericValuesN - analyticalValues)/errorFunction(analyticalValues)
                else:
                    #plotListA[:,interestConfigurationIndex,parameterValueIndex] = errorFunction(numericValuesA - analyticalValues)
                    plotListN[:,interestConfigurationIndex,parameterValueIndex] = errorFunction(numericValuesN - analyticalValues)

    for referenceInterestConfigurationIndex, referenceInterestConfiguration in enumerate(referenceInterestConfigurations):
        print("Plot : " + str(referenceInterestConfigurationIndex+1) + " on " + str(referenceInterestConfigurationsNumber))
        
        print("Scaling factor : " + str(referenceScalingFactors[referenceInterestConfigurationIndex]))

        for parameterValueIndex,parameterValue in enumerate(parameterValues):
                    
            configuration = np.concatenate(([parameterValue],referenceInterestConfiguration))
            fileIndices = np.where((P == configuration).all(axis=1))[0]

            #Get configuration frequency and dipole distance parameters
            f = configuration[frequencyIndex]
            k = 2*np.pi*f/c
            halfDipoleDistance = configuration[dipoleDistanceIndex]/2 if dipoleDistanceIndex is not None else 0

            #Create empty arrays
            #numericValuesA = []
            numericValuesN = []
            analyticalValues = []

            for l,fileIndex in enumerate(fileIndices):

                #Create empty arrays
                #numericValuesA.append([])
                numericValuesN.append([])
                analyticalValues.append([])
                R = []
                Theta = []
                Phi = []

                #Fill arrays from output file and analytical function
                with open(fileList[fileIndex], newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=';', quotechar='|')

                    for row in reader:
                        x = float(row[0])
                        y = float(row[1])
                        z = float(row[2])

                        R.append(np.sqrt(x*x + y*y + z*z))
                        Theta.append(np.arctan2(np.sqrt(x*x + y*y),z))
                        Phi.append(np.arctan2(y,x))

                        analyticalValues[l].append(analyticalFunction(f,np.sqrt(x*x + y*y + z*z),np.arctan2(np.sqrt(x*x + y*y),z),np.arctan2(y,x),halfDipoleDistance))
                        #numericValuesA[l].append(complex(float(row[3]),float(row[4])))
                        numericValuesN[l].append(complex(float(row[5]),float(row[6])))               

                R = np.array(R)
                Theta = np.array(Theta)
                Phi = np.array(Phi)

                #Remove outliers
                outliers = np.concatenate((np.where(np.isinf(np.abs(analyticalValues[l])))[0],np.where(np.abs(analyticalValues[l]) < np.mean(np.abs(analyticalValues[l])) - 2*np.std(np.abs(analyticalValues[l])))[0]))

                R = np.delete(R,outliers)
                Theta = np.delete(Theta,outliers)
                Phi = np.delete(Phi,outliers)
                analyticalValues[l] = postProcessingFunction(np.delete(analyticalValues[l],outliers))
                numericValuesN[l] = postProcessingFunction(np.delete(numericValuesN[l],outliers))
                #numericValuesA[l] = postProcessingFunction(np.delete(numericValuesA[l],outliers))

            #Compute error over the z=0 plane
            if(verticesNumber):
                #TODO Non spheric mesh ?
                sizeIndex = np.where(parametersList=="size")[0][0]
                #In this case, resolutionIndex = 0
                try:
                    mesh = meshio.read(os.path.dirname(os.path.abspath(__file__)) + "/config/meshes/sphere/" + str(np.round(configuration[sizeIndex],4)) + "_" + str(np.round(configuration[0],4)) + ".mesh")
                    referencePlotListP[referenceInterestConfigurationIndex][parameterValueIndex] = len(mesh.faces)*referenceScalingFactors[referenceInterestConfigurationIndex]
                except meshio._exceptions.ReadError:
                    points,_ = generateSphericMesh(np.round(configuration[sizeIndex],4),np.round(configuration[0],4))
                    referencePlotListP[referenceInterestConfigurationIndex][parameterValueIndex] = len(points)*referenceScalingFactors[referenceInterestConfigurationIndex]
            else:
                referencePlotListP[referenceInterestConfigurationIndex][parameterValueIndex] = parameterValue*referenceScalingFactors[referenceInterestConfigurationIndex]

            #numericValuesA = np.array(numericValuesA)
            numericValuesN = np.array(numericValuesN)
            analyticalValues = np.array(analyticalValues)

            if(errorType == "relative"):
                #plotListA[:,interestConfigurationIndex,parameterValueIndex] = errorFunction(numericValuesA - analyticalValues)/errorFunction(analyticalValues)
                referencePlotListN[:,referenceInterestConfigurationIndex,parameterValueIndex] = errorFunction(numericValuesN - analyticalValues)/errorFunction(analyticalValues)
            else:
                #plotListA[:,interestConfigurationIndex,parameterValueIndex] = errorFunction(numericValuesA - analyticalValues)
                referencePlotListN[:,referenceInterestConfigurationIndex,parameterValueIndex] = errorFunction(numericValuesN - analyticalValues)


    #Create plots
    if(postProcessingID == "re/im"):
        #figA, axA = plt.subplots(2,figsize=figsize)
        figN, axN = plt.subplots(2,figsize=figsize)
    else:
        #figA, axA = plt.subplots(1,figsize=figsize)
        figN, axN = plt.subplots(1,figsize=figsize)
        #axA = [axA]
        axN = [axN]

    #figA.canvas.manager.set_window_title('Analytical results comparaison - ' + os.path.basename(os.getcwd()))
    figN.canvas.manager.set_window_title('Numerical results comparaison - ' + os.path.basename(os.getcwd()))

    cmap = plt.cm.get_cmap('tab10')

    title = errorType + " error computed with : \n" 
    titleCounter = 0

    for j,name in enumerate(parametersList[1:]): 
        if(name == "iteration"):
            continue

        if(len(np.unique(interestConfigurations[:,j])) == 1):
            if(titleCounter < 3):
                title += name + " = " + str(interestConfigurations[0,j]) 
                if(parametersUnits[1:][j] != " "):
                    title += " " + parametersUnits[1:][j]
                title += " - "
                titleCounter += 1
            else:
                title = title[:-3]
                title += "\n" + name + " = " + str(interestConfigurations[0,j]) 
                if(parametersUnits[1:][j] != " "):
                    title += " " + parametersUnits[1:][j] + " - "
                titleCounter = 0

    title = title[:-3]
    title = title[0].upper() + title[1:]

    scalingFunction = lambda x: x

    logScale = False
    if("logScale" in kwargs and kwargs["logScale"] == True):
        logScale = True
    elif("logScale" in kwargs and kwargs["logScale"] == False):
        logScale = False
    else:
        flag = input("Log scale ? y/n")
        if(flag):
            logScale = True

    if(logScale):
        if(errorType == "absolute" and (postProcessingID == "id" or postProcessingID == "mod")):
            scalingFunction = lambda x: 20*np.log10(x/Pref)
        else:
            scalingFunction = lambda x: np.log10(x)   

    if(not logScale and errorType == "relative"):
        scalingFunction = lambda x: 100*x  

    linearRegression = False
    if("linearRegression" in kwargs and kwargs["linearRegression"] == True):
        linearRegression = True
    elif("linearRegression" in kwargs and kwargs["linearRegression"] == False):
        linearRegression = False
    else:
        flag = input("Linear regression ? y/n")
        if(flag):
            linearRegression = True

    AlphaMean = []
    AlphaMax = []

    for i,configuration in enumerate(np.unique(np.delete(interestConfigurations,iterationIndex,1),axis=0)):

        label = ""
        for j,name in enumerate(parametersList[1:]):
            if(name == "iteration"):
                continue
            elif(name == "frequency"):
                plotName = "f"
            elif(name == "size"):
                plotName = "D"
            elif(name == "resolution"):
                plotName = "h"
            else:
                plotName = name

            if(len(np.unique(interestConfigurations[:,j])) > 1):
                label += plotName + " = " + str(configuration[j]) 
                if(parametersUnits[1:][j] != " "):
                    label += " " + parametersUnits[1:][j]
                label += "\n"
        label = label[:-1]

        for j,(axNi,plotN) in enumerate(zip(axN,plotListN)):

            # shapeA = np.shape(plotA)
            # if(len(shapeA) >= 3 and shapeA[0] > 1):
            #     minSpace, maxSpace = np.ones(len(plotA[0,i]))*10**15, np.ones(len(plotA[0,i]))*10**-15
            #     for l,subPlotA in enumerate(plotA[:,i]):
            #         if(l==0):
            #             axAi.plot(plotListP[i],scalingFunction(subPlotA),label=label,color=cmap(i),marker="+",linestyle = 'None')
            #         else:
            #             axAi.plot(plotListP[i],scalingFunction(subPlotA),color=cmap(i),marker="+",linestyle = 'None')
            #         maxSpace = np.maximum(maxSpace,scalingFunction(subPlotA))
            #         minSpace = np.minimum(minSpace,scalingFunction(subPlotA))
            #     axAi.fill_between(plotListP[i],minSpace,maxSpace,color=cmap(i),alpha=0.2)

            # else:
            #     if(len(shapeA) >= 3):
            #         plotA = plotA[0]
            #     axAi.plot(plotListP[i],scalingFunction(plotA[i]),label=label,color=cmap(i),marker="+",linestyle = 'None')

            shapeN = np.shape(plotN)
            if(len(shapeN) >= 3 and shapeN[0] > 1):
                minSpace, maxSpace = np.ones(len(plotN[0,i]))*10**10, -np.ones(len(plotN[0,i]))*10**10
                for l,subPlotN in enumerate(plotN[:,i]):
                    if(l==0):
                        axNi.plot(plotListP[i],scalingFunction(subPlotN),label=label,color=cmap(i),marker=markers[i],linestyle='None',markerfacecolor='None',markersize=7,markeredgewidth=1)
                    else:
                        axNi.plot(plotListP[i],scalingFunction(subPlotN),color=cmap(i),marker=markers[i],linestyle='None',markerfacecolor='None',markersize=7,markeredgewidth=1)
                    maxSpace = np.maximum(maxSpace,scalingFunction(subPlotN))
                    minSpace = np.minimum(minSpace,scalingFunction(subPlotN))
                axNi.fill_between(plotListP[i],minSpace,maxSpace,color=cmap(i),alpha=0.15)
                plotMaxN = np.max(scalingFunction(plotN[:,i]),axis=0)
                plotMeanN = np.mean(scalingFunction(plotN[:,i]),axis=0)

                #if(len(referenceInterestConfigurations) != 0):
                    #plotRefN = scalingFunction(referencePlotListN[:,i]).flatten()

                if(linearRegression):
                    M = np.vstack((scalingFunction(plotListP[i]),np.ones(len(parameterValues)))).T
                    VN = np.dot(np.linalg.pinv(M),plotMaxN)
                    R2N = 1 - np.sum((plotMaxN - (VN[0]*scalingFunction(plotListP[i])+VN[1]))**2)/np.sum((plotMaxN - np.mean(plotMaxN))**2)

                    #axNi.plot(plotListP[i],VN[0]*scalingFunction(plotListP[i])+VN[1],label=r"$\alpha$" + " = " + str(np.round(VN[0],3)) + ", " + r"$\beta$" + " = " + str(np.round(VN[1],3)) + "\n$R^2$ = " + str(np.round(R2N,3)),color=cmap(i),linestyle="dashed")
                    axNi.plot(plotListP[i],VN[0]*scalingFunction(plotListP[i])+VN[1],label="slope = " + str(np.round(VN[0],3)),color=cmap(i),linestyle="dashed",linewidth=2)
                    AlphaMax.append(VN[0])

                    M = np.vstack((scalingFunction(plotListP[i]),np.ones(len(parameterValues)))).T
                    VN = np.dot(np.linalg.pinv(M),plotMeanN)
                    R2N = 1 - np.sum((plotMeanN - (VN[0]*scalingFunction(plotListP[i])+VN[1]))**2)/np.sum((plotMeanN - np.mean(plotMeanN))**2)

                    axNi.plot(plotListP[i],VN[0]*scalingFunction(plotListP[i])+VN[1],label="slope = " + str(np.round(VN[0],3)),color=cmap(i),linestyle="dotted",linewidth=2)
                    AlphaMean.append(VN[0])

                    # if(len(referenceInterestConfigurations) != 0):
                    #     M = np.vstack((scalingFunction(plotListP[i]),np.ones(len(parameterValues)))).T
                    #     VN = np.dot(np.linalg.pinv(M),plotRefN)
                    #     R2N = 1 - np.sum((plotRefN - (VN[0]*scalingFunction(plotListP[i])+VN[1]))**2)/np.sum((plotRefN - np.mean(plotRefN))**2)

                    #     axNi.plot(plotListP[i],VN[0]*scalingFunction(plotListP[i])+VN[1],label="Reference",color=cmap(i),alpha=0.75,zorder=0)

                else:
                    #axNi.plot(plotListP[i],plotMaxN,label="Maximum",color=cmap(i),linestyle='dashed',linewidth=2)
                    #axNi.plot(plotListP[i],plotMeanN,label="Average",color=cmap(i),linestyle='dotted',linewidth=2)
                    axNi.plot(plotListP[i],plotMaxN,color=cmap(i),linestyle='dashed',linewidth=2)
                    axNi.plot(plotListP[i],plotMeanN,color=cmap(i),linestyle='dotted',linewidth=2)
                    #if(len(referenceInterestConfigurations) != 0):
                        #axNi.plot(plotListP[i],plotRefN,label="Reference",color=cmap(i),alpha=0.75,zorder=0)

            else:
                if(len(shapeN) >= 3):
                    plotN = plotN[0]
                axNi.plot(plotListP[i],scalingFunction(plotN[i]),label=label,color=cmap(i),marker=markers[i],linestyle='None',markerfacecolor='None',markersize=7,markeredgewidth=2)

                if(linearRegression):
                    M = np.vstack((scalingFunction(plotListP[i]),np.ones(len(parameterValues)))).T
                    #VA = np.dot(np.linalg.pinv(M),scalingFunction(plotA[i]))
                    VN = np.dot(np.linalg.pinv(M),scalingFunction(plotN[i]))
                    #R2A = 1 - np.sum((scalingFunction(plotA[i]) - (VA[0]*scalingFunction(plotListP[i])+VA[1]))**2)/np.sum((scalingFunction(plotA[i]) - np.mean(scalingFunction(plotA[i])))**2)
                    R2N = 1 - np.sum((scalingFunction(plotN[i]) - (VN[0]*scalingFunction(plotListP[i])+VN[1]))**2)/np.sum((scalingFunction(plotN[i]) - np.mean(scalingFunction(plotN[i])))**2)

                    #axAi.plot(plotListP[i],VA[0]*scalingFunction(plotListP[i])+VA[1],label="(" + str(np.round(VA[0],3)) + "," + str(np.round(VA[1],3)) + ")",color=cmap(i))
                    axNi.plot(plotListP[i],VN[0]*scalingFunction(plotListP[i])+VN[1],label="slope = " + str(np.round(VN[0],3)),color=cmap(i),linestyle="dashed",linewidth=2)

                    AlphaMean.append(VN[0])
                    AlphaMax.append(VN[0])

    titleCounter = 0

    extraPadding = False
    if(parameter == "sigmaPosition" and "robot" in kwargs and kwargs["robot"]):
        if((scalingFactors == scalingFactors[0]).all()):
            extraPadding = True
            axNi.axvline(x=scalingFactors[0]*1.63*10**-3,color="dimgray",linestyle="dashed",linewidth=2)    #2.03
            t = axNi.text(scalingFactors[0]*1.63*10**-3, 1.0025, 'calibrated robot', color='dimgray', transform=axNi.get_xaxis_transform(), ha='center', va='bottom',fontsize = 20)
            #t.set_bbox(dict(facecolor="white",alpha=1.0,linewidth=0))
            axNi.axvline(x=scalingFactors[0]*7.66*10**-3,color="dimgray",linestyle="dashed",linewidth=2)    #14.4
            t = axNi.text(scalingFactors[0]*7.66*10**-3, 1.0025, 'uncalibrated robot', color='dimgray', transform=axNi.get_xaxis_transform(), ha='center', va='bottom', fontsize = 20)
            #t.set_bbox(dict(facecolor="white",alpha=1.0,linewidth=0))

    for axNi in axN:
        if(normalisation):
            if(parameter == "resolution"):
                #axAi.set_xlabel("hk")
                axNi.set_xlabel(r"$hk$")
            elif(parameter == "size"):
                #axAi.set_xlabel("Dk")
                axNi.set_xlabel(r"$Dk$")
            elif(parameter == "sigmaPosition"):
                #axAi.set_xlabel(r"$\sigma_P$k")
                axNi.set_xlabel(r"$\frac{\sigma_P}{h}$")
            else:
                #axAi.set_xlabel(parameter + "/" + r"$\lambda$")
                axNi.set_xlabel(parameter + "/" + r"$\lambda$")
        elif(parameter == "sigmaPosition"):
            #axAi.set_xlabel(r"$\sigma_P$k")
            axNi.set_xlabel(r"$\sigma_P$ (m)")
        elif(parameter == "resolution"):
            axNi.set_xlabel(r"$h$ (m)")
        else:
            #axAi.set_xlabel(parameter + " (" + parametersUnits[0] + ")")
            axNi.set_xlabel(parameter + " (" + parametersUnits[0] + ")")

        if(logScale):
            #axAi.set_xscale('log') 
            axNi.set_xscale('log')
            if(errorType == "absolute" and (postProcessingID == "id" or postProcessingID == "mod")):
                #axAi.set_ylabel("average " + errorType + " error (dB)")
                axNi.set_ylabel("average " + errorType + " error (dB)")
            else:
                #axAi.set_ylabel("log("  + errorType + " error)")
                axNi.set_ylabel(r"$\log(\epsilon)$ (-)",labelpad=15)
              
        else:
            if(errorType == "relative"):
                tmpUnit = r"(\%)"
            else:
                if(postProcessingID == "phase"):
                    tmpUnit = "(rad)"
                else:
                    tmpUnit = "(Pa)"
            #axAi.set_ylabel("average " + errorType + " error " + tmpUnit)
            axNi.set_ylabel("average " + errorType + " error " + tmpUnit)
    
        if(titleCounter < 1):
            #axAi.set_title(title,pad=30)
            #axNi.set_title(title)
            titleCounter += 1  

        #if(len(axAi.get_legend_handles_labels()[0]) != 0 or len(axAi.get_legend_handles_labels()[1]) != 0):
            #axAi.legend(bbox_to_anchor=(0.5,1.0), loc='lower center', ncol=4, borderaxespad=0.25, reverse=False)
        if(len(axNi.get_legend_handles_labels()[0]) != 0 or len(axNi.get_legend_handles_labels()[1]) != 0):
            figN.tight_layout()
            box = axNi.get_position()
            axNi.set_position([box.x0, box.y0, box.width, box.height * 0.94])
            if(extraPadding):
                axNi.legend(bbox_to_anchor=(0.5,1.0), loc='lower center', ncol=2, borderaxespad=1.1, reverse=False,columnspacing=0.1, fontsize=20)
            else:
                axNi.legend(bbox_to_anchor=(0.5,1.0), loc='lower center', ncol=2, borderaxespad=0.25, reverse=False,columnspacing=0.1, fontsize=20)
            # figN.tight_layout()
            # box = axNi.get_position()
            # axNi.set_position([box.x0, box.y0, box.width * 0.75, box.height])
            # axNi.legend(bbox_to_anchor=(1.0,0.5), loc='center left', ncol=1, borderaxespad=0.25, reverse=False)
    
    # axAi.grid(which="major")
    # axAi.grid(linestyle = '--',which="minor")
    # axAi.yaxis.set_major_locator(MaxNLocator(10))
    # axAi.yaxis.set_minor_locator(MaxNLocator(2))
    
    axNi.grid(which="major")
    axNi.grid(linestyle = '--',which="minor")
    axNi.yaxis.set_major_locator(MaxNLocator(10))
    axNi.yaxis.set_minor_locator(MaxNLocator(10))
    axNi.tick_params(axis='both', which='major', pad=7)
    #axNi.xaxis.set_minor_formatter(FuncFormatter(log_formatter))

    #figN.savefig("NormalizedErrorSigmaResolution.pdf", dpi = 300, bbox_inches = 'tight')
    #plt.show()

    return(AlphaMean,AlphaMax)

if __name__ == "__main__": 
    #postProcessing = input("Post processing function ? (default : id) " + str(list((postProcessingFunctions.keys()))))
    postProcessing = "id"
    if(postProcessing not in list((postProcessingFunctions.keys()))):
        postProcessing = "id"

    flagFunction = os.path.basename(os.getcwd()).split("_")[0] in list((analyticalFunctions.keys()))

    if(flagFunction):
        analytical = os.path.basename(os.getcwd()).split("_")[0]
    else:
        analytical = input("Analytical function ? (default : infinitesimalDipole) " + str(list((analyticalFunctions.keys()))))
        if(analytical not in list((analyticalFunctions.keys()))):
            analytical = "infinitesimalDipole"

    #error = input("Error function ? (default : l2) " + str(list((errorFunctions.keys()))))
    error="l2"
    if(error not in list((errorFunctions.keys()))):
        error = "l2"

    kwargs = {}
    #kwargs["abscissaParameter"] = "sigmaPosition"
    #kwargs["abscissaValues"] = [0.0005,  0.00125, 0.0025,  0.00375, 0.005]
    kwargs["abscissaParameter"] = "resolution"
    kwargs["abscissaValues"] = [0.005,  0.0125, 0.025,  0.0375, 0.05]
    #kwargs["abscissaParameter"] = "size"
    #kwargs["abscissaValues"] = "*"

    kwargs["fixedParameters"] = ["resolution","size","frequency","sigmaPosition","sigmaMeasure","dipoleDistance"]
    kwargs["fixedValues"] = [[0.05],[0.5],[100,500,1000,5000],[0.0],[0.0],[0.05]]

    #kwargs["fixedParameters"] = ["size","resolution","frequency","sigmaMeasure","dipoleDistance"]
    #kwargs["fixedValues"] = [[0.25],[0.025],[100,500,1000,5000],[0.0],[0.05]]

    kwargs["normalisation"] = False
    kwargs["verticesNumber"] = False

    kwargs["errorType"] = "relative"
    kwargs["linearRegression"] = True
    kwargs["logScale"] = True

    """
    sigmaPositions = [0.0, 0.0005,  0.00125, 0.0025,  0.00375, 0.005]
    AlphaMean = []
    AlphaMax = []

    for sigmaP in sigmaPositions :
        kwargs["fixedValues"][3] = [sigmaP]
        amean, amax = plotError(postProcessing,analytical,error,**kwargs)
        AlphaMean.append(amean)
        AlphaMax.append(amax)

    AlphaMean = np.array(AlphaMean)
    AlphaMax = np.array(AlphaMax)
    print(AlphaMean)
    print(AlphaMax)
    plt.close("all")
    
    figN, axN = plt.subplots(1,figsize=figsize)
    Labels = [r"$\angle_{max}$",r"$\angle_{avg}$"]
    for i,f in enumerate(kwargs["fixedValues"][2]):
        for j,list in enumerate([AlphaMax,AlphaMean]):
            axN.plot(sigmaPositions,list[:,i],color=cmap2(4*i+j),marker=markers[i],linestyle='None',markerfacecolor='None',markersize=10,markeredgewidth=2,label=Labels[j] + " - f = " + str(f) + " Hz")

    axN.axhline(y=0.5,color="dimgray",linestyle='dashed',linewidth=2)
    t = axN.text(-0.02, 0.5, r"$\sqrt{h}$", color='dimgray', transform=axN.get_yaxis_transform(), ha='right', va='center',fontsize = 20)

    figN.tight_layout()
    box = axN.get_position()
    axN.set_position([box.x0, box.y0, box.width, box.height * 0.94])


    axN.legend(bbox_to_anchor=(0.5,1.0), loc='lower center', ncol=2, borderaxespad=0.25, reverse=False,columnspacing=0.1, fontsize=20)
    axN.set_ylabel(r"slope",labelpad=15)
    axN.set_xlabel(r"$\sigma_P$ (m)")
    axN.grid(which="major")
    axN.grid(linestyle = '--',which="minor")
    axN.yaxis.set_major_locator(MaxNLocator(10))
    axN.yaxis.set_minor_locator(MaxNLocator(10))
    axN.tick_params(axis='both', which='major', pad=7)

    figN.savefig("SigmaSlope.pdf", dpi = 300, bbox_inches = 'tight')
    """

