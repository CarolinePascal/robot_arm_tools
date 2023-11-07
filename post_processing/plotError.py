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

#Utility packages
import numpy as np

#Mesh packages
import meshio as meshio
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) + "/scripts")
from MeshTools import generateSphericMesh

#Custom tools packages
from acousticTools import *
from plotTools import *

## Function plotting the error between computed values and analytical values for an output files folder according to a given parameter
#  @param postProcessingID ID of the post-processing function (c.f. plotTools.py)
#  @param analyticalFunctionID ID of the analytical function (c.f. acousticTools.py)
#  @param errorID ID of the error function (c.f. plotTools.py)
def plotError(postProcessingID,analyticalFunctionID,errorID):

    #Get post-processing and analytical functions
    postProcessingFunction = postProcessingFunctions[postProcessingID]
    analyticalFunction = analyticalFunctions[analyticalFunctionID]
    errorFunction = errorFunctions[errorID]

    #Get parameters names, values, units and corresponding output files
    P,parametersList,parametersUnits,fileList = getParametersConfigurations()

    #Studied parameter ?
    parameter = input("Abscissa parameter ? (default : resolution)" + str(parametersList))
    if(parameter not in parametersList):
        parameter = "resolution"

    try:
        parameterIndex = np.where(parametersList == parameter)[0][0]
    except:
        print("INVALID PARAMETER")
        sys.exit(-1)      

    #Alternative for resolution parameter
    verticesNumber = False
    if(parameter == "resolution"):
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
    fixedParameterValues = list(input("Abscissa parameter values ? (default : *) " + str(np.sort(parameterValues)) + " (" + parametersUnits[0] + ") ").split(' '))
    try:
        fixedParameterValues = [float(item) for item in fixedParameterValues]
    except:
        fixedParameterValues = parameterValues

    #Delete useless parameters configurations
    fixedParameterValuesIndices = [item in fixedParameterValues for item in P[:,0]]
    P = P[fixedParameterValuesIndices]
    fileList = fileList[fixedParameterValuesIndices]

    #Get all possible values for the studied parameter
    parameterValues = np.unique(P[:,0])

    #Get the scaling factors in case of normalisation
    normalisation = "n"
    if(parametersUnits[0] == "m"):
        normalisation = input("Normalise abscissa according to wavelength ? y/n")
    scalingFactors = np.ones(np.shape(P)[0])

    if(normalisation == "y"):
        #scalingFactor = 1/lambda = f/c
        scalingFactors = P[:,np.where(parametersList=="frequency")[0][0]]/c

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
    for configurations in splittedP:
        configurations = configurations[:,1:]
        interestConfigurations = interestConfigurations[(interestConfigurations[:, None] == configurations).all(-1).any(1)]

    if(len(interestConfigurations) == 0):
        print("[WARNING]: Inconsistent data, missing outputs for all parameters configurations.")
        interestConfigurations = np.unique(P[:,1:],axis=0)

    #Fixed parameter ? 
    flag = input("Any fixed parameter ? y/n")
    variableParametersList = parametersList[1:]

    while(flag == "y"):
        fixedParameter = input("What parameter ? " + str(variableParametersList))
        fixedParameterIndex = np.where(parametersList[1:] == fixedParameter)[0][0]
        tmpValue = list(input("what values ? " + str(np.unique(interestConfigurations[:,fixedParameterIndex])) + " (" + parametersUnits[1:][fixedParameterIndex] + ") ").split(' '))
        tmpValue = [float(item) for item in tmpValue]

        #Select only the parameters configurations containing the fixed parameters
        interestConfigurations = interestConfigurations[np.where(np.isin(interestConfigurations[:,fixedParameterIndex], tmpValue))[0]]
        variableParametersList = np.delete(variableParametersList,np.where(variableParametersList == fixedParameter)[0][0])
        flag = input("Any fixed parameter ? y/n")

    #Get interest files, scaling factors and complete configurations (i.e. with the studied parameter)
    interestConfigurationsIndices = np.hstack([np.where((P[:,1:] == configuration).all(axis=1))[0] for configuration in interestConfigurations])
    fileList = fileList[interestConfigurationsIndices]
    scalingFactors = scalingFactors[interestConfigurationsIndices]
    P = P[interestConfigurationsIndices]

    interestConfigurationsNumber = len(interestConfigurations)
    if("iteration" in variableParametersList):
        iterationIndex = np.where(parametersList == "iteration")[0][0]
        iterationNumber = len(np.unique(interestConfigurations[:,iterationIndex-1]))
        interestConfigurationsNumber = len(np.unique(np.delete(interestConfigurations,iterationIndex-1,1),axis=0))
        
    #Create the interest configurations / plot values matrix
    if(postProcessingID == "re/im"):
        if("iteration" in variableParametersList):
            plotListA = np.zeros((2,iterationNumber,interestConfigurationsNumber,len(parameterValues)))
            plotListN = np.zeros((2,iterationNumber,interestConfigurationsNumber,len(parameterValues)))
        else:
            plotListA = np.zeros((2,interestConfigurationsNumber,len(parameterValues)))
            plotListN = np.zeros((2,interestConfigurationsNumber,len(parameterValues)))
    else:
        if("iteration" in variableParametersList):
            plotListA = np.zeros((1,iterationNumber,interestConfigurationsNumber,len(parameterValues)))
            plotListN = np.zeros((1,iterationNumber,interestConfigurationsNumber,len(parameterValues)))
        else:
            plotListA = np.zeros((1,interestConfigurationsNumber,len(parameterValues)))
            plotListN = np.zeros((1,interestConfigurationsNumber,len(parameterValues)))
    plotListP = np.zeros((interestConfigurationsNumber,len(parameterValues)))

    #Get indices frequency (mandatory parameter !) and dipole distance (optional parameter, default is 0)
    frequencyIndex = np.where(parametersList=="frequency")[0][0]
    try:
        dipoleDistanceIndex = np.where(parametersList=="dipoleDistance")[0][0]
    except:
        dipoleDistanceIndex = None

    #Relative or absolute error ?
    errorType = "absolute"
    relativeError = input("Relative error ? y/n")
    if(relativeError == "y"):
        errorType = "relative"

    for interestConfigurationIndex, interestConfiguration in enumerate(np.unique(np.delete(interestConfigurations,iterationIndex-1,1),axis=0)):

        print("Plot : " + str(interestConfigurationIndex+1) + " on " + str(interestConfigurationsNumber))

        for parameterValueIndex,parameterValue in enumerate(parameterValues):
                    
            configuration = np.concatenate(([parameterValue],interestConfiguration))

            if("iteration" in variableParametersList):
                fileIndices = np.where((np.delete(P,iterationIndex,1) == configuration).all(axis=1))[0]
            else:
                fileIndices = np.where(P == configuration)[0]

            #Get configuration frequency and dipole distance parameters
            f = configuration[frequencyIndex]
            k = 2*np.pi*f/c
            halfDipoleDistance = configuration[dipoleDistanceIndex]/2 if dipoleDistanceIndex is not None else 0

            #Create empty arrays
            numericValuesA = []
            numericValuesN = []
            analyticalValues = []

            for k,fileIndex in enumerate(fileIndices):

                #Create empty arrays
                numericValuesA.append([])
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

                        analyticalValues[k].append(analyticalFunction(f,np.sqrt(x*x + y*y + z*z),np.arctan2(np.sqrt(x*x + y*y),z),np.arctan2(y,x),halfDipoleDistance))
                        numericValuesA[k].append(complex(float(row[3]),float(row[4])))
                        numericValuesN[k].append(complex(float(row[5]),float(row[6])))               

                R = np.array(R)
                Theta = np.array(Theta)
                Phi = np.array(Phi)

                #Remove outliers
                outliers = np.concatenate((np.where(np.isinf(np.abs(analyticalValues[k])))[0],np.where(np.abs(analyticalValues[k]) < np.mean(np.abs(analyticalValues[k])) - 2*np.std(np.abs(analyticalValues[k])))[0]))

                R = np.delete(R,outliers)
                Theta = np.delete(Theta,outliers)
                Phi = np.delete(Phi,outliers)
                analyticalValues[k] = postProcessingFunction(np.delete(analyticalValues[k],outliers))
                numericValuesN[k] = postProcessingFunction(np.delete(numericValuesN[k],outliers))
                numericValuesA[k] = postProcessingFunction(np.delete(numericValuesA[k],outliers))

            #Compute error over the z=0 plane
            if(verticesNumber):
                #TODO Non spheric mesh ?
                sizeIndex = np.where(parametersList=="size")[0][0]
                try:
                    mesh = meshio.read(os.path.dirname(os.path.abspath(__file__)) + "/config/meshes/sphere/" + str(np.round(configuration[sizeIndex],4)) + "_" + str(np.round(configuration[0],4)) + ".mesh")
                    plotListP[interestConfigurationIndex][parameterValueIndex] = len(mesh.points)*scalingFactors[interestConfigurationIndex]
                except:
                    points,_ = generateSphericMesh(np.round(configuration[sizeIndex],4),np.round(configuration[0],4))
                    plotListP[interestConfigurationIndex][parameterValueIndex] = len(points)*scalingFactors[interestConfigurationIndex]
            else:
                plotListP[interestConfigurationIndex][parameterValueIndex] = configuration[0]*scalingFactors[interestConfigurationIndex]

            numericValuesA = np.array(numericValuesA)
            numericValuesN = np.array(numericValuesN)
            analyticalValues = np.array(analyticalValues)

            #OPTION 1
            #numericValuesA = np.mean(numericValuesA,axis=1)
            #numericValuesN = np.mean(numericValuesN,axis=1)
            #analyticalValues = np.mean(analyticalValues,axis=1)

            #OPTION 2 
            if("iteration" in variableParametersList):
                for l,(analytical, numA, numN) in enumerate(zip(analyticalValues,numericValuesA,numericValuesN)):
                    if(relativeError == "y"):
                        plotListA[:,l,interestConfigurationIndex,parameterValueIndex] = errorFunction(numA - analytical)/errorFunction(analytical)
                        plotListN[:,l,interestConfigurationIndex,parameterValueIndex] = errorFunction(numN - analytical)/errorFunction(analytical)
                    else:
                        plotListA[:,l,interestConfigurationIndex,parameterValueIndex] = errorFunction(numA - analytical)
                        plotListN[:,l,interestConfigurationIndex,parameterValueIndex] = errorFunction(numN - analytical)
            else:
                if(relativeError == "y"):
                    plotListA[:,interestConfigurationIndex,parameterValueIndex] = errorFunction(numericValuesA - analyticalValues)/errorFunction(analyticalValues)
                    plotListN[:,interestConfigurationIndex,parameterValueIndex] = errorFunction(numericValuesN - analyticalValues)/errorFunction(analyticalValues)
                else:
                    plotListA[:,interestConfigurationIndex,parameterValueIndex] = errorFunction(numericValuesA - analyticalValues)
                    plotListN[:,interestConfigurationIndex,parameterValueIndex] = errorFunction(numericValuesN - analyticalValues)

    #Create plots
    if(postProcessingID == "re/im"):
        figA, axA = plt.subplots(2,1)
        figN, axN = plt.subplots(2,1)
    else:
        figA, axA = plt.subplots(1,1)
        figN, axN = plt.subplots(1,1)
        axA = [axA]
        axN = [axN]

    figA.canvas.manager.set_window_title('Analytical results comparaison - ' + os.path.basename(os.getcwd()))
    figN.canvas.manager.set_window_title('Numerical results comparaison - ' + os.path.basename(os.getcwd()))

    cmap = plt.cm.get_cmap('tab10')

    title = errorType + " error computed with : \n" 

    for j,name in enumerate(parametersList[1:]): 
        if(name == "iteration"):
            continue

        if(len(np.unique(interestConfigurations[:,j])) == 1):
            title += name + " = " + str(interestConfigurations[0,j]) 
            if(parametersUnits[1:][j] != " "):
                title += " " + parametersUnits[1:][j] + " - "
    title = title[:-2]
    title = title[0].upper() + title[1:]

    scalingFunction = lambda x: x
    log = input("Log scale ? y/n")
    if(log == "y"):
        if(errorType == "absolute" and (postProcessingID == "id" or postProcessingID == "mod")):
            scalingFunction = lambda x: 20*np.log10(x/Pref)
        else:
            scalingFunction = lambda x: np.log10(x)   

    if(log != "y" and errorType == "relative"):
        scalingFunction = lambda x: 100*x  

    linearRegression = input("Linear regression ? y/n")

    for i,configuration in enumerate(np.unique(np.delete(interestConfigurations,iterationIndex-1,1),axis=0)):

        label = ""
        for j,name in enumerate(parametersList[1:]):
            if(name == "iteration"):
                continue

            if(len(np.unique(interestConfigurations[:,j])) > 1):
                label += name + " = " + str(configuration[j]) 
                if(parametersUnits[1:][j] != " "):
                    label += " (" + parametersUnits[1:][j] + ")"
                label += "\n"
        label = label[:-1]

        for axAi,axNi,plotN,plotA in zip(axA,axN,plotListN,plotListA):

            shapeA = np.shape(plotA)
            if(len(shapeA) >= 3 and shapeA[0] > 1):
                minSpace, maxSpace = np.ones(len(plotA[0,i]))*10**15, np.ones(len(plotA[0,i]))*10**-15
                for k,subPlotA in enumerate(plotA[:,i]):
                    if(k==0):
                        axAi.plot(plotListP[i],scalingFunction(subPlotA),label=label,color=cmap(i),marker="+",linestyle = 'None')
                    else:
                        axAi.plot(plotListP[i],scalingFunction(subPlotA),color=cmap(i),marker="+",linestyle = 'None')
                    maxSpace = np.maximum(maxSpace,scalingFunction(subPlotA))
                    minSpace = np.minimum(minSpace,scalingFunction(subPlotA))
                axAi.fill_between(plotListP[i],minSpace,maxSpace,color=cmap(i),alpha=0.2)

            else:
                if(len(shapeA) >= 3):
                    plotA = plotA[0]
                plotIndex = np.arange(len(plotA[i]))
                if(log == "y"):
                    plotIndex = np.where(plotA[i] != 0)[0]
                axAi.plot(plotListP[i][plotIndex],scalingFunction(plotA[i][plotIndex]),label=label,color=cmap(i),marker="+",linestyle = 'None')

            shapeN = np.shape(plotN)
            if(len(shapeN) >= 3 and shapeN[0] > 1):
                minSpace, maxSpace = np.ones(len(plotN[0,i]))*10**10, np.ones(len(plotN[0,i]))*10**-10
                for k,subPlotN in enumerate(plotN[:,i]):
                    if(k==0):
                        axNi.plot(plotListP[i],scalingFunction(subPlotN),label=label,color=cmap(i),marker="+",linestyle = 'None')
                    else:
                        axNi.plot(plotListP[i],scalingFunction(subPlotN),color=cmap(i),marker="+",linestyle = 'None')
                    maxSpace = np.maximum(maxSpace,scalingFunction(subPlotN))
                    minSpace = np.minimum(minSpace,scalingFunction(subPlotN))
                axNi.fill_between(plotListP[i],minSpace,maxSpace,color=cmap(i),alpha=0.2)

            else:
                if(len(shapeN) >= 3):
                    plotN = plotN[0]
                plotIndex = np.arange(len(plotN[i]))
                if(log == "y"):
                    plotIndex = np.where(plotN[i] != 0)[0]
                axNi.plot(plotListP[i][plotIndex],scalingFunction(plotN[i][plotIndex]),label=label,color=cmap(i),marker="+",linestyle = 'None')

            if(linearRegression == "y" and np.shape(plotN[i])[0] == 1):
                M = np.vstack((scalingFunction(plotListP[i][plotIndex]),np.ones(len(parameterValues[plotIndex])))).T
                #VA = np.dot(np.linalg.pinv(M),scalingFunction(plotA[i][plotIndex]))
                VN = np.dot(np.linalg.pinv(M),scalingFunction(plotN[i][plotIndex]))
                #R2A = 1 - np.sum((scalingFunction(plotA[i][plotIndex]) - (VA[0]*scalingFunction(plotListP[i][plotIndex])+VA[1]))**2)/np.sum((scalingFunction(plotA[i][plotIndex]) - np.mean(scalingFunction(plotA[i][plotIndex])))**2)
                R2N = 1 - np.sum((scalingFunction(plotN[i][plotIndex]) - (VN[0]*scalingFunction(plotListP[i][plotIndex])+VN[1]))**2)/np.sum((scalingFunction(plotN[i][plotIndex]) - np.mean(scalingFunction(plotN[i][plotIndex])))**2)

                #axAi.plot(plotListP[i][plotIndex],VA[0]*scalingFunction(plotListP[i][plotIndex])+VA[1],label="(" + str(np.round(VA[0],3)) + "," + str(np.round(VA[1],3)) + ")",color=cmap(i))
                axNi.plot(plotListP[i][plotIndex],VN[0]*scalingFunction(plotListP[i][plotIndex])+VN[1],label=r"$\alpha$" + " = " + str(np.round(VN[0],3)) + ", " + r"$\beta$" + " = " + str(np.round(VN[1],3)) + ", $R^2$ = " + str(np.round(R2N,3)),color=cmap(i))

    titleCounter = 0
    for axAi,axNi in zip(axA,axN):
        if(normalisation == "y"):
            axAi.set_xlabel(parameter + "/" + r"$\lambda$")
            axNi.set_xlabel(parameter + "/" + r"$\lambda$")
        else:
            axAi.set_xlabel(parameter + " (" + parametersUnits[0] + ")")
            axNi.set_xlabel(parameter + " (" + parametersUnits[0] + ")")

        if(log=="y"):
            axAi.set_xscale('log') 
            axNi.set_xscale('log')
            if(errorType == "absolute" and (postProcessingID == "id" or postProcessingID == "mod")):
                axAi.set_ylabel("average " + errorType + " error (dB)")
                axNi.set_ylabel("average " + errorType + " error (dB)")
            else:
                axAi.set_ylabel("log(average " + errorType + " error)")
                axNi.set_ylabel("log(average " + errorType + " error)")
              
        else:
            if(errorType == "relative"):
                tmpUnit = r"(\%)"
            else:
                if(postProcessingID == "phase"):
                    tmpUnit = "(rad)"
                else:
                    tmpUnit = "(Pa)"
            axAi.set_ylabel("average " + errorType + " error " + tmpUnit)
            axNi.set_ylabel("average " + errorType + " error " + tmpUnit)
    
        if(titleCounter < 1):
            axAi.set_title(title)
            axNi.set_title(title)
            titleCounter += 1  

        if(len(axAi.get_legend_handles_labels()[0]) != 0 or len(axAi.get_legend_handles_labels()[1]) != 0):
            axAi.legend()
        if(len(axNi.get_legend_handles_labels()[0]) != 0 or len(axNi.get_legend_handles_labels()[1]) != 0):
            axNi.legend()
    
    axAi.grid(which="major")
    axAi.grid(linestyle = '--',which="minor")
    axNi.grid(which="major")
    axNi.grid(linestyle = '--',which="minor")
    plt.show()

if __name__ == "__main__": 
    postProcessing = input("Post processing function ? (default : id) " + str(list((postProcessingFunctions.keys()))))
    if(postProcessing not in list((postProcessingFunctions.keys()))):
        postProcessing = "id"

    try:
        flagFunction = os.path.basename(os.getcwd()).split("_")[0] in list((analyticalFunctions.keys()))
    except:
        flagFunction = False

    if(flagFunction):
        analytical = os.path.basename(os.getcwd()).split("_")[0]
    else:
        analytical = input("Analytical function ? (default : infinitesimalDipole) " + str(list((analyticalFunctions.keys()))))
        if(analytical not in list((analyticalFunctions.keys()))):
            analytical = "infinitesimalDipole"

    error = input("Error function ? (default : l2) " + str(list((errorFunctions.keys()))))
    if(error not in list((errorFunctions.keys()))):
        error = "l2"

    plotError(postProcessing,analytical,error)