#!/usr/bin/python3

#System packages
import os
import csv

#Utility packages
import numpy as np

#Mesh packages
import meshio as meshio
from robot_arm_acoustic.MeshTools import generateSphericMesh

#Custom tools packages
from robot_arm_acoustic.simulations.AcousticTools import *
from robot_arm_acoustic.simulations.PlotTools import *
from matplotlib.ticker import MaxNLocator

import itertools
def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

## Function plotting the error between computed values and analytical values for an output files folder according to a given parameter
#  @param postProcessingID ID of the post-processing function (c.f. PlotTools.py)
#  @param analyticalFunctionID ID of the analytical function (c.f. AcousticTools.py)
#  @param errorID ID of the error function (c.f. PlotTools.py)
#  @param figureName Name of the output figure
def plotError(postProcessingID,analyticalFunctionID,errorID,elementType="P1",figureName="figure.pdf",folderName="./",**kwargs):

    #Get post-processing and analytical functions
    postProcessingFunction = postProcessingFunctions[postProcessingID]
    analyticalFunction = analyticalFunctions[analyticalFunctionID]
    errorFunction = errorFunctions[errorID]

    #Get parameters names, values, units and corresponding output files
    parametersConfigurations,parametersList,parametersUnits,fileList = getParametersConfigurations(folderName)

    #Abscissa parameter ?
    if("abscissaParameter" in kwargs and kwargs["abscissaParameter"] in parametersList):
        parameter = kwargs["abscissaParameter"]
    else:
        parameter = input("Abscissa parameter ? (default : resolution)" + str(parametersList))
        if(parameter not in parametersList):
            parameter = "resolution"

    try:
        parameterIndex = np.where(parametersList == parameter)[0][0]
    except IndexError:
        raise ValueError("[ERROR] Invalid abscissa parameter")   

    #Alternative for resolution parameter -> Vertices number
    verticesNumber = False
    if(parameter == "resolution"):
        if("verticesNumber" in kwargs):
            if(kwargs["verticesNumber"]):
                verticesNumber = True
                parameter = "vertices"
        else:
            flag = input("Number of vertices instead of resolution ? y/n")
            if(flag == "y"):  
                verticesNumber = True    
                parameter = "vertices"

    #Put the studied parameter in the first position in each parameters configurations
    parametersConfigurations[:,[0,parameterIndex]] = parametersConfigurations[:,[parameterIndex,0]]
    parametersList[[0,parameterIndex]] = parametersList[[parameterIndex,0]]
    parametersUnits[[0,parameterIndex]] = parametersUnits[[parameterIndex,0]]
    fileList[[0,parameterIndex]] = fileList[[parameterIndex,0]]

    #Get indices frequency (mandatory parameter !) and dipole distance (optional parameter, default is 0)
    frequencyIndex = np.where(parametersList=="frequency")[0][0]
    try:
        dipoleDistanceIndex = np.where(parametersList=="dipoleDistance")[0][0]
    except IndexError:
        dipoleDistanceIndex = None

    #Get all possible values for the studied parameter
    parameterValues = np.unique(parametersConfigurations[:,0])
    if(len(parameterValues) < 2):
        raise ValueError("[ERROR] Not enough values for the studied parameter")

    #Any fixed values for the studied parameter ?
    if("abscissaValues" in kwargs):
        fixedParametersValues = kwargs["abscissaValues"]
    else:
        fixedParametersValues = list(input("Abscissa parameter values ? (default : *) " + str(np.sort(parameterValues)) + " (" + parametersUnits[0] + ") ").split(' '))

    try:
        fixedParametersValues = [float(item) for item in fixedParametersValues if float(item) in parameterValues]
        if(len(fixedParametersValues) == 0):
            fixedParametersValues = parameterValues
    except ValueError:
        fixedParametersValues = parameterValues

    #Delete useless parameters configurations
    parameterValues = fixedParametersValues
    fixedParameterValuesIndices = [item in fixedParametersValues for item in parametersConfigurations[:,0]]
    parametersConfigurations = parametersConfigurations[fixedParameterValuesIndices]
    fileList = fileList[fixedParameterValuesIndices]

    #Sort the output files according to the studied parameter values
    sortedIndices = np.argsort(parametersConfigurations[:,0])
    parametersConfigurations = parametersConfigurations[sortedIndices]
    fileList = fileList[sortedIndices]

    lastIndex = []
    for value in parameterValues[:-1]:  #The last index is the last one of the list !
        index = np.where(parametersConfigurations[:,0] == value)[0][-1] + 1
        lastIndex.append(index)

    #Split the parameters configurations according to the studied parameter values
    splittedP = np.split(parametersConfigurations,lastIndex) 

    #Get all possible parameters configurations
    interestConfigurations = np.unique(parametersConfigurations[:,1:],axis=0)

    #Select only the parameters configurations which are computed for each studied parameter value
    for configurations in splittedP:
        configurations = configurations[:,1:]
        interestConfigurations = interestConfigurations[(interestConfigurations[:, None] == configurations).all(-1).any(1)] #Returns intersection between interestConfigurations and configurations

    if(len(interestConfigurations) == 0):
        print("[WARNING]: Inconsistent data, missing outputs for all parameters configurations.")
        interestConfigurations = np.unique(parametersConfigurations[:,1:],axis=0)

    #Fixed parameters ? 
    fixedParametersList = []
    fixedParametersValues = []

    if("fixedParameters" in kwargs):
        fixedParametersList = kwargs["fixedParameters"]
        
        if(not isinstance(fixedParametersList, list)):
            fixedParametersList = [fixedParametersList]

        if("fixedValues" in kwargs):
            if(not isinstance(kwargs["fixedValues"], list) and len(fixedParametersList) == 1):
                try:
                    fixedParametersValues = [float(kwargs["fixedValues"])]
                except ValueError:
                    fixedParametersValues = [None]

            elif(isinstance(kwargs["fixedValues"], list) and len(fixedParametersList) == len(kwargs["fixedValues"])):
                fixedParametersValues = []
                for itemList in kwargs["fixedValues"]:
                    try:
                        fixedParametersValues.append([float(item) for item in itemList])
                    except ValueError:
                        fixedParametersValues.append([None])

        else:
            fixedParametersList = []

    variableParametersList = parametersList[1:]
    if(len(fixedParametersList) == 0):
        flag = input("Any fixed parameter ? y/n")

        while(flag == "y"):
            fixedParameter = input("What parameter ? " + str(variableParametersList))

            if(fixedParameter not in variableParametersList):
                continue
            else:
                fixedParametersList.append(fixedParameter)
                fixedParameterIndex = np.where(parametersList[1:] == fixedParameter)[0][0]
                variableParametersList = np.delete(variableParametersList,np.where(variableParametersList == fixedParameter)[0][0])
                fixedValues = list(input("what values ? " + str(np.unique(interestConfigurations[:,fixedParameterIndex])) + " (" + parametersUnits[1:][fixedParameterIndex] + ") ").split(' '))
                fixedValues = [float(item) for item in fixedValues if float(item) in np.unique(interestConfigurations[:,fixedParameterIndex])]
                fixedParametersValues.append(fixedValues)
            
            flag = input("Any fixed parameter ? y/n")

    singleParametersList = []
    singleParametersUnits = []
    multipleParametersList = []
    multipleParametersUnits = []

    for fixedParameter,fixedValue in zip(fixedParametersList,fixedParametersValues):
        print(str(fixedParameter) + " : " + str(fixedValue))

        try:
            fixedParameterIndex = np.where(parametersList[1:] == fixedParameter)[0][0]
        except:
            print("Skipping fixed parameter " + fixedParameter + " : invalid parameter")
            continue

        fixedValueIndex = np.where(np.isin(interestConfigurations[:,fixedParameterIndex],fixedValue))[0]
        if(fixedValue is None):
            fixedValue = np.unique(interestConfigurations[:,fixedParameterIndex])
        elif(len(fixedValueIndex) == 0):
            raise ValueError("[ERROR] Invalid fixed value for parameter " + fixedParameter)
        
        interestConfigurations = interestConfigurations[np.where(np.isin(interestConfigurations[:,fixedParameterIndex],fixedValue))[0]]

        if(len(fixedValue) == 1):
            singleParametersList.append(fixedParameter)
            singleParametersUnits.append(parametersUnits[1:][fixedParameterIndex])
        else:
            multipleParametersList.append(fixedParameter)
            multipleParametersUnits.append(parametersUnits[1:][fixedParameterIndex])

    # Get reference values too !
    # referenceInterestConfigurations = np.empty((0,np.shape(interestConfigurations)[1]))
    # if(parameter != "sigmaPosition" and "iteration" in variableParametersList):
    #     sigmaPositionIndex = np.where(parametersList[1:] == "sigmaPosition")[0][0]
    #     iterationIndex = np.where(parametersList[1:] == "iteration")[0][0]

    #     for interestConfiguration in interestConfigurations:
    #         if(interestConfiguration[sigmaPositionIndex] != 0):
    #             tmpConfiguration = copy.deepcopy(interestConfiguration)
    #             tmpConfiguration[sigmaPositionIndex] = 0
    #             tmpConfiguration[iterationIndex] = 1

    #             if(not (tmpConfiguration == referenceInterestConfigurations).all(axis=1).any()):
    #                 referenceInterestConfigurations = np.vstack((referenceInterestConfigurations,tmpConfiguration))
    
    interestConfigurationsNumber = len(interestConfigurations)

    iterationIndex = None
    iterationNumber = 1
    if("iteration" in variableParametersList):
        iterationIndex = np.where(parametersList[1:] == "iteration")[0][0]
        iterationNumber = len(np.unique(interestConfigurations[:,iterationIndex]))
        interestConfigurations[:,iterationIndex] = np.ones(interestConfigurationsNumber)
        interestConfigurations = np.unique(interestConfigurations,axis=0)
        interestConfigurationsNumber = len(interestConfigurations)

    #Get the scaling factors in case of normalisation
    normalisation = False
    if(parametersUnits[0] == "m" and parameter != "vertices"):
        if("normalisation" in kwargs): 
            if(kwargs["normalisation"]):
                normalisation = True
        else:
            flag = input("Normalise abscissa according to wavelength ? y/n")
            if(flag == "y"):  
                normalisation = True    

    scalingFactors = np.ones(np.shape(interestConfigurations)[0])

    if(normalisation):
        if(parameter == "resolution" or parameter == "size"):
            #scalingFactor = 1/lambda = f/c
            scalingFactors = interestConfigurations[:,frequencyIndex - 1]/c
            #scalingFactor = k = 2*pi*f/c
            scalingFactors *= 2*np.pi
        elif(parameter == "sigmaPosition"):
            #scalingFactor = 1/h
            scalingFactors = 1/interestConfigurations[:,np.where(parametersList=="resolution" - 1)[0][0]]

    #Create the interest configurations / plot values matrix
    if(postProcessingID == "re/im"):
        plotListN = np.zeros((2,iterationNumber,interestConfigurationsNumber,len(parameterValues)))
    else:
        plotListN = np.zeros((1,iterationNumber,interestConfigurationsNumber,len(parameterValues)))
    plotListAbscissa = np.zeros((interestConfigurationsNumber,len(parameterValues)))

    #Relative or absolute error ?
    errorType = "absolute"
    if("errorType" in kwargs and kwargs["errorType"] in ["absolute","relative"]):
        errorType = kwargs["errorType"]
    else:
        flag = input("Relative error ? y/n")
        if(flag == "y"):
            errorType = "relative"

    for interestConfigurationIndex, interestConfiguration in enumerate(interestConfigurations):

        print("Plot : " + str(interestConfigurationIndex+1) + " on " + str(interestConfigurationsNumber))
        
        print("Scaling factor : " + str(scalingFactors[interestConfigurationIndex]))

        for parameterValueIndex, parameterValue in enumerate(parameterValues):
                    
            configuration = np.concatenate(([parameterValue],interestConfiguration))

            if(iterationNumber > 1):
                fileIndices = np.where((np.delete(parametersConfigurations,1 + iterationIndex,1) == np.delete(configuration,1+iterationIndex)).all(axis=1))[0]
            else:
                fileIndices = np.where((parametersConfigurations == configuration).all(axis=1))[0]

            #Get configuration frequency and dipole distance parameters
            f = configuration[frequencyIndex]
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

                points,faces = generateSphericMesh(np.round(configuration[sizeIndex],4),np.round(configuration[0],4),elementType=elementType)
                if(elementType == "P1"):
                    plotListAbscissa[interestConfigurationIndex][parameterValueIndex] = len(points)*scalingFactors[interestConfigurationIndex]
                elif(elementType == "P0"):
                    plotListAbscissa[interestConfigurationIndex][parameterValueIndex] = len(faces)*scalingFactors[interestConfigurationIndex]
            else:
                plotListAbscissa[interestConfigurationIndex][parameterValueIndex] = parameterValue*scalingFactors[interestConfigurationIndex]

            #numericValuesA = np.array(numericValuesA)
            numericValuesN = np.array(numericValuesN)
            analyticalValues = np.array(analyticalValues)

            for l,(analytical, numN) in enumerate(zip(analyticalValues,numericValuesN)):
                if(errorType == "relative"):
                    plotListN[:,l,interestConfigurationIndex,parameterValueIndex] = errorFunction(numN - analytical)/errorFunction(analytical)
                else:
                    plotListN[:,l,interestConfigurationIndex,parameterValueIndex] = errorFunction(numN - analytical)

    #Create plots
    if(postProcessingID == "re/im"):
        figN, axN = plt.subplots(2,figsize=figsize)
    else:
        figN, axN = plt.subplots(1,figsize=figsize)
        axN = [axN]

    title = makeTitle(interestConfigurations,singleParametersList,parametersList[1:],parametersUnits[1:],errorType + " error computed with : \n")

    # for ax in axN:
    #     ax.set_title(title)

    #Log scale ?
    logScale = False
    if("logScale" in kwargs):
        if(kwargs["logScale"]):
            logScale = True
    else:
        flag = input("Log scale ? y/n")
        if(flag):
            logScale = True

    #Scaling function (ordinate)
    scalingFunction = lambda x: x
    if(logScale):
        if(errorType == "absolute" and (postProcessingID == "id" or postProcessingID == "mod")):
            scalingFunction = lambda x: 20*np.log10(x/Pref)
        else:
            scalingFunction = lambda x: np.log10(x)   

    if(not logScale and errorType == "relative"):
        scalingFunction = lambda x: 100*x  

    #Linear regression ?
    linearRegression = False
    if("linearRegression" in kwargs):
        if(kwargs["linearRegression"]):
            linearRegression = True
    else:
        flag = input("Linear regression ? y/n")
        if(flag):
            linearRegression = True

    AlphaMean = []
    AlphaMax = []

    for i,configuration in enumerate(interestConfigurations):

        legend = makeLegend(configuration,multipleParametersList,parametersList[1:],parametersUnits[1:],"")

        for (axNi,plotN) in zip(axN,plotListN):

            if(iterationNumber > 1):
                minSpace, maxSpace = np.ones(len(plotN[0,i]))*10**10, -np.ones(len(plotN[0,i]))*10**10
                for l,subPlotN in enumerate(plotN[:,i]):
                    if(l==0):
                        axNi.plot(plotListAbscissa[i],scalingFunction(subPlotN),label=legend,color=cmap(i),marker=markers[i],linestyle='None',markerfacecolor='None',markersize=7,markeredgewidth=2)
                    else:
                        axNi.plot(plotListAbscissa[i],scalingFunction(subPlotN),color=cmap(i),marker=markers[i],linestyle='None',markerfacecolor='None',markersize=7,markeredgewidth=2)
                    maxSpace = np.maximum(maxSpace,scalingFunction(subPlotN))
                    minSpace = np.minimum(minSpace,scalingFunction(subPlotN))
                axNi.fill_between(plotListAbscissa[i],minSpace,maxSpace,color=cmap(i),alpha=0.15)
                plotMaxN = np.max(scalingFunction(plotN[:,i]),axis=0)
                plotMeanN = np.mean(scalingFunction(plotN[:,i]),axis=0)

                if(linearRegression):
                    #Maximum error
                    M = np.vstack((scalingFunction(plotListAbscissa[i]),np.ones(len(parameterValues)))).T
                    VN = np.dot(np.linalg.pinv(M),plotMaxN)

                    axNi.plot(plotListAbscissa[i],VN[0]*scalingFunction(plotListAbscissa[i])+VN[1],label="slope = " + str(np.round(VN[0],3)),color=cmap(i),linestyle="dashed",linewidth=2)
                    AlphaMax.append(VN[0])

                    #Mean error
                    M = np.vstack((scalingFunction(plotListAbscissa[i]),np.ones(len(parameterValues)))).T
                    VN = np.dot(np.linalg.pinv(M),plotMeanN)

                    axNi.plot(plotListAbscissa[i],VN[0]*scalingFunction(plotListAbscissa[i])+VN[1],label="slope = " + str(np.round(VN[0],3)),color=cmap(i),linestyle="dotted",linewidth=2)
                    AlphaMean.append(VN[0])
                    
                elif("drawSlope" in kwargs and kwargs["drawSlope"] and "slopeValue" in kwargs):
                    #Maximum error
                    verticalOffset = np.mean(plotMaxN - kwargs["slopeValue"]*scalingFunction(plotListAbscissa[i]))
                    axNi.plot(plotListAbscissa[i],verticalOffset + kwargs["slopeValue"]*scalingFunction(plotListAbscissa[i]),color=cmap(i),linestyle='dashed',linewidth=2)

                    #Mean error
                    verticalOffset = np.mean(plotMeanN - kwargs["slopeValue"]*scalingFunction(plotListAbscissa[i]))
                    axNi.plot(plotListAbscissa[i],verticalOffset + kwargs["slopeValue"]*scalingFunction(plotListAbscissa[i]),color=cmap(i),linestyle='dotted',linewidth=2)
                    
                else:
                    #Maximum error
                    axNi.plot(plotListAbscissa[i],plotMaxN,color=cmap(i),linestyle='dashed',linewidth=2)
                    #Mean error
                    axNi.plot(plotListAbscissa[i],plotMeanN,color=cmap(i),linestyle='dotted',linewidth=2)

            else:
                plotN = plotN[0]    #Only one iteration !
                axNi.plot(plotListAbscissa[i],scalingFunction(plotN[i]),label=legend,color=cmap(i),marker=markers[i],linestyle='None',markerfacecolor='None',markersize=10,markeredgewidth=3)

                if(linearRegression):
                    M = np.vstack((scalingFunction(plotListAbscissa[i]),np.ones(len(parameterValues)))).T
                    VN = np.dot(np.linalg.pinv(M),scalingFunction(plotN[i]))

                    axNi.plot(plotListAbscissa[i],VN[0]*scalingFunction(plotListAbscissa[i])+VN[1],label="slope = " + str(np.round(VN[0],3)),color=cmap(i),linestyle="dashed",linewidth=2)

                    AlphaMean.append(VN[0])
                    AlphaMax.append(VN[0])
                    
                elif("drawSlope" in kwargs and kwargs["drawSlope"] and "slopeValue" in kwargs):
                    verticalOffset = np.mean(scalingFunction(plotN[i]) - kwargs["slopeValue"]*scalingFunction(plotListAbscissa[i]))
                    axNi.plot(plotListAbscissa[i],verticalOffset + kwargs["slopeValue"]*scalingFunction(plotListAbscissa[i]),color=cmap(i),linestyle='dashed',linewidth=2)

    extraPadding = False
    if(parameter == "sigmaPosition" and "robot" in kwargs and kwargs["robot"]):
        if((scalingFactors == scalingFactors[0]).all()):
            extraPadding = True
            axNi.axvline(x=scalingFactors[0]*1.63*10**-3,color="dimgray",linestyle="dashed",linewidth=2)    #2.03
            t = axNi.text(scalingFactors[0]*1.63*10**-3, 1.0025, 'calibrated robot', color='dimgray', transform=axNi.get_xaxis_transform(), ha='center', va='bottom',fontsize = 20)

            axNi.axvline(x=scalingFactors[0]*7.66*10**-3,color="dimgray",linestyle="dashed",linewidth=2)    #14.4
            t = axNi.text(scalingFactors[0]*7.66*10**-3, 1.0025, 'uncalibrated robot', color='dimgray', transform=axNi.get_xaxis_transform(), ha='center', va='bottom', fontsize = 20)

    for axNi in axN:
        if(normalisation):
            if(parameter == "resolution"):
                axNi.set_xlabel(r"$hk$")
            elif(parameter == "size"):
                axNi.set_xlabel(r"$Dk$")
            elif(parameter == "sigmaPosition"):
                axNi.set_xlabel(r"$\frac{\sigma_P}{h}$")
            else:
                axNi.set_xlabel(parameter + "/" + r"$\lambda$")
        elif(parameter == "sigmaPosition"):
            axNi.set_xlabel(r"$\sigma_P$ (m)")
        elif(parameter == "resolution"):
            axNi.set_xlabel(r"$h$ (m)")
        elif(parameter == "vertices"):
            axNi.set_xlabel("Vertices number")
        else:
            axNi.set_xlabel(parameter + " (" + parametersUnits[0] + ")")

        if(logScale):
            axNi.set_xscale('log')
            if(errorType == "absolute" and (postProcessingID == "id" or postProcessingID == "mod")):
                axNi.set_ylabel("average " + errorType + " error (dB)")
            else:
                axNi.set_ylabel(r"$\log(\epsilon)$",labelpad=15)
              
        else:
            if(errorType == "relative"):
                tmpUnit = r"(\%)"
            else:
                if(postProcessingID == "phase"):
                    tmpUnit = "(rad)"
                else:
                    tmpUnit = "(Pa)"
            axNi.set_ylabel("average " + errorType + " error " + tmpUnit)
    
        handles, labels = axNi.get_legend_handles_labels()
        ncol = 3
        spacing = 1.0

        if(len(handles)%2 == 0):
            if(len(handles)%3 != 0):
                ncol = 2
            handles = list(flip(handles, ncol))
            labels = list(flip(labels, ncol))

        if(len(handles) != 0 or len(labels) != 0):
            figN.tight_layout()
            axNi.set_position([0.165,0.125,0.8,0.85])
            if(extraPadding):
                axNi.legend(handles, labels, bbox_to_anchor=(0.5,1.0), loc='lower center', ncol=ncol, borderaxespad=1.1, reverse=False, columnspacing=spacing, fontsize=20)
            else:
                axNi.legend(handles, labels, bbox_to_anchor=(0.5,1.0), loc='lower center', ncol=ncol, borderaxespad=0.25, reverse=False, columnspacing=spacing, fontsize=20)
    
    axNi.grid(which="major")
    axNi.grid(linestyle = '--',which="minor")
    axNi.yaxis.set_major_locator(MaxNLocator(10))
    axNi.yaxis.set_minor_locator(MaxNLocator(10))
    axNi.tick_params(axis='both', which='major', pad=7)

    if(figureName is not None):
        figN.savefig(figureName, dpi = 300, bbox_inches = 'tight')

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

    kwargs["verticesNumber"] = False
    kwargs["linearRegression"] = False
    kwargs["errorType"] = "relative"
    kwargs["logScale"] = True
                
    ### Resolution 
    kwargs["abscissaParameter"] = "resolution"
    kwargs["fixedParameters"] = ["size","frequency","sigmaPosition","sigmaMeasure","dipoleDistance"]
    kwargs["fixedValues"] = [[0.5],[500,1000,5000],[0.0],[0.0],[0.45]]
    
    ### Resolution error normalized
    kwargs["abscissaValues"] = [0.005,0.0125,0.025,0.0375,0.05,0.0625,0.125,0.25]
    kwargs["normalisation"] = True
    kwargs["drawSlope"] = True
    kwargs["slopeValue"] = 2.0
    plotError(postProcessing,analytical,error,figureName="NormalizedResolutionError.pdf",**kwargs)

    ### Resolution error unnormalized
    kwargs["abscissaValues"] = [0.005,0.0125,0.025,0.0375,0.05,0.0625,0.125,0.25]
    kwargs["normalisation"] = False
    kwargs["drawSlope"] = True
    kwargs["slopeValue"] = 2.0
    plotError(postProcessing,analytical,error,figureName="ResolutionError.pdf",**kwargs)

    ### Size error normalized
    kwargs["abscissaParameter"] = "size"
    kwargs["abscissaValues"] = "*"
    kwargs["fixedParameters"] = ["resolution","frequency","sigmaPosition","sigmaMeasure","dipoleDistance"]
    kwargs["fixedValues"] = [[0.005],[500,1000,5000],[0.0],[0.0],[0.05]]
    kwargs["normalisation"] = True
    kwargs["drawSlope"] = False
    plotError(postProcessing,analytical,error,figureName="NormalizedSizeError.pdf",**kwargs)
    
    ### Sigma error normalized
    kwargs["abscissaParameter"] = "sigmaPosition"
    kwargs["abscissaValues"] = [0.0005,  0.00125, 0.0025,  0.00375, 0.005, 0.0125]
    kwargs["fixedParameters"] = ["size","resolution","frequency","sigmaMeasure","dipoleDistance"]
    kwargs["fixedValues"] = [[0.5],[0.05],[500,1000,5000],[0.0],[0.45]]
    kwargs["normalisation"] = False
    kwargs["linearRegression"] = False
    kwargs["robot"] = True
    plotError(postProcessing,analytical,error,figureName="NormalizedSigmaError.pdf",**kwargs)
    kwargs["robot"] = False

    ### When sigmaP << h, we fall back on the unnoised error estimate
    ## We took the smallest sigmaP with all resolution, knowing there is a ten fold factor between the smallest resolution and sigmaP    
    kwargs["abscissaParameter"] = "resolution"
    kwargs["abscissaValues"] = [0.005, 0.0125, 0.025,  0.0375, 0.05]
    kwargs["fixedParameters"] = ["size","resolution","frequency","sigmaMeasure","sigmaPosition","dipoleDistance"]
    kwargs["fixedValues"] = [[0.5],[0.005],[500,1000,5000],[0.0],[0.0005],[0.45]]
    kwargs["normalisation"] = False
    kwargs["drawSlope"] = True
    kwargs["slopeValue"] = 2.0
    plotError(postProcessing,analytical,error,figureName="ResolutionErrorSigma<<h.pdf",**kwargs)

    ### When sigmaP >> 1 (and I guess h is not to big), we wish to find a sqrt(h) tendancy
    ### We took the highest sigmaP value for all resolutions, but the largest sigmaP is still smaller than the largest resolution (#TODO New computation with sigmaP = 0.05/0.1/0.5 ?)
    kwargs["abscissaParameter"] = "resolution"
    kwargs["abscissaValues"] = [0.0125, 0.025,  0.0375, 0.05]
    kwargs["fixedParameters"] = ["size","resolution","frequency","sigmaMeasure","sigmaPosition","dipoleDistance"]
    kwargs["fixedValues"] = [[0.5],[0.05],[500,1000,5000],[0.0],[0.0125],[0.45]]
    kwargs["normalisation"] = False
    kwargs["drawSlope"] = True
    kwargs["slopeValue"] = 0.5
    plotError(postProcessing,analytical,error,figureName="SigmaErrorSigma>>h.pdf",**kwargs)

    ### When sigmaP ~ h, the linear sigmaP dependancy arises
    ### We took the smallest resolution, whith the 4 highest values of sigmaP, knowing that the highest sigmaP value is equal to the resolution (in practice only the 0.0005 sigmaP is removed)
    kwargs["abscissaParameter"] = "sigmaPosition"
    kwargs["abscissaValues"] = [0.0005, 0.00125, 0.0025, 0.00375, 0.005]
    kwargs["fixedParameters"] = ["size","resolution","frequency","sigmaMeasure","sigmaPosition","dipoleDistance"]
    kwargs["fixedValues"] = [[0.5],[0.005],[500,1000,5000],[0.0],[0.005],[0.45]]
    kwargs["normalisation"] = False
    kwargs["drawSlope"] = True
    kwargs["slopeValue"] = 1.0
    plotError(postProcessing,analytical,error,figureName="SigmaErrorSigma~h.pdf",**kwargs)
    
    ### Resolution error unnormalized with sigma => Slope study
    kwargs["abscissaParameter"] = "resolution"
    kwargs["abscissaValues"] = [0.005,  0.0125, 0.025,  0.0375, 0.05]
    kwargs["fixedParameters"] = ["size","frequency","sigmaPosition","sigmaMeasure","dipoleDistance"]
    kwargs["fixedValues"] = [[0.5],[500,1000,5000],[0.0],[0.0],[0.45]]
    kwargs["normalisation"] = False
    kwargs["drawSlope"] = False
    kwargs["linearRegression"] = True

    sigmaPositions = [0.0, 0.0005,  0.00125, 0.0025,  0.00375, 0.005]
    AlphaMean = []
    AlphaMax = []

    for sigmaP in sigmaPositions :
        kwargs["fixedValues"][2] = [sigmaP]
        amean, amax = plotError(postProcessing,analytical,error,None,**kwargs)
        AlphaMean.append(amean)
        AlphaMax.append(amax)

    AlphaMean = np.array(AlphaMean)
    AlphaMax = np.array(AlphaMax)
    print(AlphaMean)
    print(AlphaMax)
    plt.close("all")
    
    figN, axN = plt.subplots(1,figsize=figsize)
    Labels = [r"$\angle_{max}$",r"$\angle_{avg}$"]
    for i,f in enumerate(kwargs["fixedValues"][1]):
        for j,list in enumerate([AlphaMax,AlphaMean]):
            axN.plot(sigmaPositions,list[:,i],color=cmap2(4*i+j),marker=markers[i],linestyle='None',markerfacecolor='None',markersize=10,markeredgewidth=3,label=Labels[j] + " - f = " + str(f) + " Hz")

    axN.axhline(y=0.5,color="dimgray",linestyle='dashed',linewidth=1.5)
    t = axN.text(1.02, 0.5, r"$\sqrt{h}$", color='dimgray', transform=axN.get_yaxis_transform(), ha='left', va='center',fontsize = 20)
    axN.axhline(y=2,color="dimgray",linestyle='dashed',linewidth=1.5)
    t = axN.text(1.02, 2, r"$h^2$", color='dimgray', transform=axN.get_yaxis_transform(), ha='left', va='center',fontsize = 20)
    axN.axhline(y=1,color="dimgray",linestyle='dashed',linewidth=1.5)
    t = axN.text(1.02, 1, r"$h$", color='dimgray', transform=axN.get_yaxis_transform(), ha='left', va='center',fontsize = 20)

    figN.tight_layout()
    axN.set_position([0.165,0.125,0.8,0.85])

    axN.legend(bbox_to_anchor=(0.5,1.0), loc='lower center', ncol=2, borderaxespad=0.25, reverse=False,columnspacing=1.0, fontsize=20)
    axN.set_ylabel(r"Error slope",labelpad=15)
    axN.set_xlabel(r"$\sigma_P$ (m)")
    axN.grid(which="major")
    axN.grid(linestyle = '--',which="minor")
    axN.yaxis.set_major_locator(MaxNLocator(10))
    axN.yaxis.set_minor_locator(MaxNLocator(10))
    axN.tick_params(axis='both', which='major', pad=7)

    figN.savefig("SigmaSlope.pdf", dpi = 300, bbox_inches = 'tight')