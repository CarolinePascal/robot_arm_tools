#!/usr/bin/python3

#System packages
import glob
import os

#Utility packages
import numpy as np

#Plot packages
from robot_arm_acoustic_post_processing.PlotTools import *

#Post-processing functions
postProcessingFunctions = {}
postProcessingFunctions["re/im"] = lambda X : np.array([np.real(X),np.imag(X)])
postProcessingFunctions["mod"] = lambda X : np.abs(X)
postProcessingFunctions["phase"] = lambda X : np.angle(X)
postProcessingFunctions["id"] = lambda X : X

#Error functions
errorFunctions = {}
errorFunctions["l2"] = lambda X : np.sqrt(np.sum(np.abs(X.T)**2,axis=0))
errorFunctions["linf"] = lambda X : np.max(np.abs(X.T),axis=0)
errorFunctions["RMS"] = lambda X : np.sqrt(np.mean(np.abs(X.T)**2,axis=0))
errorFunctions["mean"] = lambda X : np.mean(X.T,axis=0)

## Function returning all the available parameters configurations in an output files folder
#  @return parametersConfigurations     Output files / parameters configurations matrix
#  @return parametersList               List of the parameters names
#  @return parametersUnits              List of the parameters units
#  @return fileLits                     List of the output files names
def getParametersConfigurations():

    #Get all output files
    fileList = np.array(glob.glob("output*"))
    fileList = np.append(fileList,np.array(glob.glob("position_noise/output*")))
    #fileList = np.append(fileList,np.array(glob.glob("measurement_noise/output*")))
    
    #Get parameters names and units, or define them if none given
    parametersList = []
    parametersUnits = []

    try:
        with open("parameters.txt") as parametersFile:
            lines = parametersFile.read().splitlines()
            for line in lines:
                try:
                    name,unit = line.split(' ')
                except ValueError:
                    name = line
                    unit=""
                parametersList.append(name)
                parametersUnits.append(unit)
    except FileNotFoundError:
        for i in range(len(fileList[0].split("_")) - 1):
            parametersList.append("parameter"+str(i+1))
            parametersUnits.append("")

    parametersList = np.array(parametersList)
    parametersUnits = np.array(parametersUnits)

    #Create the output files / parameters configurations matrix
    P = np.empty((len(fileList),len(parametersList)))

    for i,file in enumerate(fileList):
        P[i] = np.array([float(l) for l in os.path.splitext(os.path.basename(file))[0].split("_")[1:]])

    return(P,parametersList,parametersUnits,fileList)

## Function creating a legend from a configuration
#  @param configuration            Configuration to be converted into a legend
#  @param multipleParametersList   List of the parameters names with multiple values
#  @param parametersList           List of the parameters names
#  @param parametersUnits          List of the parameters units
#  @param prefix                   Initial prefix of the legend
def makeLegend(configuration, multipleParametersList, parametersList, parametersUnits, prefix):
    legend = prefix

    for name in multipleParametersList:
        if(name == "iteration"):
            continue

        try:
            j = np.where(parametersList == name)[0][0]
        except IndexError:
            continue

        if(name == "frequency"):
            plotName = r"$f$"
        elif(name == "size"):
            plotName = r"$D$"
        elif(name == "resolution"):
            plotName = r"$h$"
        elif(name == "sigmaMeasure"):
            plotName = r"$\sigma_{M}$"
        elif(name == "sigmaPosition"):
            plotName = r"$\sigma_{P}$"
        elif(name == "dipoleDistance"):
            plotName = r"$d$"
        else:
            plotName = name

        legend += plotName + " = " + str(configuration[j]) + " " + parametersUnits[j] + " - "

    if(len(legend) != len(prefix)):
        legend = legend[:-3]

    return(legend)

## Function creating a title from a configurations list
#  @param configurations           List of configurations to be converted into a legend
#  @param singleParametersList     List of the parameters names with single values
#  @param parametersList           List of the parameters names
#  @param parametersUnits          List of the parameters units
#  @param prefix                   Initial prefix of the legend
def makeTitle(configurations, singleParametersList, parametersList, parametersUnits, prefix):
    title = prefix  #Default value (noTitle = True)
    titleCounter = 0

    for name in singleParametersList:

        if(name == "iteration"):
            continue
    
        try:
            j = np.where(parametersList == name)[0][0]
        except IndexError:
            continue

        if(name == "frequency"):
            plotName = r"$f$"
        elif(name == "size"):
            plotName = r"$D$"
        elif(name == "resolution"):
            plotName = r"$h$"
        elif(name == "sigmaMeasure"):
            plotName = r"$\sigma_{M}$"
        elif(name == "sigmaPosition"):
            plotName = r"$\sigma_{P}$"
        elif(name == "dipoleDistance"):
            plotName = r"$d$"
        else:
            plotName = name

        if(titleCounter < 3):
            title += plotName + " = " + str(configurations[0,j]) + " " + parametersUnits[j] + " - "
            titleCounter += 1
        else:
            title = title[:-3]
            title += "\n" + plotName + " = " + str(configurations[0,j]) + " " + parametersUnits[j] + " - "
            titleCounter = 0

    if(len(title) != len(prefix)):        
        title = title[:-3]

    return(title)