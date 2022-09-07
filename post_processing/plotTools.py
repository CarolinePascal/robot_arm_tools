import numpy as np
import glob
import csv
import os

from measurements.Tools import *

#Post-processing functions
postProcessingFunctions = {}
postProcessingFunctions["db"] = lambda X : 20*np.log10(np.abs(X)/20e-6)
postProcessingFunctions["mod"] = lambda X : np.abs(X)
postProcessingFunctions["phase"] = lambda X : modulo(np.angle(X))
postProcessingFunctions["id"] = lambda X : X

def getParametersConfigurations():

    #Get all output files
    fileList = np.array(glob.glob("output*"))
    
    #Get parameters names and units, or define them if none given
    parametersList = []
    parametersUnits = []

    try:
        with open("parameters.txt") as parametersFile:
            lines = parametersFile.read().splitlines()
            for line in lines:
                try:
                    name,unit = line.split(' ')
                except:
                    name = line
                    unit=""
                parametersList.append(name)
                parametersUnits.append(unit)
    except:
        for i in range(len(fileList[0].split("_")) - 1):
            parametersList.append("parameter"+str(i+1))
            parametersUnits.append("")

    parametersList = np.array(parametersList)
    parametersUnits = np.array(parametersUnits)

    #Create the output files / parameters configurations matrix
    P = np.empty((len(fileList),len(parametersList)))

    for i,file in enumerate(fileList):
        P[i] = np.array([float(l) for l in os.path.splitext(file)[0].split("_")[1:]])

    return(P,parametersList,parametersUnits,fileList)