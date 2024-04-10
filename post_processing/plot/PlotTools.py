#!/usr/bin/python3

#System packages
import glob
import os

#Utility packages
import numpy as np

#Fancy plot parameter
import matplotlib.pyplot as plt

plt.rc('font', **{'size': 24, 'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
cmap = plt.get_cmap("tab10")
cmap2 = plt.get_cmap("tab20c")
markers = ["^","s","o","v","D","x","*","+"]

def log_formatter(x,pos):
    sci = "{:.0e}".format(x)
    sci = [int(item) for item in sci.split("e")]
    if(sci[0] == 5):
        return(r"$5\times10^{{{exponent}}}$".format(exponent=sci[1]))
    else:   
        return("")

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
#  @return P Output files / parameters configurations matrix
#  @return parametersList List of the parameters names
#  @return parametersUnits List of the parameters units
#  @return fileLits List of the output files names
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
        P[i] = np.array([float(l) for l in os.path.splitext(file)[0].split("_")[1:]])

    return(P,parametersList,parametersUnits,fileList)