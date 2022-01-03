import matplotlib.pyplot as plt
import numpy as np
import csv

import glob
import os
import sys

from functools import reduce

postProcessingFunctions = {}
postProcessingFunctions["db"] = lambda Re,Im : 20*np.log10(np.abs(np.complex(Re,Im))/20e-6)
postProcessingFunctions["abs"] = lambda Re,Im : np.abs(np.complex(Re,Im))
postProcessingFunctions["id"] = lambda Re,Im : np.complex(Re,Im)

def plotError(postProcessingError = "dB"):

    #Get all output files
    fileList = np.array(glob.glob("output*"))
    parametersList = []

    #Get all parameters names, or define them
    try:
        with open("parameters.txt") as parametersFile:
            parametersList = parametersFile.read().splitlines()
    except:
        for i in range(len(fileList[0].split("_")) - 1):
            parametersList.append("parameter"+str(i+1))

    parametersList = np.array(parametersList)

    #Create files/parameters matrix
    P = np.empty((len(fileList),len(parametersList)))

    for i,file in enumerate(fileList):
        P[i] = np.array([float(l) for l in os.path.splitext(file)[0].split("_")[1:]])

    #Put the studied parameter in the first column
    try:
        parameter = sys.argv[1]
    except:
        parameter = input("What parameter ? " + str(parametersList))

    parameterIndex = np.argwhere(parametersList == parameter)[0][0]
    P[:,[0,parameterIndex]] = P[:,[parameterIndex,0]]
    parametersList[[0,parameterIndex]] = parametersList[[parameterIndex,0]]

    #Sort files according to the studied parameter values
    sortedIndices = P[:, 0].argsort()
    P = P[sortedIndices]
    fileList = fileList[sortedIndices]

    #Get studied parameter values
    parameterValues = np.unique(P[:,0])
    lastIndex = []
    for value in parameterValues:
        lastIndex.append(np.argwhere(P[:,0] == value)[-1][0] + 1)

    splittedP = np.split(P,lastIndex)[:-1]

    interestConfigurations = np.unique(P[:,1:],axis=0)

    for configurations in splittedP:
        configurations = configurations[:,1:]
        interestConfigurations = interestConfigurations[(interestConfigurations == configurations[:, None]).any(axis=0).all(axis=-1)]

    plotList = np.empty((len(interestConfigurations),len(parameterValues)))

    for i,file in enumerate(fileList):
        try:
            configurationIndex = np.argwhere((P[i,1:]==interestConfigurations).all(axis=1))[0][0]
        except:
            continue

        parameterValueIndex = np.argwhere(P[i,0] == parameterValues)[0][0]

        X = []
        Y = []
        Z = []
        AmpSomme = []
        AmpAnalytique = []

        with open(file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar='|')

            for row in reader:
                X.append(float(row[0]))
                Y.append(float(row[1]))
                Z.append(float(row[2]))
        
                AmpSomme.append(postProcessingError(float(row[3]),float(row[4])))
                AmpAnalytique.append(postProcessingError(float(row[5]),float(row[6])))

        #X0 = np.argmax(np.abs(np.array(X)))

        AmpSomme = np.array(AmpSomme)
        AmpAnalytique = np.array(AmpAnalytique)

        #AmpAnalytique = np.roll(AmpAnalytique,X0)
        AmpAnalytique = np.append(AmpAnalytique,AmpAnalytique[0])
        #AmpSomme = np.roll(AmpSomme,X0)
        AmpSomme = np.append(AmpSomme,AmpSomme[0])

        plotList[configurationIndex][parameterValueIndex] = np.average(np.abs(AmpSomme-AmpAnalytique))

    figLog, axLog = plt.subplots()
    figLin, axLin = plt.subplots()

    cmap = plt.cm.get_cmap('gist_rainbow', len(interestConfigurations))

    for i,configuration in enumerate(interestConfigurations):
        label = ""
        for j,name in enumerate(parametersList[1:]):
            if(len(np.unique(interestConfigurations[:,j])) > 1):
                label += name + " = " + str(configuration[j]) + "\n"
        label = label[:-1]
        axLin.plot(parameterValues,plotList[i],label=label,color=cmap(i))
        axLog.plot(parameterValues,np.log10(plotList[i]),label=label,color=cmap(i))

    axLog.set_xlabel(parameter)
    axLog.set_ylabel("log(Average error)")
    axLog.set_xscale('log')  
    axLog.set_title("log(Average error)")    

    axLog.legend()

    axLin.set_xlabel(parameter)
    axLin.set_ylabel("Average error")
    axLin.set_title("Average error")   

    axLin.legend()

    plt.show()


### MAIN ###

functionID = input("Post processing function ? " + str(list((postProcessingFunctions.keys()))))
plotError(postProcessingFunctions[functionID])