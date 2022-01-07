import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sp
import csv

import glob
import os
import sys

postProcessingFunctions = {}
postProcessingFunctions["db"] = lambda Re,Im : 20*np.log10(np.abs(np.complex(Re,Im))/20e-6)
postProcessingFunctions["abs"] = lambda Re,Im : np.abs(np.complex(Re,Im))
postProcessingFunctions["id"] = lambda Re,Im : np.complex(Re,Im)

def plotSphericCut(postProcessingFunction):

    #Get all output files
    fileList = np.array(glob.glob("output*"))
    parametersList = []

    #Get all parameters names, or define them
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

    #Create files/parameters matrix
    P = np.empty((len(fileList),len(parametersList)))

    for i,file in enumerate(fileList):
        P[i] = np.array([float(l) for l in os.path.splitext(file)[0].split("_")[1:]])

    subP = np.copy(P)
    file = "output"
    configuration = []

    for i,parameter in enumerate(parametersList):
        if(len(np.unique(subP[:,0])) == 1):
            value = str(subP[0,0]) if subP[0,0] != int(subP[0,0]) else str(int(subP[0,0]))
            print("Only possible for " + parameter +" is " + value +  " (" + parametersUnits[i] + ")")
        else:
            value = input("What value for " + parameter + " ? " + str(np.unique(subP[:,0])) + " (" + parametersUnits[i] + ")")
            value = value if float(value) != int(float(value)) else str(int(float(value)))

        configuration.append(value)
        file += "_" + value
        subP = subP[np.argwhere(subP[:,0] == float(value))[:,0]][:,1:]
    
    file += ".txt"

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
    
            AmpSomme.append(postProcessingFunction(float(row[3]),float(row[4])))
            AmpAnalytique.append(postProcessingFunction(float(row[5]),float(row[6])))

    #X0 = np.argmax(np.abs(np.array(X)))

    AmpSomme = np.array(AmpSomme)
    AmpAnalytique = np.array(AmpAnalytique)

    #AmpAnalytique = np.roll(AmpAnalytique,X0)
    AmpAnalytique = np.append(AmpAnalytique,AmpAnalytique[0])
    #AmpSomme = np.roll(AmpSomme,X0)
    AmpSomme = np.append(AmpSomme,AmpSomme[0])

    #Removing infinite values
    IndexInf = np.argwhere(np.isinf(AmpAnalytique))
    AmpAnalytique = np.delete(AmpAnalytique,IndexInf)
    AmpSomme = np.delete(AmpSomme,IndexInf)

    TH = [0]
    counterTh = 0

    for i in range(len(X)):
        u = np.array([X[i],Y[i],Z[i]])
        try:
            v = np.array([X[i+1],Y[i+1],Z[i+1]])
        except:
            v = np.array([X[0],Y[0],Z[0]])
        counterTh += np.arccos(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)))
        TH.append(counterTh)

    TH = np.delete(TH,IndexInf)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(TH,AmpAnalytique,label="Analytical solution (dB)")
    ax.plot(TH,AmpSomme,label="Numerical solution (dB)")

    print("Error = " + str(np.sqrt(np.average((AmpSomme - AmpAnalytique)**2))))

    label = "Acoustic pressure field computed for : \n"
    for j,name in enumerate(parametersList):
        label += name + " = " + configuration[j] + " (" + parametersUnits[j] + ") "
    label = label[:-1]

    ax.set_title(label)

    maxAmp = max(max(AmpSomme),max(AmpAnalytique))*1.1

    ax.annotate('x', xy=(np.pi/40,maxAmp), xycoords='data', annotation_clip=False, size = 12)
    ax.annotate('y', xy=(np.pi/2 - np.pi/40,maxAmp), xycoords='data', annotation_clip=False, size = 12)

    plt.legend()
    plt.show()


### MAIN ###

functionID = input("Post processing function ? " + str(list((postProcessingFunctions.keys()))))
plotSphericCut(postProcessingFunctions[functionID])