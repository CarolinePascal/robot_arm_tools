import matplotlib.pyplot as plt
import numpy as np
import csv

import glob
import os
import sys

from copy import deepcopy

c = 343.4

fileList = np.array(glob.glob("output*"))
parametersList = []

try:
    with open("parameters.txt") as parametersFile:
        parametersList = parametersFile.read().splitlines()
except:
    for i in range(len(fileList[0].split("_")) - 1):
        parametersList.append("parameter"+str(i+1))

parametersList = np.array(parametersList)

P = np.empty((len(fileList),len(parametersList)))

for i,file in enumerate(fileList):
    P[i] = np.array([float(l) for l in os.path.splitext(file)[0].split("_")[1:]])

parameterIndex = np.argwhere(parametersList == sys.argv[1])[0][0]

print(P)
tempColumn = deepcopy(P[:,0])
P[:,0] = P[:,parameterIndex]
P[:,parameterIndex] = tempColumn
print(P)

print(parametersList)
tempName = parametersList[0]
parametersList[0] = parametersList[parameterIndex]
parametersList[parameterIndex] = tempName
print(parametersList)

sortedIndices = P[:, 0].argsort()
P = P[sortedIndices]
fileList = fileList[sortedIndices]

print(P)
print(fileList)

parameterValues = np.unique(P[:,0])

print(parameterValues)

configurationsNumber = len(P)//len(parameterValues)
configurationsList = P[:configurationsNumber,1:]

print(configurationsList)

plotList = np.empty((configurationsNumber,len(parameterValues)))

for i,file in enumerate(fileList):
    configurationIndex = np.where((P[i,1:]==configurationsList).all(axis=1))[0][0]

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
    
            AmpSomme.append(20*np.log10(np.abs(np.complex(float(row[3]),float(row[4])))/20e-6))
            AmpAnalytique.append(20*np.log10(np.abs(np.complex(float(row[5]),float(row[6])))/20e-6))

    #X0 = np.argmax(np.abs(np.array(X)))

    AmpSomme = np.array(AmpSomme)
    AmpAnalytique = np.array(AmpAnalytique)

    #AmpAnalytique = np.roll(AmpAnalytique,X0)
    AmpAnalytique = np.append(AmpAnalytique,AmpAnalytique[0])
    #AmpSomme = np.roll(AmpSomme,X0)
    AmpSomme = np.append(AmpSomme,AmpSomme[0])

    plotList[configurationIndex][i//configurationsNumber] = np.average(np.abs(AmpSomme-AmpAnalytique))

    #TH = np.arange(0,2*np.pi,2*np.pi/len(X))
    #TH = np.append(TH,2*np.pi)
#
    #fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    #ax.plot(TH,AmpAnalytique,label="Analytical solution (dB)")
    #ax.plot(TH,AmpSomme,label="Numerical solution (dB)")
#
    #label = "Acoustic pressure field computed for \n"
    #for j,name in enumerate(parametersList[1:]):
    #    if(len(np.unique(configurationsList[:,j])) > 1):
    #        label += name + " = " + str(P[i,j+1]) + "\n"
    #label = label[:-1]
#
    #ax.set_title(label, x=-0.25, y=0.25)
#
    #maxAmp = max(max(AmpSomme),max(AmpAnalytique))*1.1
#
    #ax.annotate('x', xy=(np.pi/40,maxAmp), xycoords='data', annotation_clip=False, size = 12)
    #ax.annotate('y', xy=(np.pi/2 - np.pi/40,maxAmp), xycoords='data', annotation_clip=False, size = 12)
#
    #plt.legend()
    #plt.show()

figLog, axLog = plt.subplots()
figLin, axLin = plt.subplots()

for i,configuration in enumerate(configurationsList):
    label = ""
    for j,name in enumerate(parametersList[1:]):
        if(len(np.unique(configurationsList[:,j])) > 1):
            label += name + " = " + str(P[i,j+1]) + "\n"
    label = label[:-1]
    axLin.plot(parameterValues,plotList[i],label=label)
    axLog.plot(parameterValues,np.log10(plotList[i]),label=label)

axLog.set_xlabel(sys.argv[1])
axLog.set_ylabel("log(Average error)")
axLog.set_xscale('log')  
axLog.set_title("log(Average error)")    

axLog.legend()

axLin.set_xlabel(sys.argv[1])
axLin.set_ylabel("Average error")
axLin.set_title("Average error")   

axLin.legend()

plt.show()
