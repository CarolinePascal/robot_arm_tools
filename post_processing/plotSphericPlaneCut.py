import matplotlib.pyplot as plt
import numpy as np
import csv

import glob

c = 343.4

directoryList = np.array(glob.glob("outputs*"))

F = []
D = []

for directory in directoryList:
    L = directory.split("_")
    F.append(float(L[1]))
    D.append(float(L[2]))
    
If = np.argsort(F)
F = np.sort(F)

for j,directory in enumerate(directoryList[If]):
    kd = D[j]*2*np.pi*F[j]/c

    fileList = np.array(glob.glob(directory + "/data_output*.txt"))
    V = []
    E = []

    for file in fileList:
        L = file.split('.')[:-1]
        file = ""
        for item in L:
            file = file+item
        V.append(int(file.split('.')[0].split('_')[-1]))

    I = np.argsort(V)
    V = np.sort(V)

    for i,file in enumerate(fileList[I]):
        print("Number of vertices : " + str(V[i]))

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

        E.append(np.average(np.abs(AmpSomme-AmpAnalytique)))
        print("Erreur : " + str(E[-1]))

        TH = np.arange(0,2*np.pi,2*np.pi/len(X))
        TH = np.append(TH,2*np.pi)
    
        """
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(TH,AmpAnalytique,label="Analytical solution (dB)")
        ax.plot(TH,AmpSomme,label="Numerical solution (dB)")

        ax.set_title("Acoustic pressure field computed for " + str(V[i]) + " vertices")

        maxAmp = max(max(AmpSomme),max(AmpAnalytique))*1.1

        ax.annotate('x', xy=(np.pi/40,maxAmp), xycoords='data', annotation_clip=False, size = 12)
        ax.annotate('y', xy=(np.pi/2 - np.pi/40,maxAmp), xycoords='data', annotation_clip=False, size = 12)

        plt.legend()
        plt.show()
        """
    plt.plot(V,np.log10(E),label="kd = "+str(np.round(kd,2)))

plt.xlabel("Number of vertices")
plt.ylabel("log(Average error)")
plt.xscale('log')  
plt.title("Average error depending on the number of vertices")    
plt.legend()
plt.show()