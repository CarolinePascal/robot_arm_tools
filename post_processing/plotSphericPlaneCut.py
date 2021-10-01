import matplotlib.pyplot as plt
import numpy as np
import csv

import glob

fileList = np.array(glob.glob("Output*.txt"))
verticesNumber = []

for file in fileList:
    verticesNumber.append(int(file.split('.')[0].split('_')[1]))

I = np.argsort(verticesNumber)

for file in fileList[I]:
    verticesNumber = file.split('.')[0].split('_')[1]
    print("Number of vertices : " + str(verticesNumber))

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
    
            #AmpSomme.append(20*np.log10(np.abs(np.complex(float(row[3]),float(row[4])))/20e-6))
            #AmpAnalytique.append(20*np.log10(np.abs(np.complex(float(row[5]),float(row[6])))/20e-6))
            AmpSomme.append(np.log(np.abs(np.complex(float(row[3]),float(row[4])))))
            AmpAnalytique.append(np.log(np.abs(np.complex(float(row[5]),float(row[6])))))


    #X0 = np.argmax(np.abs(np.array(X)))

    AmpSomme = np.array(AmpSomme)
    AmpAnalytique = np.array(AmpAnalytique)

    print(np.exp(AmpAnalytique[0] - AmpSomme[0]))

    #AmpAnalytique = np.roll(AmpAnalytique,X0)
    AmpAnalytique = np.append(AmpAnalytique,AmpAnalytique[0])
    #AmpSomme = np.roll(AmpSomme,X0)
    AmpSomme = np.append(AmpSomme,AmpSomme[0])

    TH = np.arange(0,2*np.pi,2*np.pi/len(X))
    TH = np.append(TH,2*np.pi)
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(TH,AmpAnalytique,label="Analytical solution (dB)")
    ax.plot(TH,AmpSomme,label="Numerical solution (dB)")

    ax.set_title("Acoustic pressure field computed for " + str(verticesNumber) + " vertices")

    maxAmp = max(max(AmpSomme),max(AmpAnalytique))*1.1

    ax.annotate('x', xy=(np.pi/40,maxAmp), xycoords='data', annotation_clip=False, size = 12)
    ax.annotate('y', xy=(np.pi/2 - np.pi/40,maxAmp), xycoords='data', annotation_clip=False, size = 12)

    plt.legend()
    plt.show()