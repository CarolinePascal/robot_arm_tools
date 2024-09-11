#!/usr/bin/python3

#System packages
import csv

#Utility packages
import numpy as np

#Custom tools packages
from robot_arm_acoustic.simulations.AcousticTools import *
from robot_arm_acoustic.simulations.PlotTools import *

from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator

## Function plotting the error between computed values and analytical values for an output files folder on the z=0 plane for a given parameters configuration
#  @param postProcessingID ID of the post-processing function (c.f. PlotTools.py)
#  @param analyticalFunctionID ID of the analytical function (c.f. AcousticTools.py)
#  @param errorID ID of the error function (c.f. PlotTools.py)
#  @param figureName Name of the output figure
def plotSphericCut(postProcessingID,analyticalFunctionID,errorID,figureName="output.pdf",**kwargs):

    #Get post-processing and analytical functions
    postProcessingFunction = postProcessingFunctions[postProcessingID]
    analyticalFunction = analyticalFunctions[analyticalFunctionID]
    errorFunction = errorFunctions[errorID]

    parametersConfigurations,parametersList,parametersUnits,fileList = getParametersConfigurations()

    #Get indices frequency (mandatory parameter !) and dipole distance (optional parameter, default is 0)
    frequencyIndex = np.where(parametersList=="frequency")[0][0]
    try:
        dipoleDistanceIndex = np.where(parametersList=="dipoleDistance")[0][0]
    except IndexError:
        dipoleDistanceIndex = None

    #Registering the desired parameters configurations
    configurations = parametersConfigurations
    singleParametersList = []
    multipleParametersList = []

    for i,parameter in enumerate(parametersList):
        parameterValues = np.unique(configurations[:,i])
        if(len(parameterValues) == 1):
            singleParametersList.append(parameter)
        else:
            multipleParametersList.append(parameter)
        
        fixedParameterValues = parameterValues

        if(parameter in kwargs):
            if(kwargs[parameter] == "*"):
                fixedParameterValues = parameterValues
            else:
                fixedParameterValues = kwargs[parameter]
                if(not isinstance(fixedParameterValues, list)):
                    fixedParameterValues = [fixedParameterValues]
                
        else:
            if(len(parameterValues) == 1):
                print("Only possible value for " + parameter +" is " + str(configurations[0,i]) +  " (" + parametersUnits[i] + ")")
                continue
            else:
                fixedParameterValues = list(input("What value for " + parameter + " ? (default : *) " + str(np.sort(parameterValues)) + " (" + parametersUnits[i] + ")").split(" "))

        try:
            fixedParameterValues = [float(item) for item in fixedParameterValues if float(item) in parameterValues]
            if(len(fixedParameterValues) == 0):
                fixedParameterValues = parameterValues
        except (ValueError,TypeError):
            fixedParameterValues = parameterValues

        interestIndices = np.hstack([np.where(configurations[:,i] == value)[0] for value in fixedParameterValues])
        
        configurations = configurations[interestIndices]

    if(postProcessingID == "re/im" or postProcessingID == "id"):
        fig,ax = plt.subplots(1,2,figsize=figsize,subplot_kw=dict(projection='polar'))
    elif(postProcessingID == "mod" or postProcessingID == "phase"):
        fig,ax = plt.subplots(1,figsize=(figsize[0]*0.5,figsize[0]),subplot_kw=dict(projection='polar'))
        ax = [ax]

    #title = makeTitle(configurations,singleParametersList,parametersList,parametersUnits,"Computed acoustic pressure field")
    #fig.suptitle(title)

    for i,configuration in enumerate(configurations):
        #Find the corresponding file names
        file = fileList[(parametersConfigurations == configuration).all(1)][0]

        #Get configuration frequency and dipole distance parameters
        f = configuration[frequencyIndex]
        halfDipoleDistance = configuration[dipoleDistanceIndex]/2 if dipoleDistanceIndex is not None else 0

        #Create empty arrays
        numericValuesA = []
        numericValuesN = []

        if(analyticalFunction is not None):
            analyticalValues = []

        R = []
        Theta = []
        Phi = []

        #Fill arrays from output file and analytical function
        with open(file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar='|')

            for row in reader:
                x = float(row[0])
                y = float(row[1])
                z = float(row[2])
    
                R.append(np.sqrt(x*x + y*y + z*z))
                Theta.append(np.arctan2(np.sqrt(x*x + y*y),z))
                Phi.append(np.arctan2(y,x))

                if(analyticalFunction is not None):
                    analyticalValues.append(analyticalFunction(f,np.sqrt(x*x + y*y + z*z),np.arctan2(np.sqrt(x*x + y*y),z),np.arctan2(y,x),halfDipoleDistance))
                numericValuesA.append(complex(float(row[3]),float(row[4])))
                numericValuesN.append(complex(float(row[5]),float(row[6])))

        R = np.array(R)
        Theta = np.array(Theta)
        Phi = np.array(Phi)

        #Remove outliers
        outliers = []
        if(analyticalFunction is not None):
            outliers = np.concatenate((np.where(np.isinf(np.abs(analyticalValues)))[0],np.where(np.abs(analyticalValues) < np.mean(np.abs(analyticalValues)) - 2*np.std(np.abs(analyticalValues)))[0]))
            analyticalValues = postProcessingFunction(np.delete(analyticalValues,outliers))

        R = np.delete(R,outliers)
        Theta = np.delete(Theta,outliers)
        Phi = np.delete(Phi,outliers)
        numericValuesN = postProcessingFunction(np.delete(numericValuesN,outliers))
        numericValuesA = postProcessingFunction(np.delete(numericValuesA,outliers))

        #Compute error over the z=0 plane
        tmpUnit = "Pa" if postProcessingID != "phase" else "rad"
        absoluteError = 0
        relativeError = 0
        if(analyticalFunction is not None):
            absoluteError = np.round(errorFunction(numericValuesN - analyticalValues),6)
            relativeError = np.round(100*errorFunction(numericValuesN - analyticalValues)/errorFunction(analyticalValues),6)
        else:
            absoluteError = np.round(errorFunction(numericValuesN - numericValuesA),6)
            relativeError = np.round(100*errorFunction(numericValuesN - numericValuesA)/errorFunction(numericValuesA),6)

        print("Absolute error numerical solution = " +  str(absoluteError) + " (" + tmpUnit + ")")
        print("Relative error numerical solution = " +  str(relativeError) + " %")

        #if(len(configurations) == 1):
            #fig.suptitle(title + "\n Relative error = " + str(np.round(relativeError,3)) + " \%")

        #Create plot
        Phi = np.append(Phi,Phi[0])
        if(len(np.shape(numericValuesA)) == 1):
            if(analyticalFunction is not None):
                analyticalValues = np.append(analyticalValues,analyticalValues[0])
            numericValuesA = np.append(numericValuesA,numericValuesA[0])
            numericValuesN = np.append(numericValuesN,numericValuesN[0])
        else:
            if(analyticalFunction is not None):
                analyticalValues = np.vstack((analyticalValues.T,analyticalValues.T[0])).T
            numericValuesA = np.vstack((numericValuesA.T,numericValuesA.T[0])).T
            numericValuesN = np.vstack((numericValuesN.T,numericValuesN.T[0])).T

        legend = makeLegend(configuration,multipleParametersList,parametersList,parametersUnits,"")

        def plotSphericFunction(function, functionName, unit, ax=None, legend=""):

            if(ax is None):
                _,ax = plt.subplots(1,figsize=figsize,subplot_kw=dict(projection='polar'))

            if(analyticalFunction is not None):
                if(legend != ""): 
                    #ax.plot(Phi,function(numericValuesA), label="FreeFem analytical solution",color=cmap(2),linestyle="dotted",linewidth=2.5)
                    ax.plot(Phi,function(numericValuesN), label="Numeric",color=cmap(1),linewidth=2.5)
                    ax.plot(Phi,function(analyticalValues), label="Analytic", color=cmap(0),linestyle="dashed",linewidth=2.5)
                else:
                    ax.plot(Phi,function(numericValuesN),color=cmap(1),linewidth=2.5,alpha=0.25)
            else:
                if(legend != ""):
                    ax.plot(Phi,function(numericValuesN),label="Computed solution - " + functionName + " (" + unit + ")" + legend, color=cmap(i),linewidth=2.5)
                    ax.plot(Phi,function(numericValuesA),label="Measured data - " + functionName + " (" + unit + ")" + legend, color=cmap(i), linestyle="dashed",linewidth=2.5)
                else:
                    ax.plot(Phi,function(numericValuesN), color=cmap(i),linewidth=2.5)
                    ax.plot(Phi,function(numericValuesA), color=cmap(i), linestyle="dashed",linewidth=2.5)

            return(ax)

        legendFlag = True if(len(configurations) >= 1 and i == 0) else False
 
        if(postProcessingID == "re/im"):
            plotSphericFunction(lambda x : x[0],"real part","Pa",ax[0],legend*legendFlag)
            ax[0].set_title("Real part", y=-0.325)

            plotSphericFunction(lambda x : x[1],"imaginary part","Pa",ax[1],legend*legendFlag)
            ax[1].set_title("Imaginary part", y=-0.325)

        elif(postProcessingID == "id"):
            plotSphericFunction(np.abs,"modulus","Pa",ax[0],legend*legendFlag)
            ax[0].set_title("Modulus (Pa)", y=-0.325)

            plotSphericFunction(lambda x :np.angle(x),"phase","rad",ax[1],legend*legendFlag)
            ax[1].set_title("Phase (rad)", y=-0.325)

        elif(postProcessingID == "mod"):
            plotSphericFunction(lambda x:x,"modulus","Pa",ax[0],legend*legendFlag)
            ax[0].set_title("Modulus (Pa)", y=-0.325)

        elif(postProcessingID == "phase"):
            plotSphericFunction(lambda x:x,"phase","rad",ax[0],legend*legendFlag)
            ax[0].set_title("Phase (rad)", y=-0.325)

    for i,subax in enumerate(ax):

        if((postProcessingID == "id" and i == 1) or postProcessingID == "phase"):
            thetamax = subax.get_rmax()
            subax.set_rmin(thetamax - np.pi)
            subax.set_rmax(thetamax + np.pi)
        else:
            rmax = subax.get_rmax()
            subax.set_rmax(10**(1.5/20)*rmax)
            subax.set_rmin(10**(-1.5/20)*rmax)
        
        subax.set_thetagrids(np.arange(0,360,45),['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$'])

        arrow = dict(arrowstyle='<-',color="gray")
        subax.annotate("",xy=(0.5,0.5),xytext=(1.0,0.5),xycoords="axes fraction",arrowprops=arrow)
        subax.annotate('x',xy=(0.5,0.5),xytext=(0.94,0.435),xycoords="axes fraction",color="gray")
        subax.annotate("",xy=(0.5,0.5),xytext=(0.5,1.0),xycoords="axes fraction",arrowprops=arrow)
        subax.annotate('y',xy=(0.5,0.5),xytext=(0.435,0.94),xycoords="axes fraction",color="gray")

        subax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
        subax.yaxis.set_major_locator(MaxNLocator(4))
        subax.grid(linestyle= '--', which="major")
        subax.tick_params(axis='y', which='major', pad=10, labelsize=15)
        subax.tick_params(axis='x', which='major', pad=5, labelsize=20)

    if(len(ax) == 1):
        ax[0].legend(bbox_to_anchor=(0.5,1.0), loc='lower center', ncol=4, borderaxespad=2, reverse=False,  columnspacing=0.5)
    else:
        ax[0].legend(bbox_to_anchor=(-0.2, 1.0, 2.7, .1), loc='lower left', ncol=2, borderaxespad=2, reverse=False, mode="expand", columnspacing=0.5)

    if(not figureName is None):
        fig.savefig(figureName, dpi = 300, bbox_inches = 'tight')
    else:
        plt.show()

if __name__ == "__main__": 

    #postProcessing = input("Post processing function ? (default : id) " + str(list((postProcessingFunctions.keys()))))
    postProcessing = "id"
    if(postProcessing not in list((postProcessingFunctions.keys()))):
        postProcessing = "id"

    #error = input("Error function ? (default : l2) " + str(list((errorFunctions.keys()))))
    error = "l2"
    if(error not in list((errorFunctions.keys()))):
        error = "l2"

    analytical = "infinitesimalDipole"
    try:
        analytical = os.getcwd().split("/")[-1].split("_")[0]
    except IndexError:
        analytical = input("Analytical function ? (default : infinitesimalDipole) " + str(list((analyticalFunctions.keys()))))
    if(analytical not in list((analyticalFunctions.keys()))):
        analytical = None

    kwargs = {}
    kwargs["resolution"] = 0.005
    kwargs["size"] = 0.5
    kwargs["dipoleDistance"] = 0.45
    kwargs["frequency"] = 5000
    kwargs["sigmaPosition"] = [0.0]

    plotSphericCut(postProcessing,analytical,error,"SphericPlaneCut1.pdf",**kwargs)

    kwargs["resolution"] = 0.125

    plotSphericCut(postProcessing,analytical,error,"SphericPlaneCut2.pdf",**kwargs)

    kwargs["resolution"] = 0.005
    kwargs["sigmaPosition"] = [0.0005]
    kwargs["iteration"] = "*"
    print(kwargs)

    plotSphericCut(postProcessing,analytical,error,"SphericPlaneCutSigma1.pdf",**kwargs)

    kwargs["sigmaPosition"] = [0.0025]
    kwargs["iteration"] = "*"

    plotSphericCut(postProcessing,analytical,error,"SphericPlaneCutSigma2.pdf",**kwargs)

    kwargs["sigmaPosition"] = [0.005]
    kwargs["iteration"] = "*"

    plotSphericCut(postProcessing,analytical,error,"SphericPlaneCutSigma3.pdf",**kwargs)

    kwargs["sigmaPosition"] = [0.0125]
    kwargs["iteration"] = "*"

    plotSphericCut(postProcessing,analytical,error,"SphericPlaneCutSigma4.pdf",**kwargs)


