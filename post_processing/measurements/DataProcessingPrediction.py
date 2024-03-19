#!/usr/bin/python3

#Acoustics package
import measpy as ms
from measpy._tools import wrap
from unyt import Unit

#Utility packages
import numpy as np
import glob
import os
import sys
import csv
from copy import copy

#Mesh package
import trimesh as trimesh
import meshio as meshio

#Plot package
import matplotlib.pyplot as plt

from DataProcessingTools import save_fig, set_title, plot_spatial_data, plot_polar_data, plot_absolute_error_spatial, plot_relative_error_spatial, plot_relative_separated_error_spatial, plot_weighting, plot_absolute_error, plot_relative_error, plot_relative_separated_error, compute_l2_errors, compute_l2_errors_spatial, figsize, cmap, cmap2, markers, octBandFrequencies, fminValidity, fmaxValidity

def display_point(point):
	roundedPoint = np.round(point,2)
	return("X : " + str(roundedPoint[0]) + " m, Y : " + str(roundedPoint[1]) + " m, Z : " + str(roundedPoint[2])) + " m"

if __name__ == "__main__":

	#Get interest frequency
	Frequencies = []
	try:
		Frequencies = [int(item) for item in sys.argv[1].split(",")]
	except (IndexError, ValueError):
		print("Invalid frequency, defaulting to all frequencies")
  
	#Get interest point index
	PointsIndex = []
	try:
		PointsIndex = [int(item) for item in sys.argv[2].split(",")]
	except (IndexError, ValueError):
		print("Invalid point index, defaulting to all points indices")
  
	keyFunction = lambda file : int(os.path.basename(file).split(".")[0].split("_")[-1])

	#Get data files
	Folders = ["./"]
	try:
		Folders = sys.argv[3].split(",")
	except IndexError:
		print("Invalid folder, defaulting to current folder")

	FoldersLabels = [""]*len(Folders)
	try:
		FoldersLabels = sys.argv[4].split(",")
	except IndexError:
		print("Invalid folders lables, defaulting to no labels")

	FoldersScalingFactors = [1.0]*len(Folders)
	try:
		FoldersScalingFactors = [float(item) for item in sys.argv[5].split(",")]
	except (IndexError,ValueError):
		print("Invalid folders scaling factors, defaulting to 1.0")

	globalFolder = "./"
	try:
		globalFolder = sys.argv[6]
		os.makedirs(globalFolder, exist_ok=True)
	except IndexError:
		print("Invalid global folder, defaulting to current folder")
		
	Files = []

	if(len(Frequencies) == 0):
		Files = sorted(glob.glob(Folders[0] + "output_*.csv"), key = keyFunction)
		Frequencies = np.array([keyFunction(file) for file in Files])
		Frequencies = Frequencies[Frequencies < fmaxValidity*1.1]
		Frequencies = Frequencies[Frequencies > fminValidity*0.9]

	Files = [[glob.glob(folder + "output_" + str(f) + ".csv")[0] for f in Frequencies] for folder in Folders]

	#Get measurements points
	X = []
	Y = []
	Z = []

	with open(Files[0][0], newline='') as csvfile:  #Also assumed all equal for all folders !
		reader = csv.reader(csvfile, delimiter=';')
		for row in reader:
			X.append(float(row[0]))
			Y.append(float(row[1]))
			Z.append(float(row[2]))

	X = np.array(X)
	Y = np.array(Y)
	Z = np.array(Z)

	Points = np.array([X,Y,Z]).T

	if(len(PointsIndex) == 0):
		PointsIndex = np.arange(len(Points))
	PointsIndex = np.array(PointsIndex)

	MeasuredData = np.empty((len(Files),len(Frequencies),len(PointsIndex)),dtype=complex)
	PredictedData = np.empty((len(Files),len(Frequencies),len(PointsIndex)),dtype=complex)
	for k,fileList in enumerate(Files):
		for i,file in enumerate(fileList):
			with open(file, newline='') as csvfile:
				reader = csv.reader(csvfile, delimiter=';')
				counter = 0
				for j,row in enumerate(reader):
					if(j in PointsIndex):
						MeasuredData[k,i,counter] = complex(float(row[3]),float(row[4]))
						PredictedData[k,i,counter] = complex(float(row[5]),float(row[6]))
						counter += 1

	PlotFrequencies = np.logspace(np.log10(min(Frequencies)),np.log10(max(Frequencies)),10*len(Frequencies))	#For pretty plots

	### FREQUENCIES ###

	#To avoid plotting issues
	if(len(PointsIndex) > 1):

		#High number of frequencies, or single frequency (for separate colors !)
		if(len(Frequencies) > 3 or len(Frequencies) == 1):

			if(len(Frequencies) <= 5):

				#Compute error at each fixed frequency for all points
				for i,(f,predicted,measured) in enumerate(zip(Frequencies,PredictedData.transpose(1,0,2),MeasuredData.transpose(1,0,2))):

					#figBoth, axBoth = plt.subplots(1,2,figsize=figsize,subplot_kw={'projection': 'polar'})
					figAbs, axAbs = plt.subplots(2,figsize=figsize)
					figRel, axRel = plt.subplots(1,figsize=figsize)
					figRelSep, axRelSep = plt.subplots(2,figsize=figsize)

					for j,(folder,folderLabel) in enumerate(zip(Folders,FoldersLabels)):

						print("Folder : " + folder)

						additionnalLabel = "" if len(folderLabel) == 0 else " - " + folderLabel

						#plot_polar_data(predicted[j],PointsIndex,unit=Unit("Pa/V"),ax=axBoth, marker=markers[2*j], color=cmap2(4*j), label="Prediction" + additionnalLabel)
						#plot_polar_data(measured[j],PointsIndex,unit=Unit("Pa/V"),ax=axBoth, marker=markers[2*j+1], color=cmap2(4*j+1), label="Measurement" + additionnalLabel)

						plot_absolute_error_spatial(predicted[j], measured[j], PointsIndex, ax=axAbs, marker=markers[j], color=cmap(j), label="Absolute error" + additionnalLabel)
						plot_relative_error_spatial(predicted[j], measured[j], PointsIndex, ax=axRel, marker=markers[j], color=cmap(j), label="Relative error" + additionnalLabel)
						plot_relative_separated_error_spatial(predicted[j], measured[j], PointsIndex, ax=axRelSep, marker=markers[j], color=cmap(j), label="Relative error" + additionnalLabel)

						#Compute l2 errors
						errorAbs, errorRel = compute_l2_errors_spatial(predicted[j], measured[j])
						print("Absolute L2 error at " + str(f) + " Hz and points " + str(PointsIndex) + " : " + str(errorAbs) + " Pa/V")
						print("Relative L2 error at " + str(f) + " Hz and points " + str(PointsIndex) + " : " + str(100*errorRel) + " %")

					#figBoth.suptitle("Pressure/Input signal TFE - f = " + str(f) + " Hz")
					#set_title(axAbs, "Pressure/Input signal TFE absolute error - f = " + str(f) + " Hz")
					#set_title(axRel, "Pressure/Input signal TFE relative error - f = " + str(f) + " Hz")
					#set_title(axRelSep, "Pressure/Input signal TFE modulus and phase relative errors\n - f = " + str(f) + " Hz")

					#save_fig(figBoth, globalFolder + "Both_" + str(f) + ".pdf")
					save_fig(figAbs, globalFolder + "Absolute_" + str(f) + ".pdf")
					save_fig(figRel, globalFolder + "Relative_" + str(f) + ".pdf")
					save_fig(figRelSep, globalFolder + "RelativeSeparated_" + str(f) + ".pdf")
					plt.close("all")

			#High number of points as well : plot l2 error for each frequency
			if(len(Frequencies) > 5 and len(PointsIndex) > 5):

				#If High number of points as well : plot l2 error for each frequency
				figErrorAbs, axErrorAbs = plt.subplots(1,figsize=figsize)
				figErrorRel, axErrorRel = plt.subplots(1,figsize=figsize)

				for j,(folder,folderLabel,folderScalingFactor) in enumerate(zip(Folders,FoldersLabels,FoldersScalingFactors)):
					AllErrorAbs = []
					AllErrorRel = []
					additionnalLabel = "" if len(folderLabel) == 0 else " - " + folderLabel

					for i,(f,predicted,measured) in enumerate(zip(Frequencies,PredictedData[j],MeasuredData[j])):
						#Compute l2 errors
						errorAbs, errorRel = compute_l2_errors_spatial(predicted, measured)
						AllErrorAbs.append(errorAbs)
						AllErrorRel.append(errorRel)

					wAbs = ms.Weighting(freqs=Frequencies,amp=AllErrorAbs,phase=np.zeros(len(Frequencies)))
					wRel = ms.Weighting(freqs=Frequencies,amp=AllErrorRel,phase=np.zeros(len(Frequencies)))

					plot_weighting(wAbs,PlotFrequencies,unit=Unit("Pa/V"),ax=axErrorAbs, dby=False, plot_phase=False, validity_range=[fminValidity,fmaxValidity], scalingFactor=folderScalingFactor, marker=markers[j], color=cmap(j), label="Absolute error" + additionnalLabel)
					plot_weighting(wRel,PlotFrequencies,unit=Unit("1"),ax=axErrorRel, dby=False, plot_phase=False, validity_range=[fminValidity,fmaxValidity], scalingFactor=folderScalingFactor, marker=markers[j], color=cmap(j), label="Relative error" + additionnalLabel)

				#set_title(axErrorAbs, "Pressure/Input signal TFE absolute L2 error")
				#set_title(axErrorRel, "Pressure/Input signal TFE relative L2 error")

				save_fig(figErrorAbs, globalFolder + "AbsoluteL2Error.pdf")
				save_fig(figErrorRel, globalFolder + "RelativeL2Error.pdf")
				plt.close("all")

		#Low number of frequencies
		else:
			print("Skipping multiple frequencies plot for low number of frequencies")

	### POINTS ###

	#To avoid plotting issues
	if(len(Frequencies) > 10):

		#High number of points or single point (for separate colors !)
		if(len(PointsIndex) > 3 or len(PointsIndex) == 1):

			if(len(PointsIndex) <= 5):
				#Compute error at each fixed point for all frequencies
				for i,(index,predicted,measured) in enumerate(zip(PointsIndex,PredictedData.transpose(2,0,1),MeasuredData.transpose(2,0,1))):

					#figBoth, axBoth = plt.subplots(2,figsize=figsize)
					figAbs, axAbs = plt.subplots(2,figsize=figsize)
					figRel, axRel = plt.subplots(1,figsize=figsize)
					figRelSep, axRelSep = plt.subplots(2,figsize=figsize)

					for j,(folder,folderLabel,folderScalingFactor) in enumerate(zip(Folders,FoldersLabels,FoldersScalingFactors)):

						print("Folder : " + folder)

						additionnalLabel = "" if len(folderLabel) == 0 else " - " + folderLabel
					
						wPredicted = ms.Weighting(freqs=Frequencies,amp=np.abs(predicted[j]),phase=wrap(np.angle(predicted[j])))
						wMeasured = ms.Weighting(freqs=Frequencies,amp=np.abs(measured[j]),phase=wrap(np.angle(measured[j])))

						#plot_weighting(wPredicted,PlotFrequencies,unit=Unit("Pa/V"),ax=axBoth, validity_range=[fminValidity,fmaxValidity], marker=markers[2*j], color=cmap2(4*j), label="Prediction" + additionnalLabel)
						#plot_weighting(wMeasured,PlotFrequencies,unit=Unit("Pa/V"),ax=axBoth, validity_range=[fminValidity,fmaxValidity], marker=markers[2*j+1], color=cmap2(4*j+1), label="Measurement" + additionnalLabel)

						plot_absolute_error(wPredicted, wMeasured, PlotFrequencies, ax=axAbs, validity_range=[fminValidity,fmaxValidity], scalingFactor=folderScalingFactor, marker=markers[j], color=cmap(j), label="Absolute error" + additionnalLabel)
						plot_relative_error(wPredicted, wMeasured, PlotFrequencies, ax=axRel, validity_range=[fminValidity,fmaxValidity], scalingFactor=folderScalingFactor, marker=markers[j], color=cmap(j), label="Relative error" + additionnalLabel)
						plot_relative_separated_error(wPredicted, wMeasured, PlotFrequencies, ax=axRelSep, validity_range=[fminValidity,fmaxValidity], scalingFactor=folderScalingFactor, marker=markers[j], color=cmap(j), label="Relative error" + additionnalLabel)

						#Compute l2 errors
						errorAbs, errorRel = compute_l2_errors(wPredicted, wMeasured)
						print("Absolute L2 error at point " + str(index) + " for frequencies " + str(Frequencies) + " : " + str(errorAbs) + " Pa/V")
						print("Relative L2 error at point " + str(index) + " for frequencies " + str(Frequencies) + " : " +  str(100*errorRel) + " %")

					#set_title(axBoth, "Pressure/Input signal TFE\nPoint " + display_point(Points[index]))
					#set_title(axAbs, "Pressure/Input signal TFE absolute error\nPoint " + display_point(Points[index]))
					#set_title(axRel, "Pressure/Input signal TFE relative error\nPoint " + display_point(Points[index]))
					#set_title(axRelSep, "Pressure/Input signal TFE modulus and phase relative errors\nPoint " + display_point(Points[index]))

					#save_fig(figBoth, globalFolder + "Both_" + str(index) + ".pdf")
					save_fig(figAbs, globalFolder + "Absolute_" + str(index) + ".pdf")
					save_fig(figRel, globalFolder + "Relative_" + str(index) + ".pdf")
					save_fig(figRelSep, globalFolder + "RelativeSeparated_" + str(index) + ".pdf")
					plt.close("all")

		#Low number of points
		else:
			print("Skipping multiple points plot for low number of points")
			

	if(len(Frequencies) == 1 and len(PointsIndex) == 1):
		for predicted,measured in zip(PredictedData,MeasuredData):
			#Compute l2 errors
			errorAbs, errorRel = compute_l2_errors_spatial(predicted, measured)
			print("Absolute L2 error at " + str(f) + " Hz : " + str(errorAbs) + " Pa/V")
			print("Relative L2 error at " + str(f) + " Hz : " + str(100*errorRel) + " %")
		raise ValueError("Not enough data to plot !")

	#plt.show()