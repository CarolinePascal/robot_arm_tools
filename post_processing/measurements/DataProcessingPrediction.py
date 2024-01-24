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
	Files = []
	if(len(Frequencies) != 0):
		Files = [glob.glob("./output_" + str(f) + ".csv")[0] for f in Frequencies]
	else:
		Files = sorted(glob.glob("./output_*.csv"), key = keyFunction)
		Frequencies = np.array([keyFunction(file) for file in Files])

	#Get measurements points
	X = []
	Y = []
	Z = []

	with open(Files[0], newline='') as csvfile:
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

	MeasuredData = np.empty((len(Frequencies),len(PointsIndex)),dtype=complex)
	PredictedData = np.empty((len(Frequencies),len(PointsIndex)),dtype=complex)
	for i,file in enumerate(Files):
		with open(file, newline='') as csvfile:
			reader = csv.reader(csvfile, delimiter=';')
			counter = 0
			for j,row in enumerate(reader):
				if(j in PointsIndex):
					MeasuredData[i,counter] = complex(float(row[3]),float(row[4]))
					PredictedData[i,counter] = complex(float(row[5]),float(row[6]))
					counter += 1

	PlotFrequencies = np.logspace(np.log10(min(Frequencies)),np.log10(max(Frequencies)),10*len(Frequencies))	#For pretty plots

	### FREQUENCIES ###

	#To avoid plotting issues
	if(len(PointsIndex) > 1):

		#High number of frequencies, or single frequency (for separate colors !)
		if(len(Frequencies) > 3 or len(Frequencies) == 1):

			if(len(Frequencies) <= 5):

				#Compute error at each fixed frequency for all points
				for i,(f,predicted,measured) in enumerate(zip(Frequencies,PredictedData,MeasuredData)):

					figBoth, axBoth = plt.subplots(1,2,figsize=figsize,subplot_kw={'projection': 'polar'})
					figAbs, axAbs = plt.subplots(2,figsize=figsize)
					figRel, axRel = plt.subplots(1,figsize=figsize)
					figRelSep, axRelSep = plt.subplots(2,figsize=figsize)

					plot_polar_data(predicted,PointsIndex,unit=Unit("Pa/V"),ax=axBoth, marker=markers[0], color=cmap(0), label="Prediction")
					plot_polar_data(measured,PointsIndex,unit=Unit("Pa/V"),ax=axBoth, marker=markers[1], color=cmap(1), label="Measurement")

					plot_absolute_error_spatial(predicted, measured, PointsIndex, ax=axAbs, marker=markers[0], color=cmap(0), label="Absolute error")
					plot_relative_error_spatial(predicted, measured, PointsIndex, ax=axRel, marker=markers[0], color=cmap(0), label="Relative error")
					plot_relative_separated_error_spatial(predicted, measured, PointsIndex, ax=axRelSep, marker=markers[0], color=cmap(0), label="Relative error")

					#figBoth.suptitle("Pressure/Input signal TFE - f = " + str(f) + " Hz")
					#set_title(axAbs, "Pressure/Input signal TFE absolute error - f = " + str(f) + " Hz")
					#set_title(axRel, "Pressure/Input signal TFE relative error - f = " + str(f) + " Hz")
					#set_title(axRelSep, "Pressure/Input signal TFE modulus and phase relative errors\n - f = " + str(f) + " Hz")

					save_fig(figBoth, "./Both_" + str(f) + ".pdf")
					save_fig(figAbs, "./Absolute_" + str(f) + ".pdf")
					save_fig(figRel, "./Relative_" + str(f) + ".pdf")
					save_fig(figRelSep, "./RelativeSeparated_" + str(f) + ".pdf")
					plt.close("all")

					#Compute l2 errors
					errorAbs, errorRel = compute_l2_errors_spatial(predicted, measured)
					print("Absolute L2 error at " + str(f) + " Hz : " + str(errorAbs) + " Pa/V")
					print("Relative L2 error at " + str(f) + " Hz : " + str(100*errorRel) + " %")

			#High number of points as well : plot l2 error for each frequency
			if(len(Frequencies) > 5 and len(PointsIndex) > 5):

				#If High number of points as well : plot l2 error for each frequency
				AllErrorAbs = []
				AllErrorRel = []

				for i,(f,predicted,measured) in enumerate(zip(Frequencies,PredictedData,MeasuredData)):
					#Compute l2 errors
					errorAbs, errorRel = compute_l2_errors_spatial(predicted, measured)
					AllErrorAbs.append(errorAbs)
					AllErrorRel.append(errorRel)

				figErrorAbs, axErrorAbs = plt.subplots(1,figsize=figsize)
				figErrorRel, axErrorRel = plt.subplots(1,figsize=figsize)

				wAbs = ms.Weighting(freqs=Frequencies,amp=AllErrorAbs,phase=np.zeros(len(Frequencies)))
				wRel = ms.Weighting(freqs=Frequencies,amp=AllErrorRel,phase=np.zeros(len(Frequencies)))

				plot_weighting(wAbs,PlotFrequencies,unit=Unit("Pa/V"),ax=axErrorAbs, dby=False, plot_phase=False, validity_range=[fminValidity,fmaxValidity], marker=markers[0], color=cmap(0), label="Absolute error")
				plot_weighting(wRel,PlotFrequencies,unit=Unit("1"),ax=axErrorRel, dby=False, plot_phase=False, validity_range=[fminValidity,fmaxValidity], marker=markers[0], color=cmap(0), label="Relative error")

				#set_title(axErrorAbs, "Pressure/Input signal TFE absolute L2 error")
				#set_title(axErrorRel, "Pressure/Input signal TFE relative L2 error")

				save_fig(figErrorAbs, "./AbsoluteL2Error.pdf")
				save_fig(figErrorRel, "./RelativeL2Error.pdf")
				plt.close("all")

		#Low number of frequencies
		else:

			#Compute error for all frequencies for all points
			figBoth, axBoth = plt.subplots(1,2,figsize=figsize,subplot_kw={'projection': 'polar'})
			figAbs, axAbs = plt.subplots(2,figsize=figsize)
			figRel, axRel = plt.subplots(1,figsize=figsize)
			figRelSep, axRelSep = plt.subplots(2,figsize=figsize)

			for i,(f,predicted,measured) in enumerate(zip(Frequencies,PredictedData,MeasuredData)):

				plot_polar_data(predicted,PointsIndex,unit=Unit("Pa/V"),ax=axBoth, marker=markers[2*i], color=cmap2(4*i), label="Prediction at " + str(f) + " Hz")
				plot_polar_data(measured,PointsIndex,unit=Unit("Pa/V"),ax=axBoth, marker=markers[2*i+1], color=cmap2(4*i+1), label="Measurement at " + str(f) + " Hz")

				plot_absolute_error_spatial(predicted, measured, PointsIndex, ax=axAbs, marker=markers[i], color=cmap(i), label="Absolute error at " + str(f) + " Hz")
				plot_relative_error_spatial(predicted, measured, PointsIndex, ax=axRel, marker=markers[i], color=cmap(i), label="Relative error at " + str(f) + " Hz")
				plot_relative_separated_error_spatial(predicted, measured, PointsIndex, ax=axRelSep, marker=markers[i], color=cmap(i), label="Relative error at " + str(f) + " Hz")

				#Compute l2 errors
				errorAbs, errorRel = compute_l2_errors_spatial(predicted, measured)
				print("Absolute L2 error at " + str(f) + " Hz : " + str(errorAbs) + " Pa/V")
				print("Relative L2 error at " + str(f) + " Hz : " + str(100*errorRel) + " %")

			#figBoth.suptitle("Pressure/Input signal TFE")
			#set_title(axAbs, "Pressure/Input signal TFE absolute error")
			#set_title(axRel, "Pressure/Input signal TFE relative error")
			#set_title(axRelSep, "Pressure/Input signal TFE modulus and phase relative errors")

			save_fig(figBoth, "./Both_" + str(Frequencies) + ".pdf")
			save_fig(figAbs, "./Absolute_" + str(Frequencies) + ".pdf")
			save_fig(figRel, "./Relative_" + str(Frequencies) + ".pdf")
			save_fig(figRelSep, "./RelativeSeparated_" + str(Frequencies) + ".pdf")
			plt.close("all")

	### POINTS ###

	#To avoid plotting issues
	if(len(Frequencies) > 10):

		#High number of points or single point (for separate colors !)
		if(len(PointsIndex) > 3 or len(PointsIndex) == 1):

			if(len(PointsIndex) <= 5):
				#Compute error at each fixed point for all frequencies
				for i,(index,predicted,measured) in enumerate(zip(PointsIndex,PredictedData.T,MeasuredData.T)):

					figBoth, axBoth = plt.subplots(2,figsize=figsize)
					figAbs, axAbs = plt.subplots(2,figsize=figsize)
					figRel, axRel = plt.subplots(1,figsize=figsize)
					figRelSep, axRelSep = plt.subplots(2,figsize=figsize)
					
					wPredicted = ms.Weighting(freqs=Frequencies,amp=np.abs(predicted),phase=wrap(np.angle(predicted)))
					wMeasured = ms.Weighting(freqs=Frequencies,amp=np.abs(measured),phase=wrap(np.angle(measured)))

					plot_weighting(wPredicted,PlotFrequencies,unit=Unit("Pa/V"),ax=axBoth, validity_range=[fminValidity,fmaxValidity], marker=markers[0], color=cmap(0), label="Prediction")
					plot_weighting(wMeasured,PlotFrequencies,unit=Unit("Pa/V"),ax=axBoth, validity_range=[fminValidity,fmaxValidity], marker=markers[1], color=cmap(1), label="Measurement")

					plot_absolute_error(wPredicted, wMeasured, PlotFrequencies, ax=axAbs, validity_range=[fminValidity,fmaxValidity], marker=markers[0], color=cmap(0), label="Absolute error")
					plot_relative_error(wPredicted, wMeasured, PlotFrequencies, ax=axRel, validity_range=[fminValidity,fmaxValidity], marker=markers[0], color=cmap(0), label="Relative error")
					plot_relative_separated_error(wPredicted, wMeasured, PlotFrequencies, ax=axRelSep, validity_range=[fminValidity,fmaxValidity], marker=markers[0], color=cmap(0), label="Relative error")

					#set_title(axBoth, "Pressure/Input signal TFE\nPoint " + display_point(Points[index]))
					#set_title(axAbs, "Pressure/Input signal TFE absolute error\nPoint " + display_point(Points[index]))
					#set_title(axRel, "Pressure/Input signal TFE relative error\nPoint " + display_point(Points[index]))
					#set_title(axRelSep, "Pressure/Input signal TFE modulus and phase relative errors\nPoint " + display_point(Points[index]))

					save_fig(figBoth, "Both_" + str(index) + ".pdf")
					save_fig(figAbs, "Absolute_" + str(index) + ".pdf")
					save_fig(figRel, "Relative_" + str(index) + ".pdf")
					save_fig(figRelSep, "RelativeSeparated_" + str(index) + ".pdf")
					plt.close("all")

					#Compute l2 errors
					errorAbs, errorRel = compute_l2_errors(wPredicted, wMeasured)
					print("Absolute L2 error at point " + str(index) + " : " + str(errorAbs) + " Pa/V")
					print("Relative L2 error at point " + str(index) + " : " + str(100*errorRel) + " %")

		#Low number of points
		else:

			#Compute error for all points for all frequencies
			figBoth, axBoth = plt.subplots(2,figsize=figsize)
			figAbs, axAbs = plt.subplots(2,figsize=figsize)
			figRel, axRel = plt.subplots(1,figsize=figsize)
			figRelSep, axRelSep = plt.subplots(2,figsize=figsize)

			for i,(index,predicted,measured) in enumerate(zip(PointsIndex,PredictedData.T,MeasuredData.T)):
				
				wPredicted = ms.Weighting(freqs=Frequencies,amp=np.abs(predicted),phase=wrap(np.angle(predicted)))
				wMeasured = ms.Weighting(freqs=Frequencies,amp=np.abs(measured),phase=wrap(np.angle(measured)))

				plot_weighting(wPredicted,PlotFrequencies,unit=Unit("Pa/V"),ax=axBoth, validity_range=[fminValidity,fmaxValidity], marker=markers[2*i], color=cmap2(4*i), label="Prediction at point " + str(index))
				plot_weighting(wMeasured,PlotFrequencies,unit=Unit("Pa/V"),ax=axBoth, validity_range=[fminValidity,fmaxValidity], marker=markers[2*i+1], color=cmap2(4*i+1), label="Measurement at point " + str(index))

				plot_absolute_error(wPredicted, wMeasured, PlotFrequencies, ax=axAbs, validity_range=[fminValidity,fmaxValidity], marker=markers[i], color=cmap(i), label="Absolute error at point " + str(index))
				plot_relative_error(wPredicted, wMeasured, PlotFrequencies, ax=axRel, validity_range=[fminValidity,fmaxValidity], marker=markers[i], color=cmap(i), label="Relative error at point " + str(index))
				plot_relative_separated_error(wPredicted, wMeasured, PlotFrequencies, ax=axRelSep, validity_range=[fminValidity,fmaxValidity], marker=markers[i], color=cmap(i), label="Relative error at point " + str(index))

			#set_title(axBoth, "Pressure/Input signal TFE")
			#set_title(axAbs, "Pressure/Input signal TFE absolute error")
			#set_title(axRel, "Pressure/Input signal TFE relative error")
			#set_title(axRelSep, "Pressure/Input signal TFE modulus and phase relative errors")

			save_fig(figBoth, "./Both_" + str(PointsIndex) + ".pdf")
			save_fig(figAbs, "./Absolute_" + str(PointsIndex) + ".pdf")
			save_fig(figRel, "./Relative_" + str(PointsIndex) + ".pdf")
			save_fig(figRelSep, "./RelativeSeparated_" + str(PointsIndex) + ".pdf")
			plt.close("all")

	if(len(Frequencies) == 1 and len(PointsIndex) == 1):
		#Compute l2 errors
		errorAbs, errorRel = compute_l2_errors_spatial(PredictedData[0,0], PredictedData[0,0])
		print("Absolute L2 error at " + str(f) + " Hz : " + str(errorAbs) + " Pa/V")
		print("Relative L2 error at " + str(f) + " Hz : " + str(100*errorRel) + " %")
		raise ValueError("Not enough data to plot !")

	#plt.show()