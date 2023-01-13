#!/usr/bin/python3

#System packages
import sys
import glob

#Plotting package
import vedo as vd

try:
    outputPath = sys.argv[1]
except:
    outputPath = input("Select file : " + str(glob.glob('*.vtu')))

mesh = vd.Mesh(outputPath)
colorBar = vd.ScalarBar(mesh,"Acoustic pressure (Pa)",label_format=':6.2f')
axis = vd.Axes(mesh,xtitle="x (m)",ytitle="y (m)",ztitle="z (m)")
vd.show(mesh,axis,colorBar)












