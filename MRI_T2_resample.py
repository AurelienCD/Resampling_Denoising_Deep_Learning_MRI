# coding: utf-8

import unittest
from slicer.ScriptedLoadableModule import *
import logging
from __main__ import vtk, qt, ctk, slicer
from math import *
import numpy as np
from vtk.util import numpy_support
import SimpleITK as sitk
import time
import codecs
import datetime
import vtkSegmentationCorePython as vtkSegmentationCore
import pydicom
import sys, time, os

############################si il manque une biblio exemple scikit #################################
#slicer.util.pip_install("scikit-image")
#########################################################



data_directory = ""
adresse_save_result=""
spacing_ratio=[2,2,1]



def getSec(s):
    b =int(s[0:2]) *3600 + int(s[2:4])*60 + int(s[5:6])
    return b


def ChangeImageSpacing(image, new_spacing):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetDefaultPixelValue(0)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputSpacing(new_spacing)
    orig_size = np.array(image.GetSize(), dtype=np.int)
    orig_spacing = image.GetSpacing()
    #new_size = orig_size*(orig_spacing/new_spacing)
    new_size=orig_size
    new_size[0]= orig_size[0]*(orig_spacing[0]/new_spacing[0])
    new_size[1]= orig_size[1]*(orig_spacing[1]/new_spacing[1])
    new_size[2]= orig_size[2]*(orig_spacing[2]/new_spacing[2])
    new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    newimage = resample.Execute(image)
    return newimage 

def ChangeImageSpacingRatio(image, spacing_ratio):
    orig_size = np.array(image.GetSize(), dtype=np.int)
    orig_spacing = np.array(image.GetSpacing(), dtype=np.float)
    new_spacing=orig_spacing
    new_spacing[0]=orig_spacing[0]*spacing_ratio[0]
    new_spacing[1]=orig_spacing[1]*spacing_ratio[1]
    new_spacing[2]=orig_spacing[2]*spacing_ratio[2]
    #new_spacing = np.ceil(new_spacing).astype(np.float) #  Image dimensions are in integers
    new_spacing = [float(s) for s in new_spacing] ####resample filter
    #new_size = orig_size*(orig_spacing/new_spacing)
    new_size=orig_size
    new_size[0]= orig_size[0]/(spacing_ratio[0])
    new_size[1]= orig_size[1]/(spacing_ratio[1])
    new_size[2]= orig_size[2]/(spacing_ratio[2])
    new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
    new_size = [int(s) for s in new_size] ####resample filter
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetDefaultPixelValue(0)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin()) 
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    newimage = resample.Execute(image)
    return newimage     
  
########################################################
######################################################

def main(data_directory, adresse_save_result,spacing_ratio):
    timeInit = time.time()
    Nimageouverte=0
    Nimagetraitees=0
    ########################################iterate trough subfolder##############
    directory_list=[]
    i=0 #number of sub folder
    for root, dirs, files in os.walk(data_directory):
        for subdirname in dirs:
            directory_list.append(os.path.join(root,subdirname))
    print(directory_list)
    ################################partie principale du code
    for i in range(len(directory_list)):
        data_directory=directory_list[i].replace('\\','/')
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_directory)
        if not series_IDs:
            print("ERROR: given directory \""+data_directory+"\" does not contain a DICOM series.")
            #sys.exit(1)
        else:
            for i,series_ID in enumerate(series_IDs):   
                Nimageouverte=Nimageouverte+1
                series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, series_ID,useSeriesDetails=False) #useSeriesDetails ?
                try:
                    img_metadata=pydicom.read_file(series_file_names[0])  #importation des metadata lié à l'image
                    #print(img.GetSpacing())
                    #print(img.GetSize())        
                    if img_metadata.Modality=='MR':# and img_metadata.SeriesDescription=="[DetailWB_CTAC_2i-10s-PSF] Body": #"PET TAP AC HD (AC)": #"[DetailWB_CTAC_2i-10s-PSF] Body"
                        try:
                            timeRMR1 = time.time()
                            Nimagetraitees=Nimagetraitees+1
                            series_reader = sitk.ImageSeriesReader()
                            print("\n")
                            print(series_file_names)
                            #####read image name and import#########
                            series_reader.SetFileNames(series_file_names)
                            img = series_reader.Execute()  #importation de l'image
                            ##############change spacing##################
                            image_newMatrice=ChangeImageSpacingRatio(img, spacing_ratio)
                            ###############create the name of the output path and write##########
                            name=str(Nimagetraitees)
                            name_traite=str(Nimagetraitees)+"_resampledBy2.dcm"
                            save_path = os.path.join(adresse_save_result,name)   #   get the save path
                            save_path_traite = os.path.join(adresse_save_result,name_traite)   #   get the save path
                            sitk.WriteImage(img,save_path)
                            sitk.WriteImage(image_newMatrice,save_path_traite) 
                            timeRMR2 = time.time()
                            TimeForrunFunctionRMR2 = timeRMR2 - timeRMR1
                            print(u"La fonction de traitement s'est executée en " + str(TimeForrunFunctionRMR2) +" secondes")
                            print("\n")
                        except RuntimeError:
                            print ("--> Probleme avec l'importation et/ou le triatement d'image")
                except RuntimeError:
                    print ("--> Probleme avec la lecture des metadata")
    print("\n")
    print("Nombre d'image total lue:"+str(Nimageouverte)+"\n")
    print("Nombre d'image total traité:"+str(Nimagetraitees)+"\n" )
    timefinal = time.time()
    TimeTotal = timefinal - timeInit
    print(u"Le traitement de l'ensemble des données c'est executée en " + str(TimeTotal) +" secondes")

##################################Execution du code############################################
###############################################################################################  
main(data_directory, adresse_save_result,spacing_ratio)