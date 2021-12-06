# coding: utf-8

import unittest
from slicer.ScriptedLoadableModule import *
import logging
from __main__ import vtk, qt, ctk, slicer
from math import *
import numpy as np
from vtk.util import numpy_support
import SimpleITK as sitk
import sitkUtils as su
import time
import codecs
import datetime
import vtkSegmentationCorePython as vtkSegmentationCore
import pydicom
import sys, time, os

############################si il manque une biblio exemple scikit #################################
#slicer.util.pip_install("scikit-image")
#########################################################
import skimage
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
from numpy.fft import fftshift, ifftshift, fftn, ifftn


data_directory = ""
adresse_save_result=""
type_image='float'
couplage=0.01
PercentSigmaIncrease=0.01

def transform_kspace_to_image(k, dim=None, img_shape=None):
    """ Computes the Fourier transform from k-space to image space
    along a given or all dimensions
    :param k: k-space data
    :param dim: vector of dimensions to transform
    :param img_shape: desired shape of output image
    :returns: data in image space (along transformed dimensions)
    """
    if not dim:
        dim = range(k.ndim)
    img = fftshift(ifftn(ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
    img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    return img


def transform_image_to_kspace(img, dim=None, k_shape=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions
    :param img: image space data
    :param dim: vector of dimensions to transform
    :param k_shape: desired shape of output k-space data
    :returns: data in k-space (along transformed dimensions)
    """
    if not dim:
        dim = range(img.ndim)
    k = fftshift(fftn(ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
    k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k


def cropImage(image, label):
    stats= sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(label,image)
    delta=0 #extention du label pour eviter les problemes aux bords
    LowerBondingBox=[stats.GetBoundingBox(1)[0]-delta,stats.GetBoundingBox(1)[1]-delta,stats.GetBoundingBox(1)[2]-delta]
    UpperBondingBox=[image.GetSize()[0]-(stats.GetBoundingBox(1)[0]+stats.GetBoundingBox(1)[3]+delta),image.GetSize()[1]-(stats.GetBoundingBox(1)[1]+stats.GetBoundingBox(1)[4]+delta),image.GetSize()[2]-(stats.GetBoundingBox(1)[2]+stats.GetBoundingBox(1)[5]+delta)]
    image_select=cropImagefctLabel(image, LowerBondingBox, UpperBondingBox  )
    return image_select



def getSec(s):
    b =int(s[0:2]) *3600 + int(s[2:4])*60 + int(s[5:6])
    return b

def increase_noise_inKspace(imageMRI, type_image, couplage, PercentSigmaIncrease): 
    ###############convertie en format numpy
    imageMRI_np=sitk.GetArrayFromImage(imageMRI)
    imageMRI_np=imageMRI_np.astype(type_image)
    ######la transform dans l'espace k#########
    imageMRIkspace_np=transform_image_to_kspace(imageMRI_np)
    ############################## extraction des partie reel et imaginaire
    imageMRIkspace_np_real=imageMRIkspace_np.real
    imageMRIkspace_np_imaginary=imageMRIkspace_np.imag
    #####valeur de ref###########################################################
    sigma_est_reel = estimate_sigma(imageMRIkspace_np_real, multichannel=False, average_sigmas=True)
    sigma_est_im=estimate_sigma(imageMRIkspace_np_imaginary, multichannel=False, average_sigmas=True)
    rapport_des_beta= np.sum(abs(imageMRIkspace_np_real))/np.sum(abs(imageMRIkspace_np_imaginary)) #supose que le couplage depend du module de l'image
    amplitudeMaxImage=np.max(abs(imageMRIkspace_np_real+1j*imageMRIkspace_np_imaginary))
    ##########rajoute du bruit dans les deux partie
    beta_total_image=couplage*amplitudeMaxImage
    beta_reel=beta_total_image/(1+rapport_des_beta)
    I2_simu_reel =imageMRIkspace_np_real+ (beta_reel*np.random.normal(0,PercentSigmaIncrease*sigma_est_reel,imageMRIkspace_np_real.shape))#je ne sais pas si j'ai raison de faire ca mais ca me semble plus logique
    beta_img=beta_total_image-beta_reel
    I2_simu_imaginaire =imageMRIkspace_np_imaginary+ (beta_img*np.random.normal(0,PercentSigmaIncrease*sigma_est_im,imageMRIkspace_np_imaginary.shape))
    #########reconstruction de l'image
    imageMRIkspace_np_I2=I2_simu_reel+1j*I2_simu_imaginaire
    imageMRI_np_I2=transform_kspace_to_image(imageMRIkspace_np_I2,range(imageMRIkspace_np_I2.ndim),imageMRI_np.shape) #sort des image dtype=complex
    imageMRI_np_I2=imageMRI_np_I2.astype(imageMRI_np.dtype) #convertion d'un format complex a un format image
    ########################reconvertie de numpy en image
    imageMRI_I2_simul=sitk.GetImageFromArray(imageMRI_np_I2, isVector=False)
    imageMRI_I2_simul.CopyInformation(imageMRI)
    ######################################################
    return imageMRI_I2_simul

def main(data_directory, adresse_save_result,type_image):
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
                            imageMRI = series_reader.Execute()  #importation de l'image
                            ##############IncreaseNoise##################
                            imageMRI_cast=sitk.Cast(imageMRI,sitk.sitkFloat64)
                            image_newNoise=increase_noise_inKspace(imageMRI_cast, type_image, couplage, PercentSigmaIncrease)
                            #su.PushToSlicer(image_newNoise,'test2',1)
                            ###########apply a threshold and convert to hold format##################
                            mask=sitk.OtsuThreshold(imageMRI,0,1)
                            mask=sitk.Cast(mask,sitk.sitkFloat64)
                            image_newNoise=image_newNoise*mask
                            image_newNoise = sitk.Cast(image_newNoise, sitk.sitkUInt16 ) 
                            ###############create the name of the output path and write##########
                            name=str(Nimagetraitees)+"_Noised.dcm"
                            save_path = os.path.join(adresse_save_result,name)   #   get the save path
                            sitk.WriteImage(image_newNoise,save_path) 
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
main(data_directory, adresse_save_result,type_image)
