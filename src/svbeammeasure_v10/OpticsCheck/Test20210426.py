# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 09:38:13 2021

@author: Yang Hyo
"""
from pathlib import Path # path, directory, file name, file extension

import numpy as np
import matplotlib.pyplot as plt
import cv2 # read tif file

import BeadCalibLib

InFilePath = Path(r'D:\Optical Biosystems\Regular work\RND')
InFilePath = InFilePath.joinpath(r'2020-11-30 SAO measure\Data\Sys 22\2020-10-29_0.2um Bead slide 01')

FullPathLsgd = InFilePath.joinpath(r'AcqData\PREVIEW_L532.lsgd')

##### reconstructed tif file (8192 x 8192
HFImageFilePath = InFilePath.joinpath(r'SaoRecon_8192\PREVIEW_L532_HF.tif')

# Binary file to 12 x 2048 x 2048 ndarray
ImgstInputLsgd = BeadCalibLib.LsgdToImageStack(FullPathLsgd)
# 12 x 2048 x 2048 to single 2048 x 2048 image
img0 = np.mean(ImgstInputLsgd, axis=0)

img = cv2.imread(str(HFImageFilePath), cv2.IMREAD_UNCHANGED)
img = img.astype('float64')

### result

def showResult(img0, img, loc0, loc):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(str(FullPathLsgd.name),
                 fontsize=24)                
    axImg = axs[0,0].imshow(img0,
                            cmap='jet',
                            vmin=0, vmax=2**16-1)
    axs[0,0].set_xlim(loc0[0]-4, loc0[0]+4)
    axs[0,0].set_ylim(loc0[1]-4, loc0[1]+4)
    fig.colorbar(axImg, ax=axs[0,0])
    axs[0,0].set_title('U-image',
                       fontsize=18)
    
    axImg = axs[0,1].imshow(img,
                            cmap='jet',
                            vmin=0, vmax=2**16-1)
    axs[0,1].set_xlim(loc[0]-16, loc[0]+16)
    axs[0,1].set_ylim(loc[1]-16, loc[1]+16)
    fig.colorbar(axImg, ax=axs[0,1])
    axs[0,1].set_title('HF-image',
                       fontsize=18)
    
    axImg = axs[1,0].plot(img0[loc0[1], loc0[0]-4:loc0[0]+5],
                          marker='o')
    axs[1,0].set_title('Horizontal cross-section',
                       fontsize=18)
    
    axImg = axs[1,1].plot(img[loc[1], loc[0]-16:loc[0]+17],
                          marker='o')
    axs[1,1].set_title('Horizontal cross-section',
                       fontsize=18)
    
###
loc0 = 1291, 1039
loc = 4*loc0[0]+2, 4*loc0[1]+1

showResult(img0, img, loc0, loc)

###
loc0 = 667, 967
loc = 4*loc0[0]-1, 4*loc0[1]+1

showResult(img0, img, loc0, loc)