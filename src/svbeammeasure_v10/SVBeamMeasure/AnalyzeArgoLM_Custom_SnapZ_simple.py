# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 13:07:34 2020

@author: YK
"""

import math
import numpy as np
import cv2 # read tif file
from scipy.ndimage import gaussian_filter


def argoLM(InDataPath):
    # %% Custom Argo-LM for StellarVision
    
    # %% Input
    # InDataPath = r'D:\Optical Biosystems\Regular work\RND\2020-07-14 Beam position\SN1021 (SV20C)\2020-07-02 Argilight_test'
    
    SN = '1021'
    Date = '2020-07-02'
    iCh = '4'
    RefOrTest = 'Test'
    Laser = '488'
    Power = '500'
    Exp = '200'
    
    Notes = ''
    
    InFileTitle = 'Capture'
    OutFileTitle = 'SN' + SN + ', ' + Date + ', Ch' + iCh + ', ' + RefOrTest
    OutFileTitle += ', ' + Laser + ' ' + Power + 'mw@' + Exp + 'ms ' + Notes
    
    dPixelUm = 0.325
    dSpotSpacingUm = 20
    dSpotSpacingPix = dSpotSpacingUm / dPixelUm
    
    GaussianFilterSize = 3
    GaussianSigma = 0.5
    GaussianTruncate = (((GaussianFilterSize - 1)/2)-0.5)/GaussianSigma
    # https://stackoverflow.com/questions/25216382/gaussian-filter-in-scipy
    
    
    # %% Read image
    
    # imAvg = cv2.imread(InDataPath + '\\' + InFileTitle + '.tif', cv2.IMREAD_UNCHANGED)
    imAvg = cv2.imread(InDataPath, cv2.IMREAD_UNCHANGED)
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
    
    OutFileName = '%s/%s.tif' % (InDataPath, OutFileTitle)
    
    
    # %% Crop size, Number of grid spots
    iCropSizeRow, iCropSizeCol = imAvg.shape
    
    iCntGridInX = math.floor((iCropSizeCol/2) / dSpotSpacingPix) * 2 + 1
    iCntGridInY = math.floor((iCropSizeRow/2) / dSpotSpacingPix) * 2 + 1
    
    
    # %% Make downsampling, LPF, crop image
    
    imAvgResized = imAvg
    
    imAvgResized = gaussian_filter(imAvg, sigma=GaussianSigma, truncate=GaussianTruncate)
    
    iSizeRow, iSizeCol = imAvgResized.shape
    
    iCropRowStart = int( (iSizeRow - iCropSizeRow)/2 + 1 )
    iCropColStart = int( (iSizeCol - iCropSizeCol)/2 + 1 )
    
    imCrop = imAvgResized[iCropRowStart-1:iCropRowStart+iCropSizeRow-1,
                          iCropColStart-1:iCropColStart+iCropSizeCol-1]
    
    OutFileName = '%s/%sAvgCrop.tif' % (InDataPath, OutFileTitle)
    
    
    # %% Get intensities of all grid spot
    
    GridZ = np.zeros((iCntGridInY, iCntGridInX))
    
    GridAreaWidthPix = (iCntGridInX-1) * dSpotSpacingUm / dPixelUm
    GridAreaHeightPix = (iCntGridInY-1) * dSpotSpacingUm / dPixelUm
    
    ULSpotX = (iCropSizeCol / 2) - (GridAreaWidthPix / 2)
    ULSpotY = (iCropSizeRow / 2) - (GridAreaHeightPix / 2)
    
    WindowSize = 20
    WindowWidth = 2*WindowSize+1
    
    imSpotStitch = np.zeros((iCntGridInY*WindowWidth, iCntGridInX*WindowWidth), dtype=np.uint8)
    
    for i in range(iCntGridInY):
        for j in range(iCntGridInX):
    
            CurrPatternPosX = int(ULSpotX + dSpotSpacingPix * j) - 1
            CurrPatternPosY = int(ULSpotY + dSpotSpacingPix * i) - 1
    
            imWnd = imCrop[CurrPatternPosY-WindowSize : CurrPatternPosY+WindowSize+1,
                           CurrPatternPosX-WindowSize : CurrPatternPosX+WindowSize+1]
    
            MaxIntVal = np.max(imWnd)
            GridZ[i,j] = MaxIntVal
            
            imSpotStitch[i*WindowWidth:(i+1)*WindowWidth,
                         j*WindowWidth:(j+1)*WindowWidth] = (imWnd / MaxIntVal * 255).astype(np.uint8)
    
    # plt.figure;
    # plt.imshow(imSpotStitch);
    
    
    # %% Make Heat-map
    GridZSort = GridZ.flatten()
    GridZSort.sort()
    GridZNorm = GridZ / GridZSort[-10]
    # https://stackoverflow.com/questions/33181350/quickest-way-to-find-the-nth-largest-value-in-a-numpy-matrix/43171216
    
    Idx = np.nonzero(GridZNorm > 1)
    GridZNorm[Idx] = 1.0
    
    return GridZNorm