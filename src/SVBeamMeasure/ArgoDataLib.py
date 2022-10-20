# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 13:07:34 2020

@author: YK from Rebus Biosystems

update history
---------
2020-11-03  YK changed the following.
            1. added more comments
            1. update to differentiate different Argolight slides
            
"""

import math
import numpy as np
import cv2 # read tif file
from scipy.ndimage import gaussian_filter
from pathlib import Path

# fft
import scipy.fftpack as fftim

# open, merge, save png files
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# %% Analyze ArgoLight data

def argoAnal(InFilePath):
    """
    Call different function according to file type (tif vs lsgd)
    
    Parameters
    ---------
    InFilePath : pathlib.Path obj, path for the input data file
    """
    if InFilePath.suffix == '.tif':
        result = argoTif(InFilePath)
    elif InFilePath.suffix == '.lsgd':
        result = argoNPh(InFilePath)
    else:
        result = 0
        
    return result 
    
def argoTif(InFilePath):
    """
    tif file has 1 image inside.
    
    Parameters
    ---------
    InFilePath : pathlib.Path obj, path for the input data file      
    """
    imAvg = cv2.imread(str(InFilePath), cv2.IMREAD_UNCHANGED)
    result = argoSnapZ(imAvg)
    return result
        
def argoNPh(InFilePath):
    """
    lsgd file (binary) has 12 images (4 images x 3 phases) inside.
    
    Parameters
    ---------
    InFilePath : pathlib.Path obj, path for the input data file      
    """
    # Check Lsgd file size
    InFilePath = Path(InFilePath)
    
    FileInfoBytes = InFilePath.stat().st_size
    # https://stackoverflow.com/questions/2104080/how-can-i-check-file-size-in-python
    
    if (FileInfoBytes == 2560 * 2160 * 12 * 2):
        ImageRowSize = 2160
        ImageColSize = 2560
    elif (FileInfoBytes == 2048 * 2048 * 12 * 2):
        ImageRowSize = 2048
        ImageColSize = 2048
    else:
        print('\n[Error] wrong Lsgd size\n\n')
        ImageStack = np.empty( shape=(0, 0) )
        return ImageStack
    
    # Load images
    with open(InFilePath, mode='rb') as file: # b is important -> binary
        rawData = np.fromfile(InFilePath, dtype='uint16')
        # https://stackoverflow.com/questions/39762019/how-to-read-binary-files-in-python-using-numpy
    
    imStack = rawData.reshape((12, ImageRowSize, ImageColSize))
    # Matlab Python difference (more explanation needed)
    
    # Generate result (average image, normalise value, and slide)
    iChs = range(1, 4+1)
    iCntPh = 3
    
    resultData = []
    resultNormVal = []    
    
    for iCh in iChs:
    
        iIdxStartFrame = (iCh - 1) * iCntPh
        iIdxEndFrame = iIdxStartFrame + iCntPh
    
        # averaged image for 3 phases
        imAvg = np.average(imStack[iIdxStartFrame:iIdxEndFrame,:], axis=0)
        data, normVal, slide = argoSnapZ(imAvg)
        resultData.append(data)
        resultNormVal.append(normVal)
        
    return resultData, resultNormVal, slide

def argoSnapZ(imAvg):
    """
    Return laser beam heatmap from a raw image.
    
    Parameters
    ---------
    imAvg : ndarray, raw image      
    """
    dPixelUm = 0.325
    SpotSpacingUm_LM = 20
    SpotSpacingUm_Homo = 15
    ImageRowSize, ImageColSize = imAvg.shape
    
    # Distinguish between LM and Homo slides (pixel coordinate)
    imAvgShifted = fftim.fftshift(imAvg)
    imAvgShiftedF = abs(fftim.fft2(imAvgShifted))
    
    period_LM = SpotSpacingUm_LM / dPixelUm
    period_Homo = SpotSpacingUm_Homo / dPixelUm
    
    freq_LM = 1/period_LM
    freq_Homo = 1/period_Homo
    
    deltaFreq = 1/ImageColSize
    
    freqIdx_LM = int( np.round( freq_LM / deltaFreq ) )
    freqIdx_Homo = int( np.round( freq_Homo / deltaFreq ) )

    # (0, 0) is the origin    
    if imAvgShiftedF[0, freqIdx_LM] > imAvgShiftedF[0, freqIdx_Homo]:
        slide = 'LM'
    else:
        slide = 'Homo'

    # Input
    if slide == 'LM': 
        dSpotSpacingUm = 20
    elif slide == 'Homo':
        dSpotSpacingUm = 15
    dSpotSpacingPix = dSpotSpacingUm / dPixelUm
    
    GaussianFilterSize = 3
    GaussianSigma = 0.5
    GaussianTruncate = (((GaussianFilterSize - 1)/2)-0.5)/GaussianSigma
    # https://stackoverflow.com/questions/25216382/gaussian-filter-in-scipy
    
    # Crop size, Number of grid spots
    if slide == 'LM':
        iCropSizeRow, iCropSizeCol = imAvg.shape
        
        iCntGridInX = math.floor((iCropSizeCol/2) / dSpotSpacingPix) * 2 + 1
        iCntGridInY = math.floor((iCropSizeRow/2) / dSpotSpacingPix) * 2 + 1
    elif slide == 'Homo':
        iCntGridInX = 39
        iCntGridInY = 39
        
        iCropSizeRow = math.floor((iCntGridInY + 1)*dSpotSpacingUm/dPixelUm)
        iCropSizeCol = math.floor((iCntGridInX + 1)*dSpotSpacingUm/dPixelUm)
    
    
    # Low pass filter and crop image
    
    imAvgResized = imAvg
    
    imAvgResized = gaussian_filter(imAvg, sigma=GaussianSigma, truncate=GaussianTruncate)
    
    iSizeRow, iSizeCol = imAvgResized.shape
    
    iCropRowStart = int( (iSizeRow - iCropSizeRow)/2 + 1 )
    iCropColStart = int( (iSizeCol - iCropSizeCol)/2 + 1 )
    
    imCrop = imAvgResized[iCropRowStart-1:iCropRowStart+iCropSizeRow-1,
                          iCropColStart-1:iCropColStart+iCropSizeCol-1]
    
    # Get intensities of all grid spot
    
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
    
    # Make Heat-map
    GridZSort = GridZ.flatten()
    GridZSort.sort()
    GridZNormVal = GridZSort[-10]
    GridZNorm = GridZ / GridZNormVal
    # https://stackoverflow.com/questions/33181350/quickest-way-to-find-the-nth-largest-value-in-a-numpy-matrix/43171216
    
    Idx = np.nonzero(GridZNorm > 1)
    GridZNorm[Idx] = 1.0
    
    return GridZNorm, GridZNormVal, slide

# %% merge image files
# https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python

def checkCh(InFilePath):
    """
    Check if all 4 channel files exist
    
    Parameters
    ---------
    InFilePath : pathlib.Path obj, path for the input data file      
    """
    fname = InFilePath.name
    result = True
    
    # check file name with Ch1 ~ Ch4
    for ind_ch in range(4):
        chPos = fname.find('Ch')
        findPath = fname.replace(fname[chPos:chPos+3], fname[chPos:chPos+2]+str(ind_ch+1))
        findPath = InFilePath.parent.joinpath(findPath)
        
        if findPath.is_file():
            pass
        else:
            result = False
        
    return result

def mrgRefTest(InFilePath):
    """
    Merge result images files for reference and test arms separately measured data
    
    Parameters
    ---------
    InFilePath : pathlib.Path obj, path for the input data file
    
    Returns
    ---------
    resultImg : ndarray, merged image
    """
    Img1 = mrgHor(InFilePath)
    
    fname = InFilePath.name
    
    if fname.find('Ref') != -1:
        fname2 = fname.replace('Ref', 'Test')
    else:
        fname2 = fname.replace('Test', 'Ref')
    InFilePath2 = InFilePath.parent.joinpath(fname2)
    
    Img2 = mrgHor(InFilePath2)
    
    resultImg = mrgVer(Img1, Img2)
    
    return resultImg    
    
def mrgHor(InFilePath):
    """
    Merge result images files horizontally
    
    Parameters
    ---------
    InFilePath : pathlib.Path obj, path for the input data file
    
    Returns
    ---------
    new_im : ndarray, merged image
    """
    fname = InFilePath.name
    pathOpen = []
    
    for ind_ch in range(4):
        chPos = fname.find('Ch')
        fileOpen = fname.replace(fname[chPos:chPos+3], fname[chPos:chPos+2]+str(ind_ch+1))
        pathOpen.append( InFilePath.parent.joinpath(fileOpen) )
        
    images = [Image.open(str(x)) for x in pathOpen]
    widths, heights = zip(*(i.size for i in images))
    
    total_width = sum(widths)
    max_height = max(heights)
    
    new_im = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]
    
    return new_im

def mrgVer(Img1, Img2):
    """
    Merge result images files vertically
    
    Parameters
    ---------
    InFilePath : pathlib.Path obj, path for the input data file
    
    Returns
    ---------
    new_im : ndarray, merged image
    """
    images = [Img1, Img2]
    widths, heights = zip(*(i.size for i in images))
    
    max_width = max(widths)
    sum_height = sum(heights)
    
    new_im = Image.new('RGB', (max_width, sum_height))
    
    y_offset = 0
    for im in images:
      new_im.paste(im, (0,y_offset))
      y_offset += im.size[1]
    
    return new_im

def fnameOut(InFilePath):
    """
    Returns merged output file name by removing indiviual file info string
    
    Parameters
    ---------
    InFilePath : pathlib.Path obj, path for the input data file
    
    Returns
    ---------
    OutFilePath : pathlib.Path obj, path for the output data file
    """
    fname = InFilePath.name
    
    for ind_ch in range(4):
        fname = fname.replace(' Ch'+str(ind_ch+1)+',', "")
        fname = fname.replace('_Ch'+str(ind_ch+1)+',', "")
    
    fname = fname.replace(' Ref,', "")
    fname = fname.replace(' Test,', "")
    
    OutFilePath = InFilePath.parent.joinpath(fname)
    
    return OutFilePath