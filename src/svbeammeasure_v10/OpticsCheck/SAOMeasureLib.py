# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 02:01:21 2021

@author: Yang Hyo

update history
---------
2021-02-02  YK created this library for SAO imaging quality check.
            1. This is the middle layer (application specific data analysis).
            2. Run the common foundation library (BeadCalibLib)
                on multiple cores (multiprocessing).
                
2021-03-25  YK changed the following.
            1. Shortcut with relative address generates permission error
                for spawned processes (multiprocessing) to write files on the server.
            2. Change the location to save the multiprocessing result
                from the exe file folder (server) to lsgd file folder (local computer).
"""
import numpy as np
import os # directory, folder

# RBI library
import BeadCalibLib
from BeadCalibLib import SelectIsolatedTargets_Multi2

import multiprocessing # Utilize a multi-core CPU fore speed-up

import math
    
def findLocMulti(FolderPathLsgd,
            FileTitleLsgd,
            OnePixelFallOff,
            TwoPixelFallOff):
    """
    Find Isolated Single Bead with multiprocessing (multiple-cores). 

    Parameters
    ----------
    FolderPathLsgd : Path
        Raw data (lsgd files) directory.
    FileTitleLsgd : str
        lsgd file name (for example, PREVIEW_L488).
    OnePixelFallOff : float
        Parameter for single bead detection.
    TwoPixelFallOff : float
        Parameter for single bead detection.

    Returns
    -------
    mLocation : ndarray
        N (number of locations) x 2 (row, col) array.
    """
    FileNameLsgd = FileTitleLsgd + '.lsgd'
    FullPathLsgd = FolderPathLsgd.joinpath(FileNameLsgd)
    
    # Binary file to 12 x 2048 x 2048 ndarray
    ImgstInputLsgd = BeadCalibLib.LsgdToImageStack(FullPathLsgd)
    # 12 x 2048 x 2048 to single 2048 x 2048 image
    ImgAvg = np.mean(ImgstInputLsgd, axis=0)
    
    sBackgroundLevel = BeadCalibLib.EstimateBackgroundNoise(ImgAvg)
    
    # # N (number of locations) x 2 (row, col) array
    # # (row, col) is the location inside the image array.
    # mLocation = BeadCalibLib.SelectIsolatedTargets(ImgAvg,
    #                                                sBackgroundLevel,
    #                                                OnePixelFallOff,
    #                                                TwoPixelFallOff)
    
    groups, nGroups = BeadCalibLib.SelectIsolatedTargets_Multi1(ImgAvg,
                                                                sBackgroundLevel)
    numCPU = multiprocessing.cpu_count()
    
    pList = []
    for idx in range(numCPU):
        pList.append(multiprocessing.Process(target=SelectIsolatedTargets_Multi2,
                                              args=(FolderPathLsgd,
                                                    ImgAvg,
                                                    sBackgroundLevel,
                                                    OnePixelFallOff,
                                                    TwoPixelFallOff,
                                                    groups,
                                                    nGroups,
                                                    numCPU,
                                                    idx)))
        pList[idx].start()
    
    for idx in range(numCPU):
        pList[idx].join()
        
    print('multiprocessing end for ' + FileTitleLsgd)
    print()
    
    for idx in range(numCPU):
        locIntFileName = FolderPathLsgd.joinpath( 'locIntCandidates' + str(idx) +'.npy' )
        locInt = np.load(locIntFileName)
        os.remove(locIntFileName)
        
        location = locInt[:,:2]
        intensity = locInt[:,2]
        
        if idx == 0:
            locationCandidates = location
            intensityCandidates = intensity
        else:
            locationCandidates = np.concatenate((locationCandidates, location), axis=0)
            intensityCandidates = np.concatenate((intensityCandidates, intensity), axis=0)
            
    mLocation = BeadCalibLib.SelectIsolatedTargets_Multi3(locationCandidates, intensityCandidates)
            
    return mLocation

def analPeak(img, img0, mLocation):
    """
    Return a peak intensity ratio matrix between uniform illumination and reconstructed images.
    
    Parameters
    ----------
    img : ndarray
        Reconstructed image (high resolution, x4 times more pixels)
    img0 : ndarray
        Raw image (low resolution)      
    mLocation : ndarray
        N (number of locations) x 2 (row, col) array.
        Location in the reconstructed image pixel coordinate
    
    Returns
    -------
    peakRelArray : ndarray
    """
    # Locations of the single bead are calculated from the U-image (uniform illumination, low resolution).
    # Then multiplied by 4 to match the pixel numbers of the HF-image (reconstructed, high resolution, more pixesl).

    numSaturatedBeads = 0
    peakRelArray = np.zeros((len(mLocation),1))
        
    for ind in range( len(mLocation) ):
        loc = mLocation[ind]
        
        # mLocation is the HF-image coordinate location based on the U-image (less pixels).
        # +-4 pixel sub region to find the exact peak position in the HF-image (more pixels).
        imgSub = img[loc[0]-4:loc[0]+5, loc[1]-4:loc[1]+5]
        
        deltaRow, deltaCol = findPeakLoc(imgSub, loc)
        row = loc[0] + deltaRow
        col = loc[1] + deltaCol
        
        imgSubNew = img[row-4:row+5, col-4:col+5]
        
        # Location of the bead in the U-image
        row0 = int(loc[0] / 4)
        col0 = int(loc[1] / 4)
        
        # peakRelArray[ind] = np.max(imgSubNew) / img0[row0,col0]
        imgSub = img[loc[0]-4:loc[0]+8, loc[1]-4:loc[1]+8]
        peakRelArray[ind] = np.max(imgSub) / img0[row0,col0]
        
    #     if img0[row0,col0] > (2**15-1):
    #         numSaturatedBeads += 1
            
    # print('numSaturatedBeads: ' + str(numSaturatedBeads))
        
    return peakRelArray

def findPeakLoc(img, loc):
    """
    Find the peak intensity location (row, col). 

    Parameters
    ----------
    img : 2D array
        N x N sub-image containing a single peak
        
    Returns
    -------
    (rowPeak, colPeak) : tuple
        location (row, col) of the peak intensity. (0, 0) is the center.
    """
    
    subMaxLoc = np.argmax(img)
    
    (row, col) = np.unravel_index(subMaxLoc, img.shape)
    
    # Coordinate conversion for (0, 0) center
    row_center = math.floor(img.shape[0] / 2)
    col_center = math.floor(img.shape[1] / 2)
    
    rowPeak = row - row_center
    colPeak = col - col_center
    
    return (rowPeak, colPeak)

if __name__ == "__main__":
    from pathlib import Path # path, directory, file name, file extension
    import glob # file check
    
    InFilePath = Path(r'D:\Optical Biosystems\Regular work\RND\2020-11-30 SAO measure\Data')
    
    # glob.glob() returns a list of string path names that match pathname
    lsgdPath = InFilePath.joinpath('AcqData', '*.lsgd')
    fList = glob.glob(str(lsgdPath))
    
    fPath = fList[0]
        
    FolderPathLsgd = InFilePath.joinpath('AcqData')             
    FileTitleLsgd = Path(fPath).stem
    
    OnePixelFallOff = 0.62
    TwoPixelFallOff = 0.22
    
    # Find Isolated Single Bead 
    mLocation = findLocMulti(FolderPathLsgd,
                             FileTitleLsgd,
                             OnePixelFallOff,
                             TwoPixelFallOff)
    
    print(mLocation.shape)
    print(mLocation[0])
    print(mLocation[-1])
    
    input('The end..')