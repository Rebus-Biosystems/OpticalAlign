# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 2020

@author: YK from OBI

update history
---------
2020-09-15  YK converted the folloing Matlab codes into Python.

            \\ma2files\Production systems\Production Tools\MATLAB sys 21\FullCalibStep2.1Temp\Lsgd
                1. LsgdToImageStack.m
                
            \\ma2files\Production systems\Production Tools\MATLAB sys 21\FullCalibStep2.1Temp\VtoPhMatlabGUI\VoltageToPhaseMappingToolbox
                2. EstimateBackgroundNoise.m
                3. SelectIsolatedTargets.m   
                
2020-09-18  YK added comments.

2021-01-21  YK added Multiprocessing version of SelectIsolatedTargets function.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Algorithm to find isolated single beads
from skimage.feature import peak_local_max
from skimage.measure import label


def LsgdToImageStack(InFilePath):
    """
    Return an image stack of 12 images from a lsgd file.

    Parameters
    ----------
    InFilePath : string or Path object
        Raw data (lsgd files) directory.

    Returns
    -------
    imStack : 12 x ImageRowSize x ImageColSize ndarray
        2D image stack of 12 images.

    """
    InFilePath = Path(InFilePath)
    
    # Check Lsgd file size
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
    
    return imStack
    
def EstimateBackgroundNoise(image2D):
    """
    Return estimated black background intensity value from the camera.

    Parameters
    ----------
    image2D : ndarray
        2D image array.

    Returns
    -------
    backgroundNoise : float
        Intensity for the no bead region.

    """
    # Get the minimum intensity levels along each row of the image
    minImage = np.min(image2D, axis=0)
    
    # Extract the statistics of the distribution of these min values
    avgMinImage = np.mean(minImage)
    stdMinImage = np.std(minImage)
    if avgMinImage == 0:
        # error('Unable to estimate background noise level')
        pass
        
    # Check whether this represents a tight grouping, or whether there might be
    # outliers that are biasing our results
    measSpread = stdMinImage / avgMinImage
    
    if measSpread > 0.05:
        # Exclude outliers, i.e., all points greater than 2xStdDev away from
        # the mean.
        minImage = minImage[abs(minImage - avgMinImage) < 2*stdMinImage]

    # Compute the expected background noise
    backgroundNoise = np.mean(minImage) + np.std(minImage)

    return backgroundNoise

def SelectIsolatedTargets(image, backgroundLevel, OnePixelFallOff, TwoPixelFallOff):
    """
    Return an Nx2 array of isolated single bead location from an input image.

    Parameters
    ----------
    image : ndarray
        2D image array.
    backgroundLevel : float
        no bead region intensity.
    OnePixelFallOff : float
        Parameter for single bead detection.
    TwoPixelFallOff : float
        Parameter for single bead detection.

    Returns
    -------
    locationTargets : ndarray
        N (number of locations) x 2 (row, col) array.
        (row, col) is the location of isolated single bead.

    """
    THRESHOLD_SIGNAL_TO_BACKGROUND = 20
    RATIO_ONE_PIXEL_FALL_OFF = OnePixelFallOff
    RATIO_TWO_PIXEL_FALL_OFF = TwoPixelFallOff
    RADIUS_BEAD_NEIGHBORHOOD = 2
    DEBUG_DISPLAY = 0
    
    # Get the size of the image
    szImage = image.shape
    
    # Threshold the input image under assumption that most signal-valued 
    # pixels are at least 'THRESHOLD_SIGNAL_TO_BACKGROUND' times as bright as 
    # background intensity.
    imageMask1 = image > backgroundLevel * THRESHOLD_SIGNAL_TO_BACKGROUND
    
    conn_8 = np.ones((3,3))
    imageMask2 = peak_local_max(image, footprint=conn_8, indices=False, exclude_border=0)
    # https://stackoverflow.com/questions/27598103/what-is-the-difference-between-imregionalmax-of-matlab-and-scipy-ndimage-filte
    
    imageMask = np.logical_and(imageMask1, imageMask2)
    
    # plt.figure(); plt.imshow(imageMask); plt.axis('equal')

    # Label all the connected pixels.
    # Using '4 connectivity' to separate out possible double beads.
    groups, nGroups = label(imageMask, return_num=True, connectivity=1)
    # https://stackoverflow.com/questions/53089020/what-is-equivalent-to-bwlabeln-with-18-and-26-connected-neighborhood-in-pyth
    # label() returns labels (ndarray of dtype int) and num (int, optional)
    # labels and num starts from 1
    
    # Initialize the vectors that contain candidate target's coordinates
    # 1st col = row coordinate
    # 2nd col = col coordinate    
    locationCandidates = np.zeros((nGroups, 2))
    
    # Vector containing candidate target's intensity value.
    intensityCandidates = np.zeros((nGroups, 1))
    
    # Initialize the counter that keeps track of candidate targets
    numCandidates = 0
    
    ## Test each targets and pick isolated targets
    
    # For each group of connected pixels, we have to test whether said
    # group represents an isolated target.  Test is done by: (1) first,
    # extracting a 5x5 swatch from the original image for each connected 
    # pixel group, centered at the pixel with highest intensity; then,
    # (2) using this swatch to determine: 1) whether group is a good 
    # candidate for characterization; and 2) its exact location.
    for iLabel in range(1, nGroups+1):
        # Get the coordinates of the ith collecion of connected pixels
        rowsPixelGroup, colsPixelGroup = np.nonzero(groups==iLabel)
    
        # Get the intensity values for every element of this group.
        intensitiesPixelGroup = image[groups==iLabel] - backgroundLevel
    
        # Within this group, get the max value and its corresponding index.
        intensityMaxPixel = np.max(intensitiesPixelGroup);
        indexMaxPixel = np.argmax(intensitiesPixelGroup)
    
        # Use this index to find the coordinates of the max-valued pixel.
        rowMaxPixel = rowsPixelGroup[indexMaxPixel]
        colMaxPixel = colsPixelGroup[indexMaxPixel]
    
        # Discard result if coordinates of the peak is too close to the 
        # image boundary.
        if (rowMaxPixel < RADIUS_BEAD_NEIGHBORHOOD) or \
           (rowMaxPixel >= (szImage[0] - RADIUS_BEAD_NEIGHBORHOOD)) or \
           (colMaxPixel < RADIUS_BEAD_NEIGHBORHOOD) or \
           (colMaxPixel >= (szImage[1] - RADIUS_BEAD_NEIGHBORHOOD)):
            # Skip to the next iteration
            continue
    
        # Else, extract the swatch from the image in the neighborhood of
        # the peak-valued pixel.
        swatch = image[rowMaxPixel-RADIUS_BEAD_NEIGHBORHOOD:\
                       rowMaxPixel+RADIUS_BEAD_NEIGHBORHOOD+1,
                       colMaxPixel-RADIUS_BEAD_NEIGHBORHOOD:\
                       colMaxPixel+RADIUS_BEAD_NEIGHBORHOOD+1]
    
        # --------------------------------------------------------------- %
        # Exclude clumped beads, i.e., in the case of an isolated bead, the 
        # pixel values should start falling off as we move away from the 
        # center (or peak-valued) pixel.
    
        # 1. Test for pixels immediately bordering the center pixel.
        #
        # (This test #1 may be redundant due to #2 - JPark, 11am May 04, 2009)
        # (Although not as effective as #2, we still need this test with Jaya
        #   - JPark, 12:22pm May 04, 2009) 
        threshOnePixelAway = backgroundLevel + intensityMaxPixel * \
                             RATIO_ONE_PIXEL_FALL_OFF
        if np.sum(swatch > threshOnePixelAway) > 1:
            # Detected multiple pixels with values > threshold value
            continue   # Skip
        
        # 2. Test for pixels at a distance of at least 2 away from the 
        #    center pixel.
        threshTwoPixelsAway = backgroundLevel + intensityMaxPixel * \
                              RATIO_TWO_PIXEL_FALL_OFF
        iEdge = RADIUS_BEAD_NEIGHBORHOOD*2
        
        noOutliers = ( np.sum(swatch[0,:] > threshTwoPixelsAway) +      # top-most edge
                      np.sum(swatch[iEdge,:] > threshTwoPixelsAway) +   # bottom-most edge
                      np.sum(swatch[:,0] > threshTwoPixelsAway) +       # left-most edge
                      np.sum(swatch[:,iEdge] > threshTwoPixelsAway) )   # right-most edge
    
        if noOutliers > 0:
            continue     # Skip
        
        # END exclude clumped beads
        # --------------------------------------------------------------- %
    
        # Okay, we've candidate targets, so increment the counter and
        # populate vectors containing candidate's intensity & coordinates.
        locationCandidates[numCandidates,:] = [rowMaxPixel, colMaxPixel]                        
        intensityCandidates[numCandidates] = intensityMaxPixel
        numCandidates = numCandidates + 1
        
    # Prune the book-keeping vectors to include only candidate targets.
    locationCandidates = locationCandidates[:numCandidates,:]
    intensityCandidates = intensityCandidates[:numCandidates,:]
    
    # Exclude beads that are too bright (i.e., 20% brighter than
    # average targets), which may indicate more than 1 target within
    # one CCD pixel.
    iOkTargets = np.nonzero(intensityCandidates < 1.2*np.mean(intensityCandidates))
    iOkTargets = iOkTargets[0]
    # np.nonzero returns (row,col) tuple and intensityCandidates is an 1D data.
    locationTargets = locationCandidates[iOkTargets,:]

    ## Display for debugging
    if DEBUG_DISPLAY:
        plt.figure(); plt.imshow(image); plt.colorbar(); plt.set_cmap('jet')
        
        for IdxTgt in range(len(locationTargets)):
            plt.text(locationTargets[IdxTgt,1], locationTargets[IdxTgt,0], str(IdxTgt+1), color='white', fontsize=15)
       
    
        plt.title( 'Selected Isolated Targets; nTargets = '
                  + str(len(locationTargets)) )

    return locationTargets

def SelectIsolatedTargets_Multi1(image, backgroundLevel):
    """
    Rule out possible aggregated beads and pick up isolated single beads.

    Parameters
    ----------
    image : ndarray
        2D image array.
    backgroundLevel : float
        no bead region intensity.
    
    Returns
    -------
    groups : ndarray of dtype int
        Labeled array, where all connected regions are assigned the same integer value.
    nGroups : int
        Number of labels, which equals the maximum label index
    """
    # THRESHOLD_SIGNAL_TO_BACKGROUND = 20
    THRESHOLD_SIGNAL_TO_BACKGROUND = 10
    
    
    # Threshold the input image under assumption that most signal-valued 
    # pixels are at least 'THRESHOLD_SIGNAL_TO_BACKGROUND' times as bright as 
    # background intensity.
    imageMask1 = image > backgroundLevel * THRESHOLD_SIGNAL_TO_BACKGROUND
    
    conn_8 = np.ones((3,3))
    imageMask2 = peak_local_max(image, footprint=conn_8, indices=False, exclude_border=0)
    # https://stackoverflow.com/questions/27598103/what-is-the-difference-between-imregionalmax-of-matlab-and-scipy-ndimage-filte
    
    imageMask = np.logical_and(imageMask1, imageMask2)
    
    # plt.figure(); plt.imshow(imageMask); plt.axis('equal')

    # Label all the connected pixels.
    # Using '4 connectivity' to separate out possible double beads.
    groups, nGroups = label(imageMask, return_num=True, connectivity=1)
    # https://stackoverflow.com/questions/53089020/what-is-equivalent-to-bwlabeln-with-18-and-26-connected-neighborhood-in-pyth
    # label() returns labels (ndarray of dtype int) and num (int, optional)
    # labels and num starts from 1
    
    # # Uniform number density control
    # radiusControl = np.array( [20, 35, 60, 100, 181, 300, 500, 700] )
    # numberControl = 10 * 2**( np.array(range(0,8)) ) 
    # numberGroup = np.zeros(numberControl.shape)
    
    # groupsControl = np.zeros(groups.shape)
    # nGroupsControl = 0
    
    # # np.save('groups', groups)
    
    # # input('press any key')
    
    # # nGroups starts from 1
    # for iLabel in range(1, nGroups+1):
    # # for iLabel in range(101, 200):
    #     print(str(iLabel) + '/' + str(nGroups))
    #     # Get the coordinates of the ith collecion of connected pixels
    #     rowsPixelGroup, colsPixelGroup = np.nonzero(groups==iLabel)
    #     radiusGroup = np.sqrt(
    #         (rowsPixelGroup - 2048/2)**2 + (colsPixelGroup - 2048/2)**2 )
    #     radiusMean = np.mean(radiusGroup)
    #     print('radiusMean: ' + str(radiusMean))
    #     if radiusMean > radiusControl[-1]:
    #         idxControl = -1
    #     else:
    #         idxControl = np.argmax(radiusControl > radiusMean)
    #     if numberGroup[idxControl] <= numberControl[idxControl]:
    #         numberGroup[idxControl] += 1
    #         nGroupsControl += 1
    #         groupsControl[rowsPixelGroup, colsPixelGroup] = nGroupsControl
    
    # print(numberGroup)
    # print(nGroupsControl)
    # input('press any key')
    
    # return groupsControl, nGroupsControl
    return groups, nGroups

def SelectIsolatedTargets_Multi2(FolderPathLsgd, image, backgroundLevel, OnePixelFallOff, TwoPixelFallOff, groups, nGroups, numCPU, numProc):
    """
    Select isolated target beads. The longest running task (multiprocessing).

    Parameters
    ----------
    FolderPathLsgd : Path
        Raw data (lsgd files) directory.
    image : ndarray
        2D image array.
    backgroundLevel : float
        no bead region intensity.
    OnePixelFallOff : float
        Parameter for single bead detection.
    TwoPixelFallOff : float
        Parameter for single bead detection.
    groups : ndarray of dtype int
        Labeled array, where all connected regions are assigned the same integer value.
    nGroups : int
        Number of labels, which equals the maximum label index
    numCPU : int
        Number of CPUs in the system
    numProc : int
        Process number which determines the range of target beads to process.

    Returns
    -------
    None.

    """
    
    print('SelectIsolatedTargets_Multi2 strated for Proc #' + str(numProc))
    RATIO_ONE_PIXEL_FALL_OFF = OnePixelFallOff
    RATIO_TWO_PIXEL_FALL_OFF = TwoPixelFallOff
    RADIUS_BEAD_NEIGHBORHOOD = 2
    
    # Get the size of the image
    szImage = image.shape
    
    targetPeriod = int(nGroups /numCPU)
    targetInit = targetPeriod * numProc + 1
    if numProc != numCPU:
        targetEnd = targetPeriod * (numProc + 1)
    else:
        targetEnd = nGroups
        
    nGroupsLocal = targetEnd - targetInit + 1
    
    # Initialize the vectors that contain candidate target's coordinates
    # 1st col = row coordinate
    # 2nd col = col coordinate    
    locationCandidates = np.zeros((nGroupsLocal, 2))
    
    # Vector containing candidate target's intensity value.
    intensityCandidates = np.zeros((nGroupsLocal, 1))
    
    # Initialize the counter that keeps track of candidate targets
    numCandidates = 0
    
    ## Test each targets and pick isolated targets
    
    # For each group of connected pixels, we have to test whether said
    # group represents an isolated target.  Test is done by: (1) first,
    # extracting a 5x5 swatch from the original image for each connected 
    # pixel group, centered at the pixel with highest intensity; then,
    # (2) using this swatch to determine: 1) whether group is a good 
    # candidate for characterization; and 2) its exact location.
    for iLabel in range(targetInit, targetEnd+1):
        # Get the coordinates of the ith collecion of connected pixels
        rowsPixelGroup, colsPixelGroup = np.nonzero(groups==iLabel)
    
        # Get the intensity values for every element of this group.
        intensitiesPixelGroup = image[groups==iLabel] - backgroundLevel
    
        # Within this group, get the max value and its corresponding index.
        intensityMaxPixel = np.max(intensitiesPixelGroup);
        indexMaxPixel = np.argmax(intensitiesPixelGroup)
    
        # Use this index to find the coordinates of the max-valued pixel.
        rowMaxPixel = rowsPixelGroup[indexMaxPixel]
        colMaxPixel = colsPixelGroup[indexMaxPixel]
        
        ################
        radiusMaxPixel = np.sqrt(
            (rowMaxPixel -  2048/2)**2 + (colMaxPixel -  2048/2)**2 )
        addControl = 0
        if radiusMaxPixel > 1000:
            addControl = 0.08
        elif radiusMaxPixel > 900:
            addControl = 0.06
        elif radiusMaxPixel > 800:
            addControl = 0.04
        elif radiusMaxPixel > 700:
            addControl = 0.03
        elif radiusMaxPixel > 500:
            addControl = 0.02
        elif radiusMaxPixel > 300:
            addControl = 0.01    
        
        RATIO_ONE_PIXEL_FALL_OFF = OnePixelFallOff + addControl
        RATIO_TWO_PIXEL_FALL_OFF = TwoPixelFallOff + addControl
        
        # Discard result if saturated (the peak intensity too high)
        if intensityMaxPixel > (2**15-1):
            # Skip to the next iteration
            continue
        
        # Discard result if coordinates of the peak is too close to the 
        # image boundary.
        if (rowMaxPixel < RADIUS_BEAD_NEIGHBORHOOD) or \
            (rowMaxPixel >= (szImage[0] - RADIUS_BEAD_NEIGHBORHOOD)) or \
            (colMaxPixel < RADIUS_BEAD_NEIGHBORHOOD) or \
            (colMaxPixel >= (szImage[1] - RADIUS_BEAD_NEIGHBORHOOD)):
            # Skip to the next iteration
            continue
    
        # Else, extract the swatch from the image in the neighborhood of
        # the peak-valued pixel.
        swatch = image[rowMaxPixel-RADIUS_BEAD_NEIGHBORHOOD:\
                        rowMaxPixel+RADIUS_BEAD_NEIGHBORHOOD+1,
                        colMaxPixel-RADIUS_BEAD_NEIGHBORHOOD:\
                        colMaxPixel+RADIUS_BEAD_NEIGHBORHOOD+1]
    
        # --------------------------------------------------------------- %
        # Exclude clumped beads, i.e., in the case of an isolated bead, the 
        # pixel values should start falling off as we move away from the 
        # center (or peak-valued) pixel.
    
        # 1. Test for pixels immediately bordering the center pixel.
        #
        # (This test #1 may be redundant due to #2 - JPark, 11am May 04, 2009)
        # (Although not as effective as #2, we still need this test with Jaya
        #   - JPark, 12:22pm May 04, 2009) 
        threshOnePixelAway = backgroundLevel + intensityMaxPixel * \
                              RATIO_ONE_PIXEL_FALL_OFF
        if np.sum(swatch > threshOnePixelAway) > 1:
            # Detected multiple pixels with values > threshold value
            continue   # Skip
        
        # 2. Test for pixels at a distance of at least 2 away from the 
        #    center pixel.
        threshTwoPixelsAway = backgroundLevel + intensityMaxPixel * \
                              RATIO_TWO_PIXEL_FALL_OFF
        iEdge = RADIUS_BEAD_NEIGHBORHOOD*2
        
        noOutliers = ( np.sum(swatch[0,:] > threshTwoPixelsAway) +      # top-most edge
                      np.sum(swatch[iEdge,:] > threshTwoPixelsAway) +   # bottom-most edge
                      np.sum(swatch[:,0] > threshTwoPixelsAway) +       # left-most edge
                      np.sum(swatch[:,iEdge] > threshTwoPixelsAway) )   # right-most edge
    
        if noOutliers > 0:
            continue     # Skip
        
        # END exclude clumped beads
        # --------------------------------------------------------------- %
    
        # Okay, we've candidate targets, so increment the counter and
        # populate vectors containing candidate's intensity & coordinates.
        locationCandidates[numCandidates,:] = [rowMaxPixel, colMaxPixel]                        
        intensityCandidates[numCandidates] = intensityMaxPixel
        numCandidates = numCandidates + 1
        
    # Prune the book-keeping vectors to include only candidate targets.
    locationCandidates = locationCandidates[:numCandidates,:]
    intensityCandidates = intensityCandidates[:numCandidates,:]
    
    locIntCandidates = np.concatenate((locationCandidates, intensityCandidates), axis=1)
    
    filePath = FolderPathLsgd.joinpath( 'locIntCandidates' + str(numProc) )
    np.save(filePath, locIntCandidates)
    
    print('SelectIsolatedTargets_Multi2 ended for Proc #' + str(numProc))
    
def SelectIsolatedTargets_Multi3(locationCandidates, intensityCandidates):   
    """
    Uniform number density control

    Parameters
    ----------
    locationCandidates : ndarray
        N (number of locations) x 2 (row, col) array.
        (row, col) is the location of isolated single bead candidates.
    intensityCandidates : ndarray
        the intensitiy of isolated single bead candidates.

    Returns
    -------
    locationTargets : ndarray
        N (number of locations) x 2 (row, col) array.
        (row, col) is the location of isolated single bead.
    """
    
    numberDensity = 6
    gridSize = 16
    
    rowCandidates = locationCandidates[:,0]
    colCandidates = locationCandidates[:,1]
    
    rows = int(2048/gridSize) * np.array(range(1, gridSize+1))
    cols = rows
    
    locTargets = np.empty([0, 2])
    
    rowMin = 0
    
    for row in rows:
        rowSubIdx = np.nonzero( (rowCandidates >= rowMin) & (rowCandidates < row) )
        rowSubIdx = rowSubIdx[0]
        colSub = colCandidates[rowSubIdx]
        
        colMin = 0
        
        for col in cols:
            rowColRadius = np.sqrt((row/int(2048/gridSize) - 8.5)**2 + (col/int(2048/gridSize) - 8.5)**2)
            
            if rowColRadius < 9:            
                colSubIdx = np.nonzero( (colSub >= colMin) & (colSub < col) )
                colSubIdx = colSubIdx[0]
                numberData = len(colSubIdx)
                
                if numberData > 0:
                    if numberData >= numberDensity:
                        locSubIdx = rowSubIdx[ colSubIdx[:numberDensity] ]
                    else:
                        locSubIdx = rowSubIdx[ colSubIdx ]
                        
                    locSub = locationCandidates[locSubIdx,:]
                    locTargets = np.concatenate((locTargets, locSub), axis=0)
            
            colMin = col
            
        rowMin = row
        
    return locTargets

















