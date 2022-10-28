# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:21:50 2020

@author: Yang Hyo

update history
---------
2020-09-15  YK converted the folloing Matlab codes into Python.

            \\ma2files\Production systems\Production Tools\MATLAB sys 21\FullCalibStep2.1Temp
                1. FullCalibrationStep2M1.m

2020-09-16  YK implemented the following.
            1. Clean up hard drive space by SaoRecon exe file
             
2020-09-18  YK changed the following.
            1. refactor into sub-functions
            2. added comments
            3. bug fix for the final reconstruction to use the right Pitch/Orientation    
            
2020-10-06  YK changed the following.
            1. OnePixelFallOff, TwoPixelFallOff, Number of bead targets are written in a cfg file.
            2. For legacy systems, GUI and code are adjusted (DyeName and LaserName separated).
               Legacy system lsgd file name : PREVIEW_Alexa488.lsgd (DyeName = Alexa488, LaserName = L473)
               New system lsgd file name : PREVIEW_L473.lsgd (DyeName = L473, LaserName = L473)
"""
import numpy as np
import matplotlib.pyplot as plt
import subprocess # call exe file
import shutil, os # directory, folder
import cv2 # read tif file
from datetime import datetime # current time

# OBI library
import BeadCalibLib
    
def FullCalibrationStep2M1(sysType,
                           SaoReconExe,
                           FolderPathExpTitle,
                           DyeName,
                           LaserName,
                           OnePixelFallOff,
                           TwoPixelFallOff):
    """
    Find the Pitch/Orientation generating the best contrast for single beads.
    Result: FullCalibStep2 folder
            - cfg file (reconstruction paramerters)
            - png file (Pitch/Orientation scanning)
            - SaoRecon_Max folder (best reconstruction result)

    Parameters
    ----------
    sysType : str
        'New' or 'Legacy'
        Between before/after sys 21, there is a flip between horizontal/vertical directions.
    SaoReconExe : Path
        Reconstruction exe file path.
    FolderPathExpTitle : Path
        Directory for raw data.
    DyeName : str
        Excitation laser name for New system / Dye name for Legacty system.
    LaserName : str
        Excitation laser name.
    OnePixelFallOff : float
        Parameter for single bead detection.
    TwoPixelFallOff : float
        Parameter for single bead detection.

    Returns
    -------
    None.

    """

    print('start running for ' + DyeName )
    
    # Folder and file titles/names
    
    # Example:
    #   FolderPathExpTitle (Parent directory)
    #       D:\SV_AcqData\2020-09-23 0.2um Bead001
    #   FolderPathLsgd (Sub folder with raw data)
    #       D:\SV_AcqData\2020-09-23 0.2um Bead001\AcqData
    #   FolderPathSaoReconParent (SawRecon exe result)
    #       D:\SV_AcqData\2020-09-23 0.2um Bead001\FullCalibStep2\L488
    #   FileTitleLsgd (Raw file name)
    #       PREVIEW_L488
    
    FolderPathLsgd = FolderPathExpTitle.joinpath('AcqData')
    FolderPathSaoReconParent = FolderPathExpTitle.joinpath('FullCalibStep2', DyeName)
    FileTitleLsgd = 'PREVIEW_' + DyeName
    
    # Find Isolated Single Bead 
    (mLocation, OnePixelFallOffNew, TwoPixelFallOffNew) = findLoc(FolderPathLsgd,
                                                               FileTitleLsgd,
                                                               OnePixelFallOff,
                                                               TwoPixelFallOff)
    
    # Pitch (%)/Orientation (deg) range to search   
    PitchAdj = np.arange(-0.2, 0.2+0.05, 0.05)
    OrientAdj = np.arange(-0.2, 0.2+0.05, 0.05)
    
    # SaoRecon with adjusted Pitch/Orientation for the best contrast
    SumOfReconVal = reconPitchOrient(SaoReconExe,
                                     FolderPathLsgd,
                                     FolderPathSaoReconParent,
                                     FileTitleLsgd,
                                     mLocation,
                                     PitchAdj,
                                     OrientAdj)
    
    # Show result of Pitch/Orientation scanning
    showPitchOrient(FolderPathExpTitle,
                    DyeName,
                    SumOfReconVal)
    
    # Find the best Pitch/Orientation
    MaxPitchAdj, MaxOrientAdj = findMaxPitchOrient(SumOfReconVal,
                                                   PitchAdj,
                                                   OrientAdj)
    
    # Recon with the best Pitch/Orientation Parameters
    reconMaxPitchOrient(SaoReconExe,
                        FolderPathExpTitle,
                        FolderPathLsgd,
                        FileTitleLsgd,
                        MaxPitchAdj,
                        MaxOrientAdj)
    
    # Output results and clean up
    finalResult(FolderPathExpTitle,
                FolderPathLsgd,
                FileTitleLsgd,
                DyeName,
                LaserName,
                MaxPitchAdj,
                MaxOrientAdj,
                mLocation.shape[0],
                OnePixelFallOffNew,
                TwoPixelFallOffNew)
    
def findLoc(FolderPathLsgd,
            FileTitleLsgd,
            OnePixelFallOff,
            TwoPixelFallOff):
    """
    Find Isolated Single Bead. 

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
    OnePixelFallOff : float
        Parameter for single bead detection.
    TwoPixelFallOff : float
        Parameter for single bead detection.
    """
    FileNameLsgd = FileTitleLsgd + '.lsgd'
    FullPathLsgd = FolderPathLsgd.joinpath(FileNameLsgd)
    
    # Binary file to 12 x 2048 x 2048 ndarray
    ImgstInputLsgd = BeadCalibLib.LsgdToImageStack(FullPathLsgd)
    # 12 x 2048 x 2048 to single 2048 x 2048 image
    ImgAvg = np.mean(ImgstInputLsgd, axis=0)
    
    sBackgroundLevel = BeadCalibLib.EstimateBackgroundNoise(ImgAvg)
    
    # Tune OnePixelFallOff and TwoPixelFallOff parameters so that the number of
    # isolated single beads is between 500 and 600.
    delta = 0
    noLoc = []
    for noIter in range(5):
        OnePixelFallOff = OnePixelFallOff + delta
        TwoPixelFallOff = TwoPixelFallOff + delta
        
        # N (number of locations) x 2 (row, col) array
        # (row, col) is the location inside the image array.
        mLocation = BeadCalibLib.SelectIsolatedTargets(ImgAvg,
                                                       sBackgroundLevel,
                                                       OnePixelFallOff,
                                                       TwoPixelFallOff)
        noLocTemp = mLocation.shape[0]
        noLoc.append( noLocTemp )
        
        print('************************************')
        print(FileTitleLsgd)
        print('OnePixelFallOff: ' + str(OnePixelFallOff))
        print('TwoPixelFallOff: ' + str(TwoPixelFallOff))
        print('noLocTemp: ' + str(noLocTemp))
        
        if 500 < noLocTemp < 600:
            break
        else:
            delta = (550-noLocTemp) / (200/0.02)
    
    return (mLocation, OnePixelFallOff, TwoPixelFallOff)

def reconPitchOrient(SaoReconExe,
                    FolderPathLsgd,
                    FolderPathSaoReconParent,
                    FileTitleLsgd,
                    mLocation,
                    PitchAdj,
                    OrientAdj):
    """
    SaoRecon with adjusted Pitch/Orientation (scanning to find the best).
    Find Pitch/Orientation that maximize sum(mean(4x4) for single bead) for target beads.
    mLocation is derived for the image of 2048 x 2048 (before reconstruction).
    Coodinate conversion is needed for the image of 4096 x 4096 (after reconstruction).
    1 pixel image before recon becomes a 4x4 image after recon.

    Parameters
    ----------
    SaoReconExe : Path
        Reconstruction exe file path.
    FolderPathLsgd : Path
        Raw data (lsgd files) directory.
    FolderPathSaoReconParent : Path
        Output directory for SaoRecon exe file.
    FileTitleLsgd : str
        lsgd file name (for example, PREVIEW_L488).
    mLocation : ndarray
        N (number of locations) x 2 (row, col) array.
    PitchAdj : ndarray
        Pitch (%) range to search.
    OrientAdj : ndarray
        Orientation (deg) range to search.

    Returns
    -------
    SumOfReconVal : ndarray
        N (number of Pitch/Orientation variation) x 4 (channels) array.
        Contrast (sum of intensities from isolated singe beads)
        during Pitch/Orientation scanning.

    """
    # To run exe file, current working directory must be changed into one with exe file.
    # cwd (curretn working directory) is used to come back after running the exe file.
    cwd = os.getcwd()  
    os.chdir(SaoReconExe.parent)

    # Couting variable for Pitch/Orientation scanning
    iCountRecon = len(OrientAdj) * len(PitchAdj)
    # Contrast variable (find Pitch/Orientation that maximize this variable)
    # iCountRecon (number of search) x 4 (channels) ndarray
    SumOfReconVal = np.zeros((iCountRecon, 4))
    
    iIdxRecon = 0
    
    for iIdxO in range(len(OrientAdj)):
        for iIdxP in range(len(PitchAdj)):
            iIdxRecon += 1
    
            # Arguments for an SaoRecon exe file
            # - raw data location,
            # - output directory
            # - counting (for example, 001)
            # - raw data name without extension
            # - Pitch for each channel
            # - Orientation for each channel
            strCmd = '"%s" "%s\\SaoRecon_%03d" %s.lsgd D.tif B.tif %s.cfg %s 2048 2048 0.000000 1.000000 1.000000 5.000000 AUTO AUTO AUTO AUTO 23 0 3 0.5 0 0 2048 2048 0 1.0 1.0 1.0 1.0 0 %f %f %f %f %f %f %f %f'
            strCmd = strCmd % (str(FolderPathLsgd),
                               str(FolderPathSaoReconParent),
                               iIdxRecon,
                               FileTitleLsgd, FileTitleLsgd, FileTitleLsgd,
                               PitchAdj[iIdxP], PitchAdj[iIdxP], PitchAdj[iIdxP], PitchAdj[iIdxP],
                               OrientAdj[iIdxO], OrientAdj[iIdxO], OrientAdj[iIdxO], OrientAdj[iIdxO])
            
            # Run the exe file with arguments
            subprocess.run(SaoReconExe.name + ' ' + strCmd)
            # https://stackoverflow.com/questions/15928956/how-to-run-an-exe-file-with-the-arguments-using-python
            
            # Result tif file into image stack (4 x 4096 x 4096)
            ImgHF = []
        
            for idxCh in range(1,5):
                # PREVIEW_L488_HF1-1Ch1.tif, PREVIEW_L488_HF1-1Ch2.tif, ...
                FileNameHFCh = FileTitleLsgd + ('_HF1-1Ch%d.tif' % idxCh)
                FolderListSaoRecon = 'SaoRecon_%03d' % iIdxRecon   
                FullPathHFCh = FolderPathSaoReconParent.joinpath(FolderListSaoRecon, FileNameHFCh)
                Img = cv2.imread(str(FullPathHFCh), cv2.IMREAD_UNCHANGED)
                ImgHF.append(Img)
            
            # Clean up hard drive space used by the exe file
            PathToRemove = FolderPathSaoReconParent.joinpath(FolderListSaoRecon)
            shutil.rmtree(PathToRemove)
            
            # mLocation is N (number of locations) x 2 (row, col) array for the image of 2048 x 2048
            # before reconsctruction.
            # Coordinate conversion needed for the image after reconsctruction (4096 x 4096)
            for iIdxBead in range(len(mLocation)):
                CoordRow = int( mLocation[iIdxBead, 0] * 4 )
                CoordCol = int( mLocation[iIdxBead, 1] * 4 )
                
                for idxCh in range(4):
                    # 1 pixel becomes a 4x4 image
                    ImgSub = ImgHF[idxCh][CoordRow:CoordRow+3, CoordCol:CoordCol+3]
                    ReconValCh = np.mean(ImgSub)
                    SumOfReconVal[iIdxRecon-1, idxCh] += ReconValCh

    # Return to the original working directory
    os.chdir(cwd)
    
    return SumOfReconVal

def showPitchOrient(FolderPathExpTitle,
                    DyeName,
                    SumOfReconVal):
    """
    Show plots of contrast (SumOfReconVal) with Pitch/Orientation scanning
    and save them into a png file.

    Parameters
    ----------
    FolderPathExpTitle : Path
        Directory for raw data.
    DyeName : str
        Excitation laser name.
    SumOfReconVal : ndarray
        N (number of Pitch/Orientation variation) x 4 (channels) array.
        Contrast (sum of intensities from isolated singe beads)
        during Pitch/Orientation scanning.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots()
    ax.plot(SumOfReconVal)
    ax.legend(['Ch1', 'Ch2', 'Ch3', 'Ch4'])
    
    # L488.png
    FileNamePlot = '%s.png' % DyeName
    FileNamePlot = FolderPathExpTitle.joinpath('FullCalibStep2', FileNamePlot)
    fig.savefig(FileNamePlot)

def findMaxPitchOrient(SumOfReconVal,
                       PitchAdj,
                       OrientAdj):
    """
    Find the best Pitch/Orientation.

    Parameters
    ----------
    SumOfReconVal : ndarray
        N (number of Pitch/Orientation variation) x 4 (channels) array.
        Contrast (sum of intensities from isolated singe beads)
        during Pitch/Orientation scanning.
    PitchAdj : ndarray
        Pitch (%) range to search.
    OrientAdj : ndarray
        Orientation (deg) range to search.

    Returns
    -------
    MaxPitchAdj : ndarray
        Pitch (%) for the maximum contrast.
    MaxOrientAdj : ndarray
        Orientation (deg) for the maximum contrast.

    """
    # N (number of Pitch/Orientation variation) x 4 (channels) array
    MaxIdx = np.argmax(SumOfReconVal, axis=0)
    
    # Conversion of index for Pitch/Orientation with the following for loops
    # for iIdxO in range(len(OrientAdj)):
    #     for iIdxP in range(len(PitchAdj)):
    MaxIdxOrient = np.floor(MaxIdx / len(PitchAdj))
    MaxIdxPitch = np.mod(MaxIdx, len(PitchAdj))
    
    MaxPitchAdj = PitchAdj[MaxIdxPitch.astype(int)]
    MaxOrientAdj = OrientAdj[MaxIdxOrient.astype(int)]
    
    return MaxPitchAdj, MaxOrientAdj

def reconMaxPitchOrient(SaoReconExe,
                        FolderPathExpTitle,
                        FolderPathLsgd,
                        FileTitleLsgd,
                        MaxPitchAdj,
                        MaxOrientAdj):
    """
    Recon with the best Pitch/Orientation Parameters.

    Parameters
    ----------
    SaoReconExe : Path
        Reconstruction exe file path.
    FolderPathExpTitle : Path
        Directory for raw data.
    FolderPathLsgd : Path
        Raw data (lsgd files) directory.
    FileTitleLsgd : str
        lsgd file name (for example, PREVIEW_L488).
    MaxPitchAdj : ndarray
        Pitch (%) for the maximum contrast.
    MaxOrientAdj : ndarray
        Orientation (deg) for the maximum contrast.

    Returns
    -------
    None.

    """
    # To run exe file, current working directory must be changed into one with exe file.
    # cwd (curretn working directory) is used to come back after running the exe file.
    cwd = os.getcwd()
    os.chdir(SaoReconExe.parent)
    
    # Output directory
    FolderPathSaoReconParent = FolderPathExpTitle.joinpath('FullCalibStep2', 'SaoRecon_Max')
    
    # Counting variable
    iIdxRecon = 1
    
    # Arguments for an SaoRecon exe file
    # - raw data location,
    # - output directory
    # - counting (for example, 001)
    # - raw data name without extension
    # - Pitch for each channel
    # - Orientation for each channel    
    strCmd = '"%s" "%s\\SaoRecon_%03d" %s.lsgd D.tif B.tif %s.cfg %s 2048 2048 0.000000 1.000000 1.000000 5.000000 AUTO AUTO AUTO AUTO 23 0 3 0.5 0 0 2048 2048 0 1.0 1.0 1.0 1.0 0 %f %f %f %f %f %f %f %f'
    strCmd = strCmd % (str(FolderPathLsgd),
                       str(FolderPathSaoReconParent),
                       iIdxRecon,
                       FileTitleLsgd, FileTitleLsgd, FileTitleLsgd,
                       MaxPitchAdj[0], MaxPitchAdj[1], MaxPitchAdj[2], MaxPitchAdj[3],
                       MaxOrientAdj[0], MaxOrientAdj[1], MaxOrientAdj[2], MaxOrientAdj[3])
    
    # Run the exe file with arguments
    subprocess.run(SaoReconExe.name + ' ' + strCmd)
    # https://stackoverflow.com/questions/15928956/how-to-run-an-exe-file-with-the-arguments-using-python
    
    # Return to the original working directory
    os.chdir(cwd)
    
def finalResult(FolderPathExpTitle,
                FolderPathLsgd,
                FileTitleLsgd,
                DyeName,
                LaserName,
                MaxPitchAdj,
                MaxOrientAdj,
                NoTargets,
                OnePixelFallOffNew,
                TwoPixelFallOffNew):
    """
    Convert Pitch/Orientation result into right units, save them as a cfg file, and clean up.
    Temporary working folder has a big size reconstruction data coming from SaoRecon exe.
    After gererating final result, it must be cleared.

    Parameters
    ----------
    FolderPathExpTitle : Path
        Directory for raw data.
    FolderPathLsgd : Path
        Raw data (lsgd files) directory.
    FileTitleLsgd : str
        lsgd file name (for example, PREVIEW_L488).
    DyeName : str
        Excitation laser name for New system / Dye name for Legacty system.
    LaserName : str
        Excitation laser name.
    MaxPitchAdj : ndarray
        Pitch (%) for the maximum contrast.
    MaxOrientAdj : ndarray
        Orientation (deg) for the maximum contrast.
    NoTargets : int
        Number of isolated single beads used for calibration
    OnePixelFallOffNew : float
        Parameter for single bead detection.
    TwoPixelFallOffNew : float
        Parameter for single bead detection.

    Returns
    -------
    None.

    """
    # Time to finish running
    timeStamp = datetime.now()
    strTimeStamp = timeStamp.strftime("%m/%d/%Y %H:%M:%S")
    # https://docs.python.org/3/library/time.html
    
    # Open the original cfg file to read parameters
    FileNameCfg = '%s.cfg' % FileTitleLsgd
    FullPathCfg = FolderPathLsgd.joinpath(FileNameCfg)
    
    # Typical cfg file contents
    # F1: 0.784121865 0.784021815 0.78725343 0.788183895
    # F2: -0.00449 0.776 1.56601 2.35067
    # F3: 1.2759387594 -1.2759387594 0.9107940354 -0.9107940354 0.0060781515 -0.0060781515 -0.8926140932 0.8926140932
    # F4: -0.0057284399 0.0057284399 0.8938253372 -0.8938253372 1.2708532331 -1.2708532331 0.9025299905 -0.9025299905
    # F5: 3.7548124672 3.7548124672 4.4556562641 4.4556562641 3.7554264692 3.7554264692 4.4315681751 4.4315681751 1.3426331361
    
    fid = open(FullPathCfg, 'r')
    FParam = []
    for idx in range(5):
        FParam.append( fid.readline() )
    fid.close()
    
    # Split a string into a list
    # F1: 0.784121865 0.784021815 0.78725343 0.788183895
    # ['F1:', '0.784121865', '0.784021815', '0.78725343', '0.788183895']
    strPitch = FParam[0].split()
    strOrient = FParam[1].split()
    
    # Number only list
    PList = np.array(strPitch[1:])
    OList = np.array(strOrient[1:])
    
    PList = PList.astype(np.float)
    OList = OList.astype(np.float)
    
    # Calculate adjusted parameters for maximum contrast
    # Convert Pitch (%)/Orientation (deg) into right units
    PitchFinal = np.multiply(PList, 1 + MaxPitchAdj/100)
    OrientFinal = np.add(OList, MaxOrientAdj/180*np.pi)
    
    # Adjust the number of digits
    PitchFinal = np.round(PitchFinal, 10)
    OrientFinal = np.round(OrientFinal, 10)
    
    FParam[0] = 'F1: ' + ' '.join(PitchFinal.astype(str)) + '\n'
    FParam[1] = 'F2: ' + ' '.join(OrientFinal.astype(str)) + '\n'
    
    # Save as Txt file
    FullPathOutput = FolderPathExpTitle.joinpath('FullCalibStep2',
                                             'global.' + LaserName[1:] + '.cfg')
    fid = open(FullPathOutput, 'w')
    fid.writelines(FParam)
    fid.write('\n')
    fid.write(strTimeStamp +', Calibration 2' + '\n')
    fid.write('OnePixelFallOff: ' + str(OnePixelFallOffNew) + '\n')
    fid.write('TwoPixelFallOff: ' + str(TwoPixelFallOffNew) + '\n')
    fid.write('Number of bead targets: ' + str(NoTargets))
    fid.close()
    
    # Clean the temp folder
    FolderPathTemp = FolderPathExpTitle.joinpath('FullCalibStep2', DyeName)
    shutil.rmtree(FolderPathTemp)
    
    # Print on Screen    
    print()
    print(strTimeStamp)
    print()
    print('Full Calibration Step2 Pitch/Orientation:')
    print()
    print(PitchFinal)
    print(OrientFinal)
    print('\n')   