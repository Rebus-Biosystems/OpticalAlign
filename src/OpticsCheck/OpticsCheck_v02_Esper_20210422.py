# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 2020

@author: YK from OBI

update history
---------
2021-02-17  YK release 'Optic Check v.0.2'
            YK worked with the name of 'SVSAOMeasure'

            Frozen successfully with the following version of softwares:
            1. Windows 10
            2. Anaconda3-2020.07-Windows-x86_64.exe
            3. Python 3.8.3
            4. matplotlib 3.3.4
            5. PyInstaller 4.2
            
            !pyinstaller --onedir -y OpticCheck_v02_20210217.py
            
2021-02-26  YK changed the following.
            1. Name change: 'OpticCheck' -> 'OpticsCheck'
            
2021-03-18  YK changed the following.
            1. Utilize U-image instead of S-image.
            2. HF-image auto-contrast for result
            3. Contour lines are added to peak value ratio plot (Super-Gaussian fitting)
            4. Added more comments
            
2021-03-25  YK changed the following.
            1. Save result figures automatically.
            2. Update Pass/Fail result to the image.
            3. Automatically run the software multiple times
                - adjusting single particle seeking parameters.
            4. Old system (dye name is used as a file name) compatibility
                - read log txt file and automatically assign excitation laser.
            5. Enable multiple date data processing.
            6. Enable 'all' checkbox function.
            7. Temporary data saving location changed.
                - from the exe file directory to raw data directory
                - prevent err for running in the file server
                - err detail: file writing permission
                
2021-04-16
!pyinstaller --onedir -y -n OpticsCheck OpticsCheck_v02_Esper_20210409.py

2021-04-27  YK changed the following.
            1. Changed wxPython GUI frame name from 'TestFrame' to 'GUIFrame'.
            2. Remove long option for input files from command-line argument.
            
2021-05-14  YK changed the following.
            1. Signal saturation prevention: reject beads brighter than 20,000
            
References:
https://docs.python.org/3/library/multiprocessing.html
https://www.blog.pythonlibrary.org/2010/05/22/wxpython-and-threads/
https://wiki.wxpython.org/LongRunningTasks
https://wxpython.org/Phoenix/docs/html/events_overview.html
https://stackoverflow.com/questions/15928956/how-to-run-an-exe-file-with-the-arguments-using-python
https://www.tutorialspoint.com/python/python_command_line_arguments.htm
-------------------------
To keep the GUI responsive when the application has to do long running tasks,
wx.PostEvent and Threads are used.

wx.CallAfter(callableObj, *args, **kw)
Call the specified function after the current and pending event handlers have been completed.
Make GUI method calls from non-GUI threads.

runBtnClick(self, event)
->
self.runData1Worker = runData1Thread(self,
                                     InFilePath,
                                     self.laserChoList,
                                     self.runCbxList,
                                     self.dicOne,
                                     self.dicTwo)
->
class runData1Thread(Thread):
	def run(self):
		# Find Isolated Single Bead
                mLocation = SAOMeasureLib.findLocMulti(FolderPathLsgd,
                                                       FileTitleLsgd,
                                                       OnePixelFallOff,
                                                       TwoPixelFallOff)

		wx.PostEvent(self._notify_window, ResultEvent((self._InFilePath, FileTitleLsgd, idx)))
->
# Set up event handler for any worker thread results
EVT_RESULT(self, self.runData2)
->
runData2(self, event)
"""
from pathlib import Path # path, directory, file name, file extension
import glob # file check
import os
import sys, getopt

# GUI library (wxPython)
import wx
import wx.lib.scrolledpanel as scrolled

import ctypes # High DPI monitor

import multiprocessing # Utilize a multi-core CPU fore speed-up
from threading import Thread # Prevent GUI freezing

import numpy as np
import matplotlib.pyplot as plt
import cv2 # read tif file

# OBI library
import BeadCalibLib
import SAOMeasureLib
import GaussFitLib

### version info
global version
version = 'OpticsCheck_v02_Esper_20210422'

# Define notification event for thread completion
EVT_RESULT_ID = wx.ID_ANY

def EVT_RESULT(win, func):
    """Define Result Event."""
    win.Connect(-1, -1, EVT_RESULT_ID, func)

class ResultEvent(wx.PyEvent):
    """Event class to carry arbitrary result data."""
    def __init__(self, data):
        """Init Result Event."""
        wx.PyEvent.__init__(self)
        self.SetEventType(EVT_RESULT_ID)
        self.data = data

# Thread class that executes processing
class runData1Thread(Thread):
    """Worker Thread Class."""
    def __init__(self, notify_window, InFilePath, laserChoList, runCbxList, oneScrList, twoScrList):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self._notify_window = notify_window
        self._InFilePath = InFilePath
        self.laserChoList = laserChoList
        self.runCbxList = runCbxList
        self.oneScrList = oneScrList
        self.twoScrList = twoScrList
        self._want_abort = 0
        # This starts the thread running on creation, but you could
        # also make the GUI thread responsible for calling this
        self.start()

    def run(self):
        """Run Worker Thread."""
        # This is the code executing in the new thread. Simulation of
        # a long process (well, 10s here) as a simple loop - you will
        # need to structure your processing so that you periodically
        # peek at the abort variable
        
        # glob.glob() returns a list of string path names that match pathname
        lsgdPath = self._InFilePath.joinpath('AcqData', '*.lsgd')
        fList = glob.glob(str(lsgdPath))
        
        FolderPathLsgd = self._InFilePath.joinpath('AcqData')
        
        FileTitleLsgdList = []
        FileIdxList = []
        laserNameList = []
        OneList = []
        TwoList = []
        
        for idx, fPath in enumerate(fList):
            if self.runCbxList[idx].IsChecked():
                laserIdx = self.laserChoList[idx].GetSelection()
                if laserIdx == wx.NOT_FOUND:
                    wx.MessageBox("Laser not selected.", "Run button",
                                      wx.OK | wx.ICON_WARNING)
                else:
                    laserName = self.laserChoList[idx].GetString(laserIdx)
                    laserNameList.append(laserName)
                    FileTitleLsgd = Path(fPath).stem
                    
                    FileTitleLsgdList.append(FileTitleLsgd)
                    FileIdxList.append(idx)
                    
                    OnePixelFallOff = self.oneScrList[idx].GetValue()
                    TwoPixelFallOff = self.twoScrList[idx].GetValue()
                    
                    # if laserName != 'L750':
                    if laserName != 'LL750':
                    
                        print('***********************************')
                        print(FileTitleLsgd + ' with index ' + str(idx) + ' started.')
        
                        numberLocation = 0
                        
                        # Find Isolated Single Bead
                        mLocation = SAOMeasureLib.findLocMulti(FolderPathLsgd,
                                                                FileTitleLsgd,
                                                                OnePixelFallOff,
                                                                TwoPixelFallOff)
                        numberLocation = np.max(mLocation.shape)
                        
                        print('OnePixelFallOff: ' + str(OnePixelFallOff))
                        print('TwoPixelFallOff: ' + str(TwoPixelFallOff)) 
                        print('numberLocation: ' + str(numberLocation))
                        
                        ### Total number control >65%
                        numberLocationMin = 900
                        numberRun = 1
                        
                        for idxRun in range(4):
                            adaptiveSlope = 0.02/50 * (3**idxRun)
                            if (numberLocation < numberLocationMin) and (TwoPixelFallOff<1):
                                OnePixelFallOff += (numberLocationMin - numberLocation)*adaptiveSlope
                                TwoPixelFallOff += (numberLocationMin - numberLocation)*adaptiveSlope
                                
                                if OnePixelFallOff > 1:
                                    OnePixelFallOff = 1
                                if TwoPixelFallOff > 1:
                                    TwoPixelFallOff = 1
                                    
                                # Find Isolated Single Bead
                                mLocation = SAOMeasureLib.findLocMulti(FolderPathLsgd,
                                                                        FileTitleLsgd,
                                                                        OnePixelFallOff,
                                                                        TwoPixelFallOff)
                                numberLocation = np.max(mLocation.shape)
                                
                                print('OnePixelFallOff: ' + str(OnePixelFallOff))
                                print('TwoPixelFallOff: ' + str(TwoPixelFallOff)) 
                                print('numberLocation: ' + str(numberLocation))
                                
                                numberRun += 1
                                
                        print('Number of trials: ' + str(numberRun))
                        
                        mLocation = mLocation.astype(int)
                        mLocation = mLocation * 4
                        
                        fnameSave = self._InFilePath.joinpath('mLocation' + str(idx))
                        np.save(fnameSave, mLocation) 
                        
                        OnePixelFallOff = round(OnePixelFallOff, 3)
                        TwoPixelFallOff = round(TwoPixelFallOff, 3)
                        
                        OneList.append(OnePixelFallOff)
                        TwoList.append(TwoPixelFallOff)
                    
        ### worker thread post event
        wx.PostEvent(self._notify_window, ResultEvent((self._InFilePath,
                                                       FileTitleLsgdList,
                                                       FileIdxList,
                                                       laserNameList,
                                                       OneList,
                                                       TwoList)))
                                 
    def abort(self):
        """abort worker thread."""
        # Method for use by main thread to signal an abort
        self._want_abort = 1

class GUIFrame(wx.Frame):
    def __init__(self):
        try:
            # High DPI monitor
            ctypes.windll.shcore.SetProcessDpiAwareness(True)
        except:
            pass

        wx.Frame.__init__(self, None, -1, version)
        # self.panel = wx.Panel(self)
        self.panel = scrolled.ScrolledPanel(self)
        
        self.initParams()
        
        self.createControl()
        self.layoutControl()
        
        # Set up event handler for any worker thread results
        EVT_RESULT(self, self.runData2)
        
        # And indicate we don't have a worker thread yet
        self.runData1Worker = None
        
    def initParams(self):
        self.laserList = ['L473','L488','L532','L595','L647','L660','L750']
        
        # Empirical OnePixelFallOff/TwoPixelFallOff values for each laser
        self.dicOne = {'L473':0.74,
                        'L488':0.74,
                        'L532':0.74,
                        'L595':0.8,
                        'L647':0.8,
                        'L660':0.8,
                        'L750':0.86}
        self.dicTwo = {'L473':0.36,
                       'L488':0.36,
                       'L532':0.36,
                       'L595':0.4,
                       'L647':0.4,
                       'L660':0.4,
                       'L750':0.46}
        
        ### Empirical Pass/Fail threshold for each laser
        self.passFail = {'L473': [22, 0.5, 11],
                         'L488': [22, 0.5, 11],
                         'L532': [22, 0.63, 7],
                         'L595': [22, 0.56, 13],
                         'L647': [22, 0.54, 14],
                         'L660': [22, 0.54, 14],
                         'L750': [22, 0.54, 14]}
        
        self.InFilePath = ''
        self.systemSerial = ''
        self.acqTime = ''
        
        # Show result figures
        self.figOn = True
          
    def createControl(self):
        """create widgets (controls) for GUI"""
        panel = self.panel
        
        self.dirPic = wx.DirPickerCtrl(panel, -1, "", size=(800,-1))
        
        self.fnameTxt = wx.StaticText(panel, -1, "lsgd file", size=(400, 40), style=wx.ST_NO_AUTORESIZE)
        self.lnameTxt = wx.StaticText(panel, -1, "laser", size=(70, 40), style=wx.ST_NO_AUTORESIZE)
        
        self.runallCbx = wx.CheckBox(self.panel, -1, "all")
        self.Bind(wx.EVT_CHECKBOX, self.runallCbxClick, self.runallCbx)
        
        self.oneTxt = wx.StaticText(panel, -1, "One", size=(70, 40), style=wx.ST_NO_AUTORESIZE)
        self.twoTxt = wx.StaticText(panel, -1, "Two", size=(70, 40), style=wx.ST_NO_AUTORESIZE)
        
        self.fileTextList = []
        self.laserChoList = []
        self.runCbxList = []
        self.oneScrList = []
        self.twoScrList = []
        
        for idx in range(5):
            self.fileTextList.append( wx.StaticText(panel, -1, label="None", size=(400, 40), style=wx.ST_NO_AUTORESIZE) )
            self.laserChoList.append( wx.Choice(self.panel, -1, choices=self.laserList) )
            self.runCbxList.append( wx.CheckBox(self.panel, -1) )
            self.oneScrList.append( wx.SpinCtrlDouble(self.panel, -1, size=(100, 40), min=0, max=1, initial=0, inc=0.02) )
            self.twoScrList.append( wx.SpinCtrlDouble(self.panel, -1, size=(100, 40), min=0, max=1, initial=0, inc=0.02) )
            
            ###
            self.Bind(wx.EVT_CHOICE, self.OnLaserCho, self.laserChoList[idx])
            
        self.runBtn = wx.Button(panel, -1, 'Run', size=(300,50))
        self.Bind(wx.EVT_BUTTON, self.runBtnClick, self.runBtn)
        
        self.closePlotsBtn = wx.Button(panel, -1, 'Close plots', size=(300,50))
        self.Bind(wx.EVT_BUTTON, self.closePlotsBtnClick, self.closePlotsBtn)        
     
    def layoutControl(self):
        """sizers are used for layout (resizing and moving the widgets inside)"""
        # mainSizer is the top-level one that manages everything
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        
        mainSizer.Add(self.dirPic, 0, wx.EXPAND|wx.ALL, 10)
        
        border = 10
        
        infoSizerList = []
        for idx in range(6):
            infoSizerList.append( wx.BoxSizer(wx.HORIZONTAL) )
            if idx == 0:
                infoSizerList[idx].Add(self.fnameTxt, 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, border)
                infoSizerList[idx].Add(self.lnameTxt, 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, border)
                infoSizerList[idx].Add(self.runallCbx, 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, border)
                infoSizerList[idx].Add(self.oneTxt, 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, border)
                infoSizerList[idx].Add(self.twoTxt, 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, border)
            else:
                infoSizerList[idx].Add(self.fileTextList[idx-1], 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, border)
                infoSizerList[idx].Add(self.laserChoList[idx-1], 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, border)
                infoSizerList[idx].Add(self.runCbxList[idx-1], 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, border)
                infoSizerList[idx].Add(self.oneScrList[idx-1], 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, border)
                infoSizerList[idx].Add(self.twoScrList[idx-1], 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, border)
            mainSizer.Add(infoSizerList[idx], 0, wx.EXPAND | wx.TOP, border)
        
        mainSizer.Add((10,10)) # some empty space
        
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer.Add(self.runBtn, 0, wx.ALL, 10)
        btnSizer.Add(self.closePlotsBtn, 0, wx.ALL, 10)
        
        mainSizer.Add(btnSizer, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border)
        
        # Connect the mainSizer to the container widget (panel)
        self.panel.SetSizer(mainSizer)
        
        # Fit the frame to the needs of the sizer. The frame will
        # automatically resize the panel as needed. Also prevent the
        # frame from getting smaller than this size.
        mainSizer.Fit(self)
        # mainSizer.SetSizeHints(self)  
        
        self.panel.SetupScrolling()
        
    def OnLaserCho(self, event):
        laserName = event.GetString()
        idx = self.laserChoList.index( event.GetEventObject() )
        # https://stackoverflow.com/questions/3410196/wxpython-how-to-determine-source-of-an-event
        # https://stackoverflow.com/questions/176918/finding-the-index-of-an-item-in-a-list
       
        self.oneScrList[idx].SetValue(self.dicOne[laserName])
        self.twoScrList[idx].SetValue(self.dicTwo[laserName])
    
    def runallCbxClick(self, event):
        checkValue = self.runallCbx.GetValue()
        for idx, fileTextList in enumerate(self.fileTextList):
            if fileTextList.GetLabel() != 'None':
                self.runCbxList[idx].SetValue(checkValue)
        
    def runBtnClick(self, event):
        """run button click event handler (initiate data processing)"""        
        InFilePath = self.dirPic.GetPath()
        
        # check if the input path from GUI is vaild
        if InFilePath == "":
            wx.MessageBox("Directory not entered.", "Run button",
                          wx.OK | wx.ICON_WARNING)
        else:
            InFilePath = Path(InFilePath)
            
            if not InFilePath.exists():
                wx.MessageBox("Directory not exist.", "Run button",
                              wx.OK | wx.ICON_WARNING)
            else:
                if InFilePath != self.InFilePath:
                    self.fileTextList[0].SetLabel('None')
                    self.InFilePath = InFilePath
                    
                if self.fileTextList[0].GetLabel() == 'None':
                    if not self.checkFiles(InFilePath):
                        wx.MessageBox("Data file not found.", "Run button",
                                      wx.OK | wx.ICON_WARNING)
                else:
                    # To prevent GUI freezing,
                    # spawn a long running process onto a separate thread.
                    # Trigger the worker thread unless it's already busy
                    if not self.runData1Worker:
                        # self.status.SetLabel('Starting computation')
                        
                        self.runData1Worker = runData1Thread(self,
                                                              InFilePath,
                                                              self.laserChoList,
                                                              self.runCbxList,
                                                              self.oneScrList,
                                                              self.twoScrList)
                    
    def closePlotsBtnClick(self, event):
        plt.close('all')
                
    def checkFiles(self, InFilePath):
        """
        Check lsgd file existence

        Parameters
        ----------
        InFilePath : pathlib.Path obj, path for the input data file

        Returns
        -------
        bool if lsgd files exist

        """
        # glob.glob() returns a list of string path names that match pathname
        lsgdPath = InFilePath.joinpath('AcqData', '*.lsgd')
        fList = glob.glob(str(lsgdPath))
        numLsgd = len( fList )
        if numLsgd == 0:
            return False
        else:
            for idx, fPath in enumerate(fList):
                fileName = Path(fPath).stem
                self.fileTextList[idx].SetLabel(fileName)
                
                # Find out the excitation laser from the log file
                # Some old system stores the data with dye name instead of laser name
                
                # https://stackoverflow.com/questions/4940032/how-to-search-for-a-string-in-text-files
                logPath = InFilePath.joinpath('AcqData', fileName + '.AcqLog.txt')
                
                with open(logPath) as f:
                    txt = f.read()
                
                if idx == 0:
                    for item in txt.split("\n"):
                        if "System serial #:" in item:
                            self.systemSerial = item
                            
                        if "Acquisition start time:" in item:
                            self.acqTime = item
                
                keyword = 'Excitation (nm) - Emission (nm): '
                laseNameStartIdx = txt.find(keyword) + len(keyword)
                laserName = txt[laseNameStartIdx:laseNameStartIdx+3]
                laserName = 'L' + laserName

                self.laserChoList[idx].SetStringSelection(laserName)
                self.oneScrList[idx].SetValue(self.dicOne[laserName])
                self.twoScrList[idx].SetValue(self.dicTwo[laserName])
                self.runCbxList[idx].SetValue(True)
                
            return True
        
    def OnResult(self, event):
        """Show Result status."""
        if event.data is None:
            # Thread aborted (using our convention of None return)
            self.status.SetLabel('Computation aborted')
        else:
            plt.figure().show()
            
            # Process results here
            self.status.SetLabel('Computation Result: %s' % event.data)
        # In either event, the worker is done
        self.worker = None
                
    def runData2(self, event):
        """
        Post event hander for worker thread

        """
        print('runData2')
        
        InFilePath, FileTitleLsgdList, FileIdxList, LaserList, OneList, TwoList = event.data
        
        
        print('FileIdxList length: ' + str(len(FileIdxList)))
        
        for idx in range(len(FileIdxList)):
            FileTitleLsgd = FileTitleLsgdList[idx]
            FileIdx = FileIdxList[idx]
            
            # if LaserList[idx] != 'L750':
            if LaserList[idx] != 'LL750':
                print('Showing result for ' + FileTitleLsgd + ' with index ' + str(FileIdx))
                
                locFileName = InFilePath.joinpath( 'mLocation' + str(FileIdx) + '.npy' )
                mLocation = np.load(locFileName)
                os.remove(locFileName)
                
                mLocationSub = []
                sizeImg = 8192
                sizeQualityZone = 5464
                minLoc = int((sizeImg - sizeQualityZone)/2)
                maxLoc = minLoc + sizeQualityZone
        
                for ind in range( mLocation.shape[0] ):
                    # if (2048 <= mLocation[ind,0] < 2048+4096) and (2048 <= mLocation[ind,1] < 2048+4096):
                    if (minLoc <= mLocation[ind,0] <= maxLoc) and (minLoc <= mLocation[ind,1] <= maxLoc):
                        mLocationSub.append(mLocation[ind])
                
                FolderPathLsgd = InFilePath.joinpath('AcqData')
                
                ##### uniform illumination image
                FileNameLsgd = FileTitleLsgd + '.lsgd'
                FullPathLsgd = FolderPathLsgd.joinpath(FileNameLsgd)
                
                # Binary file to 12 x 2048 x 2048 ndarray
                ImgstInputLsgd = BeadCalibLib.LsgdToImageStack(FullPathLsgd)
                # 12 x 2048 x 2048 to single 2048 x 2048 image
                img0 = np.mean(ImgstInputLsgd, axis=0)
                
                ##### reconstructed tif file (8192 x 8192)
                FolderPathRecon = InFilePath.joinpath('SaoRecon_8192')
                HFImageFilePath = FolderPathRecon.joinpath(FileTitleLsgd + '_HF.tif')
                
                img = cv2.imread(str(HFImageFilePath), cv2.IMREAD_UNCHANGED)
                img = img.astype('float64')
                
                peakRelArray = SAOMeasureLib.analPeak(img, img0, mLocation)
                peakRelArraySub = SAOMeasureLib.analPeak(img, img0, mLocationSub)        
            
                # Show result
                fig, axs = plt.subplots(1, 3, figsize=(14,6))
                plt.subplots_adjust(wspace=0.3, hspace=0)
                # https://stackoverflow.com/questions/20057260/how-to-remove-gaps-between-subplots-in-matplotlib
                                
                imgplots = []
                
                imgplots.append( axs[0].imshow(img, cmap='gray') )
                
                climMean = np.mean(img[np.nonzero(img>50)])
                climMin = 0.25 * climMean
                climMax = 1.5 * climMean
                imgplots[0].set_clim(climMin, climMax)
                axs[0].title.set_text('HF-image of beads')
                axs[0].title.set_fontsize('x-large')
                fig.colorbar(imgplots[0], ax=axs[0], fraction=0.046, pad=0.04)
        
                imgplots.append( axs[1].scatter(mLocation[:,1],mLocation[:,0], c=peakRelArray, cmap='jet') )
                # imgplots[1].set_clim(0.5, 2.25)
                axs[1].set_xlim(0, sizeImg); axs[1].set_ylim(0, sizeImg)
                axs[1].invert_yaxis(); axs[1].set_aspect('equal')
                axs[1].title.set_text('Peak value ratio of HF-image to U-image')
                axs[1].title.set_fontsize('x-large')
                fig.colorbar(imgplots[1], ax=axs[1], fraction=0.046, pad=0.04)
                climRatioMin, climRatioMax = imgplots[1].get_clim()
                
                # Dimension matching for XY coordinate and data
                X = np.expand_dims(mLocation[:,1], axis=1)
                Y = np.expand_dims(mLocation[:,0], axis=1)
                
                XY = (X, Y)
                
                data = peakRelArray
                
                params = GaussFitLib.fitSuperGaussian(XY, data)
                fit = GaussFitLib.superGaussian(*params)
                
                # plt.figure()
                indices = np.flip(np.indices((sizeImg, sizeImg)),0)
                
                imgplots.append( axs[1].contour(fit(*indices), cmap=plt.cm.jet, linewidths = 3) )
                imgplots[1].set_clim(climRatioMin, climRatioMax)
                
                (height, center_x, center_y, width_x, width_y, rotation, order) = params
                
                # Zyla 4.2 PLUS pixel size = 6.5 um, Nikon objective lens = 20x, x4 pixels between raw and reconstructed images
                dPixelUm = 6.5/20/4
                center_xUm = round( (center_x - sizeImg/2) * dPixelUm, 1 )
                center_yUm = round( (center_y - sizeImg/2) * dPixelUm, 1 )
                
                idxQualityZone = np.flip(np.indices((sizeQualityZone, sizeQualityZone)),0)
                idxQualityZone += int(sizeImg/2 - sizeQualityZone/2)
                fitQualityZone = fit( *idxQualityZone )
                minMaxQualityZone = round( np.min(fitQualityZone) / height, 2 )
                
                # np.save('mLocation', mLocation)
                # np.save('peakRelArray', peakRelArray)
                
                peakRelArraySubFlat = peakRelArraySub.flatten()
                peakRelArraySubFlat.sort()
                peakRelArraySubMax = np.mean( peakRelArraySubFlat[-50:] )
                peakRelArraySubMax = round(peakRelArraySubMax,1)
                peakRelArraySubMin = peakRelArraySubMax * 0.6
                
                imgplots.append( axs[2].hist(peakRelArraySub,
                                              bins = 20,
                                              range = (peakRelArraySubMin, peakRelArraySubMax),
                                              cumulative = True,
                                              density = True) )
                histValue = imgplots[3][0]
                histBins = imgplots[3][1]
                
                binPassFail = self.passFail[LaserList[idx]][2]
                percentPassFail = 0.2
                
                binData = np.argmax(histValue >= percentPassFail)
                valueData = (histBins[binData-1] + histBins[binData]) / 2
                valuePassFail = (histBins[binPassFail-1] + histBins[binPassFail]) / 2
                
                axs[2].axvline(x=valueData, color="red")
                
                axs[2].axvline(x=valuePassFail, color="black", linestyle="--")
                axs[2].axhline(y=percentPassFail, color="black", linestyle="--")
                
                            
                axs[2].set_aspect(1./axs[2].get_data_ratio() * 0.92)
                axs[2].title.set_text('Histogram of peak value ratio')
                axs[2].title.set_fontsize('x-large')
                
                ### Figure super title
                passFail = 'Pass'
                
                figTitle = 'File (laser): ' + FileTitleLsgd + '.lsgd'
                figTitle = figTitle + ' (' + LaserList[idx] + ')'
                
                figTitle = figTitle + '\n' + 'OnePixel / TwoPixel: '
                figTitle = figTitle + str(OneList[idx]) + ' / ' + str(TwoList[idx])
                
                figTitle = figTitle + '\n' + 'No of beads total / 5464: '
                figTitle = figTitle + str(mLocation.shape[0]) + ' / ' + str(len(mLocationSub))
                figTitle = figTitle + ' (Total >= 900: '
                if mLocation.shape[0] >= 900:
                    figTitle = figTitle + 'Pass)'
                else:
                    figTitle = figTitle + 'Fail)'
                    passFail =  'Fail'
                    
                figTitle = figTitle + '\n' + 'Gaussian center (um) X/Y: '
                figTitle = figTitle + str(center_xUm) + ' / ' + str(center_yUm)
                centerPassFail = self.passFail[LaserList[idx]][0]
                figTitle = figTitle + ' (<= ' + str(centerPassFail) + ': '
                if (abs(center_xUm) <= centerPassFail) and (abs(center_yUm) <= centerPassFail):
                    figTitle = figTitle + 'Pass)'
                else:
                    figTitle = figTitle + 'Fail)'
                    passFail = 'Fail'
                
                figTitle = figTitle + '\n' + 'Gaussian Min/Max ratio in 5464: ' 
                figTitle = figTitle + str(minMaxQualityZone)
                minMaxRatioPassFail = self.passFail[LaserList[idx]][1]
                figTitle = figTitle + ' (>= ' + str(minMaxRatioPassFail) + ': '
                if minMaxQualityZone >= minMaxRatioPassFail:
                    figTitle = figTitle + 'Pass)'
                else:
                    figTitle = figTitle + 'Fail)'
                    passFail = 'Fail'
                
                figTitle = figTitle + '\n' + 'Low 20% peak value ratio in 5464: ' 
                figTitle = figTitle + str(binData)
                figTitle = figTitle + ' (bin number >= ' + str(binPassFail) + ': '
                if binData >= binPassFail:
                    figTitle = figTitle + 'Pass)'
                else:
                    figTitle = figTitle + 'Fail)'
                    passFail = 'Fail'
                    
                fig.suptitle(figTitle, x=0.1, ha='left', fontsize='x-large', fontweight='bold')
                
                if passFail == 'Pass':
                    txtColor = 'g'
                else:
                    txtColor = 'r'
                    
                fig.text(0.8, 0.8, passFail, fontsize=42, fontweight='bold', color=txtColor)
               
                plt.tight_layout()
                if self.figOn == True:
                    plt.show(block=False)
                
                # make output directory if not exits
                OutFilePath = InFilePath / 'SAO Inspection'
                
                if Path.is_dir(OutFilePath) != True:
                    Path.mkdir(OutFilePath)
                
                fnameSave = LaserList[idx] + "_SAOInspection.png"
                
                OutFileNamePng = OutFilePath.joinpath(fnameSave)
                fig.savefig(OutFileNamePng)
                
                ### Result txt file
                fnameSave = LaserList[idx] + "_SAOInspection.txt"
                OutFileNameTxt = OutFilePath.joinpath(fnameSave)
                
                f = open(OutFileNameTxt, "w")
                f.write(self.systemSerial + '\n')
                f.write(self.acqTime + '\n')
                f.write('\n')
                f.write(figTitle + '\n')
                f.write('Pass/Fail: ' + passFail)
                f.close()           
            
        self.runData1Worker = None
        
        if self.figOn == False:
            self.Close()
            
############################################################################################
### wxPython (GUI) app running part
    
if __name__ == "__main__":
    # On Windows calling this function is necessary for a frozen executable.
    # One needs to call this function straight after the if __name__ == '__main__' line of the main module.
    multiprocessing.freeze_support()
    
    if len(sys.argv) == 1:
        app = wx.App()
        frm = GUIFrame()
        frm.Show()
        app.MainLoop()
    else:
        # sys.argv[0] is the program ie. script name.
        argv = sys.argv[1:]
        
        try:
            #opts: a list of (option, value) pairs
            #args: a list of program arguments left after stripping the option list
            opts, args = getopt.getopt(argv, "hvi:")
        except getopt.GetoptError:
            print('OpticsCheck.exe -i "inputfile"')
            sys.exit(2)
            
        for opt, arg in opts:
            if opt == '-h':
                print('OpticsCheck.exe -i "inputfile"')
                sys.exit()
            elif opt == '-v':
                print(version)
                sys.exit()
            elif opt == '-i':
                filePath = arg

        app = wx.App()
        frm = GUIFrame()
        # frm.Show()
        
        frm.figOn = False
        frm.dirPic.SetPath(filePath)
        frm.runBtnClick(0)
        frm.runBtnClick(0)
        
        app.MainLoop()
