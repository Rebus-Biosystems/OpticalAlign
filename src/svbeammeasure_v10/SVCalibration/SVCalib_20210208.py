# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 2020

@author: YK from OBI

update history
---------
2020-09-15  Current Matlab code structure

            MainScriptFullCalibStep2Run488
            FullCalibrationStep2M1
                LsgdToImageStack
                FcfFindSingleBeads
                    EstimateBackgroundNoise (VtoPhMatlabGUI\VoltageToPhaseMappingToolbox)
                    SelectIsolatedTargets (VtoPhMatlabGUI\VoltageToPhaseMappingToolbox)
                dos(strCmd)
                
            YK converted the Matlab codes into Python.
            
2020-09-16  YK implemented the following.
            1. Use a temporary working folder in C drive (ssd) to speed-up
            2. Clean up hard drive space by SaoRecon exe file
             
2020-09-18  YK changed the following.
            1. implemented GUI
            2. added comments
            3. lsgd file existence checking
            3. SaoRecon exe file existence checking
            4. bug fix for the final reconstruction to use the right Pitch/Orientation

            Frozen successfully with the following version of softwares:
            1. Windows 10
            2. Anaconda3-2020.07-Windows-x86_64.exe
            3. Python 3.8.3
            4. matplotlib 3.2.2
            5. PyInstaller 4.0
            
            Remaining issue:
            1. Warning when the pyinstaller executable is run.
            The MATPLOTLIBDATA environment variable was deprecated in Matplotlib 3.1 and will be removed in 3.3.
            It will be resolved by later version of PyInstaller.
            
2020-10-06  YK changed the following.
            1. OnePixelFallOff, TwoPixelFallOff, Number of bead targets are written in a cfg file.
            2. For legacy systems, GUI and code are adjusted (DyeName and LaserName separated).
               Legacy system lsgd file name : PREVIEW_Alexa488.lsgd (DyeName = Alexa488, LaserName = L473)
               New system lsgd file name : PREVIEW_L473.lsgd (DyeName = L473, LaserName = L473)
               
            PyInstaller command in Spyder
            !pyinstaller --onedir -y SVCalib_20201006.py
            
2020-02-03  YK changed the following.
            1. L660 laser added
            2. Update matplotlib 3.3.4
            3. Update pyinstaller 4.2
"""
from pathlib import Path # path, directory, file name, file extension
import shutil # copy a file
import glob # file check

# GUI library (wxPython)
import wx
import wx.lib.scrolledpanel as scrolled

import ctypes # High DPI monitor

import multiprocessing # Utilize a multi-core CPU fore speed-up
import threading # Prevent GUI freezing

from datetime import datetime # current time

# OBI Argolight data processing
from CalibLib import FullCalibrationStep2M1

class TestFrame(wx.Frame):
    def __init__(self):
        try:
            # High DPI monitor
            ctypes.windll.shcore.SetProcessDpiAwareness(True)
        except:
            pass

        wx.Frame.__init__(self, None, -1, "SV Calibration")
        # self.panel = wx.Panel(self)
        self.panel = scrolled.ScrolledPanel(self)
        
        self.createControl()
        self.layoutControl()
          
    def createControl(self):
        """create widgets (controls) for GUI"""
        panel = self.panel
        
        self.dirPic = wx.DirPickerCtrl(panel, -1, "", size=(800,-1))
        self.sysRadio = wx.RadioBox(panel, -1, 'System type (before/after sys 21)',
                               wx.DefaultPosition, (-1,80),
                               ['New','Legacy'], 2, wx.RA_SPECIFY_COLS)
        self.fileTextList = []
        self.laserChoList = []
        for idx in range(5):
            self.fileTextList.append( wx.StaticText(panel, -1, label="None", size=(400, 40), style=wx.ST_NO_AUTORESIZE) )
            self.laserChoList.append( wx.Choice(self.panel, -1, choices=['L473','L488','L532','L595','L647','L660','L750']) )
        
        self.runBtn = wx.Button(panel, -1, 'Run', size=(300,50))
        self.Bind(wx.EVT_BUTTON, self.runBtnClick, self.runBtn)
        
     
    def layoutControl(self):
        """sizers are used for layout (resizing and moving the widgets inside)"""
        # mainSizer is the top-level one that manages everything
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        
        mainSizer.Add(self.dirPic, 0, wx.EXPAND|wx.ALL, 10)
        mainSizer.Add(self.sysRadio, 0, wx.EXPAND|wx.ALL, 10)
        
        border = 10
        
        infoSizerList = [] 
        for idx in range(5):
            infoSizerList.append( wx.BoxSizer(wx.HORIZONTAL) )
            infoSizerList[idx].Add(self.fileTextList[idx], 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, border)
            infoSizerList[idx].Add(self.laserChoList[idx], 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, border)
            mainSizer.Add(infoSizerList[idx], 0, wx.EXPAND | wx.TOP, border)
        
        mainSizer.Add((10,10)) # some empty space
        mainSizer.Add(self.runBtn, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 10)
        
        # Connect the mainSizer to the container widget (panel)
        self.panel.SetSizer(mainSizer)
        
        # Fit the frame to the needs of the sizer. The frame will
        # automatically resize the panel as needed. Also prevent the
        # frame from getting smaller than this size.
        mainSizer.Fit(self)
        # mainSizer.SetSizeHints(self)  
        
        self.panel.SetupScrolling()
          
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
                if not self.checkLsgd(InFilePath):
                    wx.MessageBox("No lsgd file found.", "Run button",
                                  wx.OK | wx.ICON_WARNING)
                else:
                    SaoReconExe = self.pathSaoReconExe()
                    
                    if SaoReconExe == "":
                        wx.MessageBox("SaoRecon exe file not exist.", "Run button",
                              wx.OK | wx.ICON_WARNING)
                    else:
                        # To prevent GUI freezing,
                        # spawn a long running process onto a separate thread.
                        threading.Thread(target=self.runData,
                                         args=(SaoReconExe, InFilePath)).start()
                        # https://stackoverflow.com/questions/42422139/how-to-easily-avoid-tkinter-freezing
            
    def checkLsgd(self, InFilePath):
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
        # containing a path specification.
        findPath = InFilePath.joinpath('AcqData', '*.lsgd')
        numLsgd = len( glob.glob(str(findPath)) )
        if numLsgd != 0:
            return True
        else:
            return False
        
    def pathSaoReconExe(self):
        """
        Check SaoRecon exe file existence

        Parameters
        ----------
        InFilePath : pathlib.Path obj
            Path for the input data file.

        Returns
        -------
        SaoReconExe : Path
            Reconstruction exe file path.

        """
        # Between before/after sys 21, there is a flip between horizontal/vertical directions.
        # Use a different SaoRecon exe file for each system.
        sysType = self.sysRadio.GetStringSelection()
        
        if sysType == 'New':
            SaoReconExe = r'C:\Program Files\StellarVision\sv_r_v.2.9.118t.Inv.exe'
        elif sysType == 'Legacy':
            SaoReconExe = r'C:\Program Files\StellarVision\sv_r_v.2.9.118t.exe'
            
        if len(glob.glob(SaoReconExe)) == 0:
            return ""
        else:
            return Path(SaoReconExe)
        
    def runData(self, SaoReconExe, InFilePath):
        """
        reads a data file, processes the data, and saves the result

        Parameters
        ---------
        SaoReconExe : 
        InFilePath : pathlib.Path obj
            Path for the input data file.
        """
        # Original directory with raw data
        # - Parent directory (D:\SV_AcqData\2020-09-23 0.2um Bead001)
        #   with sub folders, AcqData (Raw data) and FullCalibStep2 (Result)
        FolderPathExpTitle_org = InFilePath
        # - Sub folder for raw data (D:\SV_AcqData\2020-09-23 0.2um Bead001\AcqData)
        FolderPathLsgd_org = FolderPathExpTitle_org.joinpath('AcqData')
        
        # Temporary working directory in C driver (ssd) for speed-up
        # - Parent diretory
        FolderPathExpTitle = Path(r'C:\SV work 1234')
        # - Sub folder for raw data
        FolderPathLsgd = FolderPathExpTitle.joinpath('AcqData')
        
        # Copy to c drive (ssd)
        if Path.is_dir(FolderPathLsgd):
            shutil.rmtree(FolderPathLsgd)
        shutil.copytree(FolderPathLsgd_org, FolderPathLsgd)
    
        # Empirical OnePixelFallOff/TwoPixelFallOff values for each laser
        dicOne = {'L473':0.62,
                  'L488':0.62,
                  'L532':0.71,
                  'L595':0.74,
                  'L647':0.8,
                  'L660':0.8,
                  'L750':0.88}
        dicTwo = {'L473':0.22,
                  'L488':0.22,
                  'L532':0.2,
                  'L595':0.24,
                  'L647':0.31,
                  'L660':0.31,
                  'L750':0.68}
        
        # glob.glob() returns a list of string path names that match pathname
        lsgdPath = FolderPathLsgd.joinpath('*.lsgd')
        fList = glob.glob(str(lsgdPath))
        
        # Between before/after sys 21, there is a flip between horizontal/vertical directions.
        # Use a different SaoRecon exe file for each system.
        sysType = self.sysRadio.GetStringSelection()
        
        if sysType == 'Legacy':
            for idx, fPath in enumerate(fList):
                self.fileTextList[idx].SetLabel(Path(fPath).name)
        
        procCalibList = []
        # For legacy system, check if the user assign the laser name for each data file.
        LaserNameReady = True
        
        for idx, fPath in enumerate(fList):
            # D:\SV_AcqData\2020-09-23 0.2um Bead001\AcqData\PREVIEW_L488.lsgd
            fName = Path(fPath).name
            
            if sysType == 'New':
                # PREVIEW_L488.lsgd
                DyeName = fName[fName.find('L'): 
                               fName.find('.lsgd')]
                OnePixelFallOff = dicOne[DyeName]
                TwoPixelFallOff = dicTwo[DyeName]
                LaserName = DyeName
            elif sysType == 'Legacy':
                LaserName = self.laserChoList[idx].GetStringSelection()
                if LaserName == "":
                    wx.MessageBox("Choose a right laser for each file", "Run button",
                              wx.OK | wx.ICON_WARNING)
                    LaserNameReady = False
                    break
                else:
                    DyeName = fName[len('PREVIEW_'): 
                               fName.find('.lsgd')]
                    OnePixelFallOff = dicOne[LaserName]
                    TwoPixelFallOff = dicTwo[LaserName]
            
            # For Windows, multiprocessing.Process() must be called inside
            # if __name__ == "__main__":
            # to prevent infinite succession of new processes.
            # https://stackoverflow.com/questions/20222534/python-multiprocessing-on-windows-if-name-main
            
            # multiprocessor for interactive python doesn't output print()
            # Spyder remedty
            # Run > Configuration per file > Execute in an external system terminal
            procCalib = multiprocessing.Process(target=FullCalibrationStep2M1, args=(sysType,
                                                                                     SaoReconExe,
                                                                                     FolderPathExpTitle,
                                                                                     DyeName,
                                                                                     LaserName,
                                                                                     OnePixelFallOff,
                                                                                     TwoPixelFallOff))
            # starting a calibration process 
            procCalib.start()
            procCalibList.append(procCalib)
    
        if LaserNameReady == False:
            pass
        else:
            # Wait until calibration processes are finished
            for idx in range(len(procCalibList)):
                procCalibList[idx].join()
        
            # Move & Clean
                
            # C:\SV work 1234\FullCalibStep2
            FolderPathResult = FolderPathExpTitle.joinpath('FullCalibStep2')
            # D:\SV_AcqData\2020-09-23 0.2um Bead001
            FolderPathResult_move = FolderPathExpTitle_org
            # If FullCalibStep2 folder exist, erase it.
            # Without this part, shutil.move shows an error for an existing folder.
            PathToCheck = FolderPathResult_move.joinpath('FullCalibStep2')
            if PathToCheck.exists():
                shutil.rmtree(PathToCheck)
            # Move 'FullCalibStep2' folder under the target directory    
            shutil.move(str(FolderPathResult), str(FolderPathResult_move))
            
            # Erase the temp working folder
            shutil.rmtree(FolderPathExpTitle)
            
            # Time to finish running
            timeStamp = datetime.now()
            strTimeStamp = timeStamp.strftime("%m/%d/%Y %H:%M:%S")
            # https://docs.python.org/3/library/time.html
        
            # Print on Screen    
            print()
            print(strTimeStamp)
            print()
            print('Full Calibration Step2 ended.')
            print('You can close the program now.')



############################################################################################
# wxPython (GUI) app running part

if __name__ == "__main__":
    # On Windows calling this function is necessary.
    multiprocessing.freeze_support()
    app = wx.App()
    frm = TestFrame()
    frm.Show()
    app.MainLoop()