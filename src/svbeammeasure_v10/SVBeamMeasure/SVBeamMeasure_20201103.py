# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 2020

@author: YK from OBI

update history
---------
2020-08-21  YK changed the following.
            1. added scrollbar function to enable decreased window size
            according to Vinh's request
            2. added result text for d (sqrt(x^2+y^2)) and heat map norm value
            accoring to Hojin's request
            3. keep input file option according to Vinh's request
            
2020-08-23  YK changed the following.
            1. make file open dialog start from the file path text box input
            2. implement result image merge function
            
2020-11-03  YK changed the following.
            1. wx.ALIGN_RIGHT in wxBoxSizer removed (wxPython 4.1 bug).
            2. update to differentiate different Argolight slides
                - 'Argolight slide' radio box removed.
                - ArgoDataLib updated to differentiate different Argolight slides
                - runData function
                    datas for lsgd file changed from array to list
                - titleFig function
                    file name for lsgd file changed from "_Ch" to ", Ch"
                    (the same as tif file case)
                    
            Frozen successfully with the following version of softwares:
            1. Windows 10
            2. Anaconda3-2020.07-Windows-x86_64.exe
            3. Python 3.8.3
            4. matplotlib 3.3.4
            5. PyInstaller 4.2
            
            !pyinstaller --onedir -y -w SVBeamMeasure_20201103.py
"""
# path, directory, file name, file extension
from pathlib import Path
# copy a file
import shutil

import numpy as np

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# GUI library (wxPython)
import wx
import wx.lib.scrolledpanel as scrolled

import ctypes

# OBI Argolight data processing
import ArgoDataLib
import GaussFitLib


class TestFrame(wx.Frame):
    def __init__(self):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(True)
        except:
            pass

        wx.Frame.__init__(self, None, -1, "SV Beam measure")
        # self.panel = wx.Panel(self)
        self.panel = scrolled.ScrolledPanel(self)
        
        self.createControl()
        self.layoutControl()
          
    def createControl(self):
        """create widgets (controls) for GUI"""
        panel = self.panel
        
        dictLbl = {'fname': "File: ",
                   'file': "...",
                   'info': "system info to output file name",
                   'keep': "keep input file",
                   'sn': "SN: ",
                   'date': "Date (y-m-d): ",
                   'ch': "Ch: ",
                   'arm': "Arm: ",
                   'lsr': "Laser: ",
                   'pwr': "Power (mW): ",
                   'exp': "Exposer (ms): ",
                   'run': "Run",
                   'mrg': "Merge files"}
        
        dictList = {'yr': [str(x) for x in list(range(2020,2031))],
                    'mo': ['%02d' % x for x in list(range(1,13))],
                    'day': ['%02d' % x for x in list(range(1,32))],
                    'ch': [str(x) for x in list(range(1,5))],
                    'arm': ['Ref', 'Test'],
                    'lsr': ['473', '488', '532', '595', '647', '750'],
                    'pwr': ['500', '2000']}
        
        self.fnameLbl = self.createLabel(dictLbl['fname'])
        self.fileTxt = wx.TextCtrl(panel, -1, "", size=(150,-1))
        self.fileBtn = wx.Button(panel, -1, dictLbl['file'], size=(50,35))
        self.Bind(wx.EVT_BUTTON, self.fileBtnClick, self.fileBtn)
        
        self.infoCbx = wx.CheckBox(panel, label = dictLbl['info'])
        self.keepCbx = wx.CheckBox(panel, label = dictLbl['keep'])
        
        self.snLbl = self.createLabel(dictLbl['sn'])
        self.snTxt = wx.TextCtrl(panel, -1, "", size=(60,-1))
        
        self.dateLbl = self.createLabel(dictLbl['date'])
               
        self.yrCho = self.createChoice(dictList['yr'])
        self.moCho = self.createChoice(dictList['mo'])
        self.dayCho = self.createChoice(dictList['day'])
        
        self.chLbl = self.createLabel(dictLbl['ch'])
        self.chCho = self.createChoice(dictList['ch'])
        
        self.armLbl = self.createLabel(dictLbl['arm'])
        self.armCho = self.createChoice(dictList['arm'])
        
        self.lsrLbl = self.createLabel(dictLbl['lsr'])
        self.lsrCho = self.createChoice(dictList['lsr'])
        
        self.pwrLbl = self.createLabel(dictLbl['pwr'])
        self.pwrCho = self.createChoice(dictList['pwr'])
        
        self.expLbl = self.createLabel(dictLbl['exp'])
        self.expTxt = wx.TextCtrl(panel, -1, "", size=(60,-1))
        
        self.runBtn = wx.Button(panel, -1, dictLbl['run'], size=(200,35))
        self.Bind(wx.EVT_BUTTON, self.runBtnClick, self.runBtn)
        
        self.mrgBtn = wx.Button(panel, -1, dictLbl['mrg'], size=(200,35))
        self.Bind(wx.EVT_BUTTON, self.mrgBtnClick, self.mrgBtn)
        
        self.fig = Figure(figsize=(8.5, 4.8))        
        self.canvas = FigureCanvas(panel, -1, self.fig)
        
    def createLabel(self, lbl):
        return wx.StaticText(self.panel, -1, lbl)
    
    def createChoice(self, listCho):
        """pull-down choice (drop-down list)"""
        return wx.Choice(self.panel, -1, choices=listCho)
        
    def layoutControl(self):
        """sizers are used for layout (resizing and moving the widgets inside)"""
        # mainSizer is the top-level one that manages everything
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        
        # create children sizers
        fileSizer = self.createFileSizer()
        optSizer = self.createOptSizer()
        infoSizer = self.createInfoSizer()
        runSizer = self.createRunSizer()
        plotSizer = self.createPlotSizer()
        
        # add the children sizers to the mainSizer
        # Add (sizer, proportion=0, flag=0, border=0, userData=None)
        #   proportion : weight for change its size (0 means non-changeable)
        #   flag : control sizer’s behaviour
        #   border : empty space
        #   userData : an extra object to be attached to the sizer
        
        # with wx.TOP flag, insert empty border among sizers
        border = 10
        
        mainSizer.Add(fileSizer, 0, wx.EXPAND)
        mainSizer.Add(optSizer, 0, wx.EXPAND | wx.TOP, border)
        mainSizer.Add(infoSizer, 0, wx.EXPAND | wx.TOP, border)
        mainSizer.Add(runSizer, 0, wx.EXPAND | wx.TOP, border)
        mainSizer.Add(plotSizer, 0, wx.EXPAND | wx.TOP, border)
        
        # Connect the mainSizer to the container widget (panel)
        self.panel.SetSizer(mainSizer)
        
        # Fit the frame to the needs of the sizer. The frame will
        # automatically resize the panel as needed. Also prevent the
        # frame from getting smaller than this size.
        mainSizer.Fit(self)
        # mainSizer.SetSizeHints(self)  
        
        self.panel.SetupScrolling()
        
    def createFileSizer(self):
        """sizer to open data file"""
        fileSizer = wx.BoxSizer(wx.HORIZONTAL)
        fileSizer.Add(self.fnameLbl, 0,
                      wx.ALIGN_CENTER_VERTICAL|wx.LEFT, 10)
        fileSizer.Add(self.fileTxt, 1,
                      wx.ALIGN_CENTER_VERTICAL|wx.LEFT|wx.RIGHT, 5)
        fileSizer.Add(self.fileBtn, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 10)
        
        return fileSizer
    
    def createOptSizer(self):
        """sizer for slide option and system info to file name option 
            (serial number, laser, etc.)"""
        optSizer = wx.BoxSizer(wx.HORIZONTAL)
        optSizer.Add(self.infoCbx, 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, 10)
        optSizer.Add(self.keepCbx, 0, wx.ALIGN_CENTER_VERTICAL)
        
        return optSizer
    
    def createInfoSizer(self):
        """the system info (serial number, laser, etc.)"""
        # with wx.LEFT or wx.RIGHT flag, insert empty border among widgets (controls)
        border = 10
        
        infoSizer = wx.BoxSizer(wx.HORIZONTAL)
            
        infoSizer.Add(self.snLbl, 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, border)
        infoSizer.Add(self.snTxt, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, border)
        
        infoSizer.Add(self.dateLbl, 0, wx.ALIGN_CENTER_VERTICAL)
        infoSizer.Add(self.yrCho, 0, wx.ALIGN_CENTER_VERTICAL)
        infoSizer.Add(self.moCho, 0, wx.ALIGN_CENTER_VERTICAL)
        infoSizer.Add(self.dayCho, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, border)
        
        infoSizer.Add(self.chLbl, 0, wx.ALIGN_CENTER_VERTICAL)
        infoSizer.Add(self.chCho, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, border)
        
        infoSizer.Add(self.armLbl, 0, wx.ALIGN_CENTER_VERTICAL)
        infoSizer.Add(self.armCho, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, border)
        
        infoSizer.Add(self.lsrLbl, 0, wx.ALIGN_CENTER_VERTICAL)
        infoSizer.Add(self.lsrCho, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, border)
        
        infoSizer.Add(self.pwrLbl, 0, wx.ALIGN_CENTER_VERTICAL)
        infoSizer.Add(self.pwrCho, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, border)
        
        infoSizer.Add(self.expLbl, 0, wx.ALIGN_CENTER_VERTICAL)
        infoSizer.Add(self.expTxt, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, border)
        
        return infoSizer
    
    def createRunSizer(self):
        """runSizer holds a run button"""
        # The button row is added as a horizontal box sizer
        # with some empty spacer elements.
        # The spacers are given a proportion of 1 (changeable)
        # and the button has 0 proportion (non-changeable).
        # The button size is fixed, while empty spacers freely stretch.
        runSizer = wx.BoxSizer(wx.HORIZONTAL)
        runSizer.Add((20,20), 1)
        runSizer.Add(self.runBtn, 0, wx.ALIGN_CENTER_VERTICAL)
        runSizer.Add((40,-1), 0)
        runSizer.Add(self.mrgBtn, 0, wx.ALIGN_CENTER_VERTICAL)
        runSizer.Add((20,20), 1)
        
        return runSizer
    
    def createPlotSizer(self):
        """plotSizer to show the result of laser beam characterization"""
        plotSizer = wx.BoxSizer(wx.HORIZONTAL)
        plotSizer.Add((20,-1), 1)
        plotSizer.Add(self.canvas, 0, wx.ALIGN_CENTER)
        plotSizer.Add((20,-1), 1)
        
        return plotSizer
        
    def fileBtnClick(self, event):
        """file button click event handler (File dialog)"""
        wildcard = "Valid data file (*.tif,*.lsgd,*.png)|*.tif;*.lsgd;*.png|" \
            "All files (*.*)|*.*"
        defaultDir = self.fileTxt.GetValue()
        defaultDir = Path( defaultDir ).parent
        defaultDir = str(defaultDir)
        dialog = wx.FileDialog(None, "Choose a file", defaultDir,
                               "", wildcard, wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            self.fileTxt.Clear()
            self.fileTxt.SetValue(dialog.GetPath())
        dialog.Destroy()

    def runBtnClick(self, event):
        """run button click event handler (initiate data processing)"""        
        InFilePath = self.fileTxt.GetValue()
        infoCheck = self.infoCbx.IsChecked()
        keepCheck = self.keepCbx.IsChecked()
        
        if InFilePath != "":
            InFilePath = Path(InFilePath)
            self.runData(InFilePath, infoCheck, keepCheck)
        else:
            wx.MessageBox("File path not entered.", "Run button",
                            wx.OK | wx.ICON_WARNING)
        
    def runData(self, InFilePath, infoCheck, keepCheck):
        """
        read a data file, processe the data, and show the result

        Parameters
        ---------
        InFilePath : pathlib.Path obj, path for the input data file
        infoCheck : bool, option about system info transfered to file name
        keepCheck : bool, option to keep the input file
        """         
        # reading a data file and processed data returns
        
        # change for datas
        # before : tif file (N, N) array lsgd file (4, N, N) array
        # after  : tif file (N, N) array lsgd file 4 element list
        
        # datas, normVals = ArgoDataLib.argoAnal(InFilePath, slide)
        datas, normVals, slide = ArgoDataLib.argoAnal(InFilePath)
        
        if type(datas) == list:
            noData = len(datas)
        else:
            noData = 1
        
        if infoCheck == True:
            fname = self.sysInfoFileName(InFilePath.suffix)
        else:
            fname = InFilePath.stem
        
        for dIdx in range(noData):
            fnameTitle = self.plotFig(infoCheck, InFilePath, datas, normVals,
                                      fname, slide, dIdx, noData)
            
        # change the input data file name if the user wants
        if infoCheck == True:
            self.changeDataFileName(InFilePath, fnameTitle, slide, keepCheck)
            
    def sysInfoFileName(self, suffix):
        """
        returns a string about system information
        
        Parameters
        ---------
        slide : string, sample slide (LM or Homo)
        
        Returns
        ---------
        fname : string, system information for a figure title and a file name 
        """
        if suffix == '.tif':
            fname = ("SN" + self.snTxt.GetValue() + ", " +
                      self.yrCho.GetStringSelection() + "-" +
                      self.moCho.GetStringSelection() + "-" +
                      self.dayCho.GetStringSelection() + ", Ch" +
                      self.chCho.GetStringSelection() + ", " +
                      self.armCho.GetStringSelection() + ", " +
                      self.lsrCho.GetStringSelection() + " " +
                      self.pwrCho.GetStringSelection() + "mW@" +
                      self.expTxt.GetValue() + "ms")
        elif suffix == '.lsgd':
            fname = ("SN" + self.snTxt.GetValue() + ", " +
                      self.yrCho.GetStringSelection() + "-" +
                      self.moCho.GetStringSelection() + "-" +
                      self.dayCho.GetStringSelection() + ", Ch")
            
        return fname
    
    def changeDataFileName(self, InFilePath, fnameTitle, slide, keepCheck):
        """
        change the input data file name with one containing system info
        
        Parameters
        ---------
        InFilePath : pathlib.Path obj, path for the input data file
        fnameTitle : string, system information for figure title
        slide : string, sample slide (LM or Homo)
        """
        old_extension = InFilePath.suffix
        directory = InFilePath.parent
        new_name = fnameTitle + old_extension
        if slide == 'Homo':
            new_name.replace(', Ch4','')
        InFilePathNew = Path(directory, new_name)
        if keepCheck == True:
            shutil.copy(str(InFilePath), str(InFilePathNew))
        else:
            InFilePath.rename(InFilePathNew)

    def plotFig(self, infoCheck, InFilePath, datas, normVals,
                fname, slide, dIdx, noData):
        """
        Post-data processing and displaying result figure
        
        Parameters
        ---------
        infoCheck : bool, option about system info transfered to file name
        InFilePath : pathlib.Path obj, path for the input data file
        datas : ndarray, heat map image array
        normVals : ndarray, heat map normalization value
        fname : string, input data file name or system information 
        slide : string, sample slide (LM or Homo)
        dIdx : data index for nth 2D heat map image
        noData : number of 2D heat map images
        
        Returns
        ---------
        fnameTitle : string, system information for figure title
        """
        if noData == 1:
            data = datas
            normVal = normVals 
        else:
            data = datas[dIdx]
            normVal = normVals[dIdx]
            
        y_size, x_size = data.shape
    
        x_cen = int((x_size - 1)/2)
        y_cen = int((y_size - 1)/2)
        
        # data -> params = (height, x, y, width_x, width_y, rotation)
        params = GaussFitLib.fitgaussian(data)
        # (height, x, y, width_x, width_y, rotation) -> Gaussian function
        gaussianFitFcn = GaussFitLib.gaussian(*params)
        # (height, x, y, width_x, width_y, rotation)
        # from pix to um & additional data processing
        posWidthRot = self.processParamsFig(params, x_cen, y_cen, slide)
        # error (deviation) from Gaussian function
        err = GaussFitLib.errgaussianPercent(params, data, plot=False)
        
        self.fig.clear()
        self.ax = self.fig.add_axes([-0.12, 0.1, 0.8, 0.8])
        
        cs = self.ax.imshow(data, cmap=plt.cm.jet)
        self.fig.colorbar(cs)
    
        self.ax.contour(gaussianFitFcn(*np.flip(np.indices(data.shape),0)), cmap=plt.cm.copper)        
    
        self.tickFig(x_cen, y_cen)

        fnameTitle = self.titleFig(fname, InFilePath.suffix, slide, infoCheck, dIdx)
                
        self.showResultTxtFig(posWidthRot, err, normVal)
        
        self.canvas.draw()
        
        self.saveFig(InFilePath, fnameTitle)
        
        return fnameTitle
        
    def processParamsFig(self, params, x_cen, y_cen, slide):
        """
        Convert Gaussian fit parameters in pixel into the result in um
        with data processing
        
        Parameters
        ---------
        params : tuple, (height, x, y, width_x, width_y, rotation) (pixel, deg)
        x_cen, y_cen : int, center of the field-of-view in pixel 
        slide : string, sample slide (LM or Homo)
        
        Returns
        ---------
        xUm, yUm, dUm : float, laser beam center position in xy coordinates (d = sqrt(x^2+y^2)) (um)
        width_xUm - widthUm, width_yUm - widthUm : float, laser beam width error in horizontal and vertical directions (um)
        rotation : string, laser beam rotation angle (deg)
        """
        (height, x, y, width_x, width_y, rotation) = params
        # coordinate conversion
        # from the original (0,0) position is upper left corner
        # to the center of the field-of-view
        (x, y) = (x-x_cen, -(y-y_cen))
   
        # The distance between spots in Argolight slide 
        if slide == 'LM': 
            dSpotSpacingUm = 20
        elif slide == 'Homo':
            dSpotSpacingUm = 15
        
        # The laser beam FWHM specification
        widthUm = 1100
        
        # GaussFitLib.fitgaussian() returns Gaussian width
        # Gaussian width to FWHM
        widthToFWHM = 2.35482
        width_x, width_y = tuple([x*widthToFWHM for x in (width_x, width_y)])
        
        # pixel to um
        xUm, yUm, width_xUm, width_yUm = tuple([x*dSpotSpacingUm
                                                for x in (x, y, width_x, width_y)])
        dUm = np.sqrt(xUm**2 + yUm**2)
        
        # for beam width, deviation from the spec value is returned
        return (xUm, yUm, dUm, width_xUm - widthUm, width_yUm - widthUm, rotation)
        
    def tickFig(self, x_cen, y_cen):
        """
        Set ticks and ticklabels properly
        
        Parameters
        ---------
        params : tuple, (height, x, y, width_x, width_y, rotation)
        x_cen, y_cen : int, center of the field-of-view in pixel 
        slide : string, sample slide (LM or Homo)
        """
        # ticks are display for every tick_step
        tick_step = 5
        
        xtick_radius = tick_step*int(x_cen/tick_step)
        ytick_radius = tick_step*int(y_cen/tick_step)
        
        xtick_range = np.arange(-xtick_radius, xtick_radius+1, tick_step)
        ytick_range = np.arange(-ytick_radius, ytick_radius+1, tick_step) 
        
        # The default (0,0) position is upper-left corner.
        # Set the tick position based on the original coordinate
        # Put the tick label with wanted numbers
        self.ax.set_xticks( xtick_range + x_cen )
        self.ax.set_xticklabels( xtick_range )
        self.ax.set_yticks( ytick_range + y_cen )
        self.ax.set_yticklabels( ytick_range )
        
    def titleFig(self, fname, suffix, slide, infoCheck, dIdx):
        """
        Generate and set a figure title string with system information (option)
        
        Parameters
        ---------
        fname : string, input data file name or system information 
        slide : string, sample slide (LM or Homo)
        infoCheck : bool, option about system info transfered to file name
        dIdx : data index for nth 2D heat map image
        
        Returns
        ---------
        fnameTitle : string, a figure title string with system information
        """
        if suffix == '.tif':
            fnameTitle = fname
        elif suffix == ".lsgd":
            if infoCheck == False:
                # fnameTitle = fname + "_Ch" + str(dIdx+1)
                fnameTitle = fname + ", Ch" + str(dIdx+1)
            else:
                fnameTitle = (fname + str(dIdx+1) + ", " +
                          self.lsrCho.GetStringSelection() + " " +
                          self.pwrCho.GetStringSelection() + "mW@" +
                          self.expTxt.GetValue() + "ms")
        if ", " + slide in fnameTitle:
            pass
        else:
            fnameTitle = fnameTitle + ", " + slide
                
        self.ax.set_title(fnameTitle, fontsize=14, x=0.7, y=1.02)
        return fnameTitle
    
    def showResultTxtFig(self, posWidthRot, err, normVal):
        """
        Show the final result as a text in the figure
        
        Parameters
        ---------
        posWidthRot : tuple, (x, y, width_x, width_y, rotation)
        err : tuple, (errmena, errmax)
        """
        figTxt = ("x (um): %.1f\n"
                  "y (um): %.1f\n"
                  "d (um): %.1f\n\n"
                  "width_x (um): %.1f\n"
                  "width_y (um): %.1f\n\n"
                  "rotation (deg): %.1f\n\n"
                  "errmean (%%): %1.f\n"
                  "errmax (%%): %1.f\n\n"
                  "heat map norm: %d")
        figTxt = figTxt % (posWidthRot + err + (normVal,))
        
        self.fig.text(0.65, 0.2,  figTxt, fontsize=14, backgroundcolor='white')
        
    def saveFig(self, InFilePath, fnameTitle):
        """
        Save the final result into a file
        
        Parameters
        ---------
        InFilePath : pathlib.Path obj, path for the input data file
        fnameTitle : string, system information for figure title
        """
        # make output directory if not exits
        OutFilePath = InFilePath.parent / 'HeatMap'
        
        if Path.is_dir(OutFilePath) != True:
            Path.mkdir(OutFilePath)
        
        fnameSave = fnameTitle + "_HeatMap1Gauss.png"
        
        OutFileNamePng = OutFilePath.joinpath(fnameSave)

        self.fig.savefig(OutFileNamePng)
        
    def mrgBtnClick(self, event):
        """merge button click event handler"""
        InFilePath = self.fileTxt.GetValue()
        InFilePath = Path(InFilePath)
        
        fname = InFilePath.name

        # Two different data
        # - Reference and test arms separately measured data 
        # - Both arms interfered data
        isRefTest = False
        isAllFiles = True
        
        # check if all 4 channel files exist
        isAllFiles = ArgoDataLib.checkCh(InFilePath)
        
        # check ref and test separately measured or not
        if fname.find('Ref') != -1:
            isRefTest = True    
            
            findPath = fname.replace('Ref', 'Test')
            findPath = InFilePath.parent.joinpath(findPath)
            # check if all 4 channel files exist
            isAllFiles = ArgoDataLib.checkCh(findPath)
        elif fname.find('Test') != -1:
            isRefTest = True
            
            findPath = fname.replace('Test', 'Ref')
            findPath = InFilePath.parent.joinpath(findPath)
            # check if all 4 channel files exist
            isAllFiles = ArgoDataLib.checkCh(findPath)
        else:
            isRefTest = False
            
        # after file check
        if isAllFiles == False:
            wx.MessageBox("All 4 channel files needed. Check the file existence or name.",
                          "Merge files button", wx.OK | wx.ICON_WARNING)
        else:
            if isRefTest:
                resultImg = ArgoDataLib.mrgRefTest(InFilePath)
            else:
                resultImg = ArgoDataLib.mrgHor(InFilePath)
            
            OutFilePath = ArgoDataLib.fnameOut(InFilePath)
            resultImg.save(OutFilePath)
            
            # show merged image to the window
            self.fig.clear()
            self.ax = self.fig.add_axes([0, 0, 1, 1])
            self.ax.imshow(resultImg)
            # make axis tick invisible
            self.ax.axes.xaxis.set_visible(False)
            self.ax.axes.yaxis.set_visible(False)
            self.canvas.draw() 
            
# %% wxPython (GUI) app running part
app = wx.App()
frm = TestFrame()
frm.Show()
app.MainLoop()