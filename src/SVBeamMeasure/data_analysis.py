# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:53:58 2020

@author: YK
"""

from pathlib import Path
from ArgoDataLib import *
from GaussFitLib import *
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np

def showresult(data, slide):
    y_size, x_size = data.shape
    
    x_cen = int((x_size - 1)/2)
    y_cen = int((y_size - 1)/2)
    
    params = fitgaussian(data)
    fit = gaussian(*params)
    
    (height, x, y, width_x, width_y, rotation) = params
    (x, y) = (x-x_cen, -(y-y_cen))
    errmean, errmax = errgaussianPercent(params, data, plot=False)
    
    dPixelUm = 0.325
    if slide == 'LM': 
        dSpotSpacingUm = 20
    elif slide == 'Homo':
        dSpotSpacingUm = 15
    fovUm = 2048*dPixelUm
    
    widthUm = 1100
    
    # Gaussian width to FWHM
    widthToFWHM = 2.35482
    width_x, width_y = tuple([x*widthToFWHM for x in (width_x, width_y)])
    
    # pixel to um
    xUm, yUm, width_xUm, width_yUm = tuple([x*dSpotSpacingUm
                                            for x in (x, y, width_x, width_y)])
    
    # # um to percent
    # xPct, yPct = tuple([x/fovUm*100 for x in (xUm, yUm)])
    # width_xPct, width_yPct = tuple([x/widthUm*100 for x in (width_xUm, width_yUm)])
    
    # return (xPct, yPct, width_xPct, width_yPct, rotation, errmean, errmax)
    return (xUm, yUm, width_xUm, width_yUm, rotation, errmean, errmax)
    
    
# %%
data_path = r'\\MA2FILES\Production systems\SN1021 (SV20C)\ArgoData from Chemistry team\2020-01-30_ArgoHomo_01\AcqData'
data_path = Path(data_path)
lnames = [488, 532, 595, 647]
    
path = Path(r'\\MA2FILES\Production systems\SN1021 (SV20C)\ArgoData from Chemistry team')
dirs = [e for e in path.iterdir() if e.is_dir()]
# http://zetcode.com/python/pathlib/
dirsRight = []

for dir in dirs:
    dirAcqData = dir.joinpath('AcqData')
    goodDir = True
    
    for lname in lnames:
        fnameTxt = '*' + str(lname) + '*.lsgd'
        fname = list(dirAcqData.glob(fnameTxt))
        
        if len(fname) == 0:
            goodDir = False
    
    if goodDir == True:
        dirsRight.append(dirAcqData)

# Bad datae: 2020-02-04_ArgoHomo_01, 2020-02-07_ArgoHomo 01
dirsRight.remove(dirsRight[2])
dirsRight.remove(dirsRight[2])

xList = [[],[],[],[]]
yList = [[],[],[],[]]
dList = [[],[],[],[]]
        
for dir in dirsRight:
    for lIdx in range(len(lnames)):
        fnameTxt = '*' + str(lnames[lIdx]) + '*.lsgd'
    
        fname = list(dir.glob(fnameTxt))
        file_to_open = fname[0]
        
        data = argoAnal(file_to_open, 'Homo')
        (x, y, width_x, width_y, rotation, errmean, errmax) = showresult(data, 'Homo')
        xList[lIdx].append(x); yList[lIdx].append(y)
        
        d = math.sqrt(x**2 + y**2)
        dList[lIdx].append(d)

plt.figure()        
for lIdx in range(len(lnames)):
    plt.plot(xList[lIdx])
plt.legend([str(x) for x in lnames])
plt.ylabel('x (um)')
plt.title('Sys 21 beam position x (um) for 6 months')

plt.figure()        
for lIdx in range(len(lnames)):
    plt.plot(yList[lIdx])
plt.legend([str(x) for x in lnames])
plt.ylabel('y (um)')
plt.title('Sys 21 beam position y (um) for 6 months')

plt.figure()        
for lIdx in range(len(lnames)):
    plt.plot(dList[lIdx])
plt.legend([str(x) for x in lnames])
plt.ylabel('d (um)')
plt.title('Sys 21 beam position d (um) for 6 months')
