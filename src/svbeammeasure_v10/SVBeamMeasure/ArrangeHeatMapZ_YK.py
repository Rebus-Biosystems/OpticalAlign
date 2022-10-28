# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 12:58:46 2020

@author: YK
"""
from pathlib import Path
from PIL import Image

InFilePath = r'D:\Optical Biosystems\Regular work\RND\2020-07-14 Beam position\SN1021 (SV20C)\2020-07-02 Argilight_test\HeatMap\SN1021, 2020-07-02, Ch1, Ref, 488 500mw@200ms _HeatMap1_HeatMap1Gauss.png'
InFilePath = Path(InFilePath)

fname = InFilePath.name

isRefTest = False
isAllFiles = True

isAllFiles = checkCh(InFilePath)

if fname.find('Ref') != -1:
    isRefTest = True    
    
    findPath = fname.replace('Ref', 'Test')
    findPath = InFilePath.parent.joinpath(findPath)
    isAllFiles = checkCh(findPath)
elif fname.find('Test') != -1:
    isRefTest = True
    
    findPath = fname.replace('Test', 'Ref')
    findPath = InFilePath.parent.joinpath(findPath)
    isAllFiles = checkCh(findPath)
else:
    isRefTest = False
    
# after file check
if isAllFiles == False:
    # err msg
    pass
else:
    if isRefTest:
        resultImg = mrgRefTest(InFilePath)
    else:
        resultImg = mrgHor(InFilePath)
        
    resultImg.save(fnameOut(InFilePath))
       
def mrgRefTest(InFilePath):
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
    fname = InFilePath.name
    
    for ind_ch in range(4):
        fname = fname.replace(' Ch'+str(ind_ch+1)+',', "")
    
    fname = fname.replace(' Ref,', "")
    fname = fname.replace(' Test,', "")
    
    OutFilePath = InFilePath.parent.joinpath(fname)
    
    return OutFilePath

def checkCh(InFilePath):
    fname = InFilePath.name
    result = True
    
    for ind_ch in range(4):
        chPos = fname.find('Ch')
        findPath = fname.replace(fname[chPos:chPos+3], fname[chPos:chPos+2]+str(ind_ch+1))
        findPath = InFilePath.parent.joinpath(findPath)
        
        if findPath.is_file():
            pass
        else:
            result = False
        
    return result
 

 