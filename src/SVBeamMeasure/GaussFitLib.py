# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 2020

@author: YK from OBI

2D Gaussian fitting is based on least-square optimization method.
The initial starting point for the optimization is calculated from moments and
fed into the iterative optimization algorithm.
This optimization algorithm needs an error function to minimize.
The error function takes Gaussian parameters as arguments with given data.

Reference:
https://scipy-cookbook.readthedocs.io/items/FittingData.html
https://gist.github.com/andrewgiessel/6122739
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
https://numpy.org/doc/stable/reference/generated/numpy.indices.html
https://numpy.org/devdocs/reference/generated/numpy.flip.html
"""
import numpy as np
from scipy import optimize


def gaussian(height, center_x, center_y, width_x, width_y, rotation):
    """
    Return a Gaussian function with the given parameters
    
    Parameters
    ---------
    height, center_x, center_y, width_x, width_y, rotation :
        float, parameters for Gaussian function
        
    Returns
    ---------
    rotgauss : function, calculate Gaussian function value on (x,y)
    """
    width_x = float(width_x)
    width_y = float(width_y)

    rotation = np.deg2rad(rotation)
    
    def rotgauss(x,y):
        xs = x - center_x
        ys = y - center_y
        xp = xs * np.cos(rotation) - ys * np.sin(rotation)
        yp = xs * np.sin(rotation) + ys * np.cos(rotation)
        g = height*np.exp(
            -((xp/width_x)**2+
              (yp/width_y)**2)/2.)
        return g
    return rotgauss

def moments(data):
    """
    Return gaussian parameters of a 2D distribution by calculating its moments
    
    Parameters
    ---------
    data : 2D ndarray
    
    Returns
    height, x, y, width_x, width_y, 0. (rotation)
    """
    total = data.sum()
    Y, X = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    row = data[int(y), :]
    width_x = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    col = data[:, int(x)]
    width_y = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    height = data.max()
    return height, x, y, width_x, width_y, 0.

def fitgaussian(data):
    """
    Return gaussian parameters of a 2D distribution by Gaussian fitting
    
    Parameters
    ---------
    data : 2D ndarray
    
    Returns
    height, x, y, width_x, width_y, rotation
    """
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.flip(np.indices(data.shape),0)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    
    (height, x, y, width_x, width_y, rotation) = p
    rotation_abs = np.abs(rotation)
    if rotation_abs >= 180:
        rotation_new = rotation - round(rotation/180)*180
    else:
        rotation_new = rotation
            
    params = (height, x, y, width_x, width_y, rotation_new)
    p, success = optimize.leastsq(errorfunction, params)
    
    (height, x, y, width_x, width_y, rotation) = p
    rotation_abs = np.abs(rotation)
    
    if 45 < rotation_abs < (90+45):
        rotation = rotation - int(rotation/rotation_abs)*90
        width_x, width_y = width_y, width_x
    elif rotation_abs >= (90+45):
        rotation = rotation - int(rotation/rotation_abs)*180
        
    return (height, x, y, width_x, width_y, rotation)

def errgaussianPercent(parameters, data, plot):
    """
    Calculate deviation from a 2D Gaussian function.
    
    Parameters
    ---------
    parameters : height, x, y, width_x, width_y, rotation
    data : 2D ndarray
    plot : Boolean, if plotting 2D error map
    
    Returns
    errabsnorm : 
    errmean :
    errmax : 
    """
    gaussianData = gaussian(*parameters)(*np.flip(np.indices(data.shape),0))
    err = gaussianData - data
    errabsnorm = 100 * np.abs( err / gaussianData )
    errmean = np.average(errabsnorm)
    errmax = np.max(errabsnorm)
    if plot == True:
        return errabsnorm
    else:
        return errmean, errmax