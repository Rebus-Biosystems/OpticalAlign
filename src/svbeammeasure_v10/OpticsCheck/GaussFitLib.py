# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 2020

@author: YK from OBI

Reference:
https://scipy-cookbook.readthedocs.io/items/FittingData.html
https://gist.github.com/andrewgiessel/6122739
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
https://numpy.org/doc/stable/reference/generated/numpy.indices.html
https://numpy.org/devdocs/reference/generated/numpy.flip.html

update history
---------
2021-03-11  YK changed the following.
            1. Added more detailed comments like input parameters and output returns
            2. Added functions for bead data (randomly scattered data)
                - superGaussian()
                - momentsBead()
                - fitSuperGaussian()
"""
import numpy as np
from scipy import optimize

########################################################
# ArgoLight data (grid data)

def gaussian(height, center_x, center_y, width_x, width_y, rotation):
    """
    Returns a Gaussian function with the given parameters
    
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
    Returns Gaussian parameters of a 2D distribution by calculating its moments
    
    Parameters
    ---------
    data : 2D ndarray
    
    Returns
    ---------
    Returns height, x, y, width_x, width_y, rotation=0. :
        float, parameters for Gaussian function
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
    Returns Gaussian parameters of a 2D distribution found by a fitting
    
    Parameters
    ---------
    data : 2D ndarray
    
    Returns
    ---------
    height, x, y, width_x, width_y, rotation :
        float, parameters for Gaussian function
    """
    params = moments(data)
    # gaussian function takes x, y while np.indices generates [row, col].
    # np.flip needed to change from [row, col] to [col, row].
    # unpacking operator (*) to unpack [col, row] to (col, row).
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
    Returns error (deviation) matrix or values from Gaussian function
    
    Parameters
    ---------
    parameters : float tuple, Gaussian function parameters
    data : 2D ndarray
    plot : Bool, determines return value (matrix or values)
    
    Returns
    ---------
    errabsnorm : 2D ndarray, normalised error matrix (percent)
    errmean : float, average error (percent)
    errmax : float, maximum error (percent)
    """
    err = gaussian(*parameters)(*np.flip(np.indices(data.shape),0)) - data
    errabsnorm = 100 * np.abs( err / data )
    errmean = np.average(errabsnorm)
    errmax = np.max(errabsnorm)
    if plot == True:
        return errabsnorm
    else:
        return errmean, errmax
    
########################################################
# bead data (randomly scattered data)

def superGaussian(height, center_x, center_y, width_x, width_y, rotation, order):
    """
    Returns a super-Gaussian function with the given parameters
    
    Parameters
    ---------
    height, center_x, center_y, width_x, width_y, rotation, order :
        float, parameters for Gaussian function
        
    Returns
    ---------
    rotgauss : function, calculate super-Gaussian function value on (x,y)
    """
    width_x = float(width_x)
    width_y = float(width_y)

    rotationRad = np.deg2rad(rotation)
    
    def rotgauss(x,y):
        xs = x - center_x
        ys = y - center_y
        xp = xs * np.cos(rotationRad) - ys * np.sin(rotationRad)
        yp = xs * np.sin(rotationRad) + ys * np.cos(rotationRad)
        
        r2 = (xp/width_x)**2 + (yp/width_y)**2
        g = height * np.exp(- 1/2 * r2 ** (order/2) )
        
        return g
    
    return rotgauss

def momentsBead(XY, data):
    """
    Returns the Gaussian parameters of a 2D distribution by calculating its
    normal Gaussian (order=2) moments in the center area
    
    Parameters
    ---------
    XY : (X, Y) tuple, xy coordinates of single beads
    data : camera intensity value of each single bead
        
    Returns
    ---------
    height, x, y, width_x, width_y :
        float, parameters for Gaussian function
    """
    X, Y = XY
    total = data.sum()
    
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    
    # width calculation needs a data along a straight line.
    # beads are located in random locations, not in a fixed grid location.
    # pick up data within some finite deviation from a straight line
    deviationPix = 200
    
    # Horizontal width
    col_ind = np.argwhere( np.abs(Y-int(y)) < deviationPix )
    Xline = X[col_ind]
    col = data[col_ind]
    width_x = np.sqrt( np.abs((Xline-x)**2*col).sum() / col.sum() )
    
    # Vertical width
    row_ind = np.argwhere( np.abs(X-int(x)) < deviationPix )
    Yline = Y[row_ind]
    row = data[row_ind]
    width_y = np.sqrt( np.abs((Yline-y)**2*row).sum() / row.sum() )
    
    height = data.max()
    
    return height, x, y, width_x, width_y

def fitSuperGaussian(XY, data):
    """
    Returns the super_Gaussian parameters of a 2D distribution.
    
     
    Parameters
    ---------
    XY : (X, Y) tuple, xy coordinates of single beads
    data : camera intensity value of each single bead
        
    Returns
    ---------
    height, x, y, width_x, width_y, rotation, order :
        float, parameters for super-Gaussian function
    """
    params = momentsBead(XY, data)
    # Add rotation and order
    params = (*params, 0, 2)
    
    # scipy.optimize.least_squares(fun, x0, bounds=- inf, inf)
    # Parameters:   fun : callable, It must allocate and return a 1-D array_like of shape (m,) or a scalar.
    #               x0 : array_like with shape (n,) or float
    #               bounds : 2-tuple of array_like, optional
    # Returns:      result with data fields of (x, cost, fun, ...)
    
    # np.ravel() return a contiguous flattened array (1-D array).
    
    errorfunction = lambda p: np.ravel( superGaussian(*p)(*XY) - data )
    # height, x, y, width_x, width_y, rotation, order
    # bounds = [low bounds], [high bounds]
    bounds = [1, 1, 1, 1, 1, -90, 2], [1e5, 8192, 8192, 1e5, 1e5, 90, 20]
    
    result = optimize.least_squares(errorfunction, params, bounds = bounds)    
    (height, x, y, width_x, width_y, rotation, order) = result.x
    
    rotation_abs = np.abs(rotation)
    
    if rotation_abs > 45:
        rotation = rotation - int(rotation/rotation_abs)*90
        width_x, width_y = width_y, width_x
        
    return (height, x, y, width_x, width_y, rotation, order)