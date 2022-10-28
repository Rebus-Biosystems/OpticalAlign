import os 
from pathlib import Path
import pandas as pd 
import napari 
from magicgui import magicgui
import sys 
import numpy as np 
from matplotlib import image 
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from SVBeamMeasure import ArgoDataLib, GaussFitLib
from magicgui.widgets import FunctionGui, MainFunctionGui

@magicgui(
    auto_call=False, 
    layout='vertical',
    call_button="Generate SVBeamMeasure",
    result_widget=True,
    )
def SVBeamMeasureUI(viewer : napari.Viewer):
    
    try:
        all_layers = viewer.layers.selection
        for layer in all_layers:
            if layer._type_string=='image':
                print(layer.name)
                
                datas, normVals, slide = ArgoDataLib.argoNPH_direct(layer.data)
                data_size = datas[0].shape
                layer.refresh()
                layer_shapes = (layer.level_shapes)[0]
                resize_factor = layer_shapes[1]/data_size[1]
                datas_np = np.asarray(datas)
                
                layer_matadata = layer.metadata
                layer_path = layer_matadata['Path']
                
                if type(datas) == list:
                    noData = len(datas)
                else:
                    noData = 1
                
                
                img_data_filename_no_extension, save_dir_path, serial_number, lsrCho, pwrCho, expTxt = get_file_metadata_info(layer_path)
                
                list_of_saved_files = []
                for dIdx in range(noData):
                    infoCheck = False  
                    InFilePath = None 
                    fname = "None"
                    
                    fig = plotFig(infoCheck, InFilePath, datas, normVals,
                                        fname, slide, dIdx, noData)
    
                    fnameTitle = serial_number + "_" + img_data_filename_no_extension + "_Ch_" + str(dIdx+1) + "_" + lsrCho + "nm_" + pwrCho + "_" + expTxt + "_" + slide + ".png"
                    saving_path = os.path.join(save_dir_path, fnameTitle)
                    
                    list_of_saved_files.append(saving_path)
                                        
                    fig.savefig(saving_path)
                
                saved_img_all = []
                for saved_img in list_of_saved_files:
                    img_file = np.asarray(image.imread(saved_img))
                    os.remove(saved_img)
                    saved_img_all.append(img_file)
                
                fnameTitle_combined = serial_number + "_" + img_data_filename_no_extension + "_" + lsrCho + "nm_" + pwrCho + "_" + expTxt + "_" + slide + ".png"
                imgs_comb = saved_img_all[0]
                for i in range(1,len(saved_img_all)):
                    imgs_comb = np.concatenate((imgs_comb, saved_img_all[i]), axis=1)
                
                saved_path = os.path.join(save_dir_path, fnameTitle_combined)
                plt.imsave(saved_path, imgs_comb)
                
                print("Done performing SVBeamMeasure")

        return saved_path
    except Exception as e:
        print("Could not perform SVBeamMeasure")
        print(e)
        return f"Could not generate the SVBeamMeasure because of {e}"

   
def get_file_metadata_info(layer_path):
    
    """ From the layer path, gets the metada information and filenames to save """
    
    img_data_filename = os.path.basename(layer_path)
    img_data_filename_no_extension = Path(img_data_filename).resolve().stem
    metadata_file = img_data_filename_no_extension + ".AcqLog.txt"
    img_dirname = os.path.dirname(layer_path)
    metadata_file_fullpath = os.path.join(img_dirname, metadata_file)
    save_dir_path = os.path.join(img_dirname, "HeatMap")
    if not os.path.isdir(save_dir_path):
        os.mkdir(save_dir_path)
            
    if os.path.isfile(metadata_file_fullpath):
        df = pd.read_csv(metadata_file_fullpath, sep=": ", header=None, engine='python')
        print(df.head())
        df.columns = ["Item", "Value"]
                
        serial_number = "SN" + ((df.loc[df["Item"]=="System serial #", "Value"]).values)[0]
        lsrCho_ex_em = ((df.loc[df["Item"]=="Excitation (nm) - Emission (nm)", "Value"]).values)[0]
        lsrCho = lsrCho_ex_em.split(" - ")[0]
        pwrCho = ((df.loc[df["Item"]=="Laser power", "Value"]).values)[0]
        expTxt = ((df.loc[df["Item"]=="Camera exposure time per raw image frame", "Value"]).values)[0]
            
    else:
        serial_number = "SN_Unknown"
        lsrCho = "Unkown_lsrCho"
        pwrCho = "Unknown_pwrCho"
        expTxt = "Unknown_expTxt"
        
    return img_data_filename_no_extension,save_dir_path,serial_number,lsrCho,pwrCho,expTxt


def plotFig(infoCheck, InFilePath, datas, normVals,
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
    
    params = GaussFitLib.fitgaussian(data)
    gaussianFitFcn = GaussFitLib.gaussian(*params)
    posWidthRot = processParamsFig(params, x_cen, y_cen, slide)
    err = GaussFitLib.errgaussianPercent(params, data, plot=False)
    
    fig = Figure(figsize=(8.5, 4.8))        
    ax = fig.add_axes([-0.12, 0.1, 0.8, 0.8])
    cs = ax.imshow(data, cmap=plt.cm.jet)
    fig.colorbar(cs)
    ax.contour(gaussianFitFcn(*np.flip(np.indices(data.shape),0)), cmap=plt.cm.copper)        
    tickFig(ax, x_cen, y_cen)            
    showResultTxtFig(fig, posWidthRot, err, normVal)

    return fig


def tickFig(ax, x_cen, y_cen):
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
    ax.set_xticks( xtick_range + x_cen )
    ax.set_xticklabels( xtick_range )
    ax.set_yticks( ytick_range + y_cen )
    ax.set_yticklabels( ytick_range )
    
    return True 
        

def showResultTxtFig(fig, posWidthRot, err, normVal):
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
    
    fig.text(0.65, 0.2,  figTxt, fontsize=14, backgroundcolor='white')
    
            
def processParamsFig(params, x_cen, y_cen, slide):
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
 
  
