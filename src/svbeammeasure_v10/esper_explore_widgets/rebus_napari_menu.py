
import os
import sys
import time
from glob import glob
from pathlib import Path
from typing import Annotated
import dask.array as da
from imageio import volread
from imageio.v2 import imread
from napari import Viewer
from napari import run as run_napari
from napari._qt.menus._util import NapariMenu, populate_menu
from napari._qt.menus.file_menu import FileMenu
from napari._qt.qt_main_window import Window
from napari.utils.history import get_open_history, update_open_history
from napari.utils.misc import in_ipython
from napari.utils.translations import trans
from numpy import fromfile, uint16
from qtpy.QtWidgets import QFileDialog, QListWidget
from magicgui import magicgui, widgets
from PyQt5.QtWidgets import QHBoxLayout, QWidget, QListView, QTreeView
# import PyQt5.QtCore

# THIS_SCRIPT_DIR = os.path.dirname(__file__)
# sys.path.insert(0, os.path.join(THIS_SCRIPT_DIR, "../esper_explore_widgets"))
# sys.path.insert(0, THIS_SCRIPT_DIR)

from .fov_utils import read_lsgd_as_imagestack
from .napari_file_explore_widget import FolderBrowser

IMG_SIZE_Y = 2048 
RAW_DATA_FORMATS = ['lsgd', 'lout']
PYRAMID_DATA_FORMATS = ['zarr']
Num_pyramid_layers = 16
MAX_CONTRAST_LIMIT = 2500


class RebusFileMenu(FileMenu):
    def __init__(self, window: 'Window'):
        self._win = window
        super().__init__(window)
        ACTIONS = [
        {
            'text': trans._('Open File(s)...'),
            'slot': window._qt_viewer._open_files_dialog,
            'shortcut': 'Ctrl+O',
        },
        ]   
        populate_menu(self, ACTIONS)
   

class RebusMenu(NapariMenu):
    
    def __init__(self, viewer, window: 'Window'):
        self._win = window
        self.viewer = viewer 
        super().__init__(trans._('&RebusMenu'), window._qt_window)
        ACTIONS = [
        {
            'text': trans._('Open File(s) ...'),
            'slot': window._qt_viewer._open_files_dialog,
            'shortcut': 'Ctrl+O',
        },
        {
            'text': trans._('Open .LSGD file(s) ...'),
            'slot': self._open_lsgd_file,
            'shortcut': 'Ctrl+Alt+O+L',
        },
        {
            'text': trans._('Open .ZARR file ...'),
            'slot': self._open_zarr_file,
            'shortcut': 'Ctrl+Alt+O+Z',
        },   
        {
            'text': trans._('Open DAPI Files ...'),
            'slot': self._open_dapi_file,
            'shortcut': 'Ctrl+Alt+O+D',
        },
        {
            'text': trans._('Open folder with Pyramid(s) ...'),
            'slot': self._open_pyramids_folder,
            'shortcut': 'Ctrl+Alt+O+P',
        },
        ]   
        populate_menu(self, ACTIONS)


    def _open_lsgd_file(self):
        """Add .lsgd file(s) from the menubar."""
        dlg = QFileDialog()
        hist = get_open_history()
        dlg.setHistory(hist)
        dlg.setNameFilter("Raw Data (*.lsgd *.lout)")
        dlg.setDirectory(hist[0])

        filenames, _ = dlg.getOpenFileNames(
            parent=self,
            caption=trans._('Select lsgd file(s)...'),
            directory=hist[0],
            filter=("Raw Data (*.lsgd *.lout)"),
            options=(
                QFileDialog.DontUseNativeDialog
                if in_ipython()
                else QFileDialog.Options()
            ),
        )

        if (filenames != []) and (filenames is not None):
            for filename in filenames:
                if (filename != '') and (filename is not None):
                    file_extension = (os.path.basename(filename)).split(".")[-1]
                    if file_extension in RAW_DATA_FORMATS:                
                        imgstack = read_lsgd_as_imagestack(filename, image_ysize=IMG_SIZE_Y) 
                        self.viewer.add_image(imgstack, name=os.path.basename(filename))
                        update_open_history(Path(filename))
        
    
    def _open_zarr_file(self):
        """Add Pyramid .zarr files from the menubar."""
        dlg = QFileDialog()
        hist = get_open_history()
        dlg.setHistory(hist)
        dlg.setDirectory(hist[0])

        filename = dlg.getExistingDirectory(
            parent=self,
            caption=trans._('Select zarr file(s)...'),
            directory=hist[0],
            options=(QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks),
        )

        if (filename != '') and (filename is not None):
            file_name_no_extension, file_extension = (os.path.basename(filename)).split(".")
            file_name_no_extension = file_name_no_extension.replace('Pyramid_S_', '') if 'Pyramid_S_' in file_name_no_extension else file_name_no_extension
            if file_extension in PYRAMID_DATA_FORMATS:                 
                pyramid_s_img = [da.from_zarr(filename + "/" + str(i)) for i in range(Num_pyramid_layers)]
                self.viewer.add_image(
                    data=pyramid_s_img,
                    contrast_limits=[0, MAX_CONTRAST_LIMIT],
                    colormap="gray",
                    multiscale=True,
                    visible=True, 
                    blending='additive',
                    name=file_name_no_extension,
                    )
                update_open_history(Path(filename))
    
    
    def _open_pyramids_folder(self):
        """Add all Pyramid .zarr files in the folder from the menubar."""
        dlg = QFileDialog()
        hist = get_open_history()
        dlg.setHistory(hist)
        dlg.setDirectory(hist[0])

        experiment_path = dlg.getExistingDirectory(
            parent=self,
            caption=trans._('Select the folder containing all zarr files...'),
            directory=hist[0],
            options=(QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks),
        )
        
        all_pyramids = glob(os.path.join(experiment_path, "Pyramid_S_*"))
        all_pyramid_nuclei = glob(os.path.join(experiment_path, "Pyramid_DAPI_*"))
        if len(all_pyramid_nuclei)>0:
            all_pyramids += all_pyramid_nuclei

        for filename in all_pyramids:
            zarr_img_filename = os.path.splitext(os.path.basename(filename))[0]
        
            print(zarr_img_filename, filename)
            if "Pyramid_S_" in zarr_img_filename:
                img_name = zarr_img_filename.replace("Pyramid_S_",'')
            elif "Pyramid_DAPI_" in zarr_img_filename:
                img_name = zarr_img_filename.replace("Pyramid_",'')
            else:
                img_name = zarr_img_filename
            
            if (filename != '') and (filename is not None):
                file_name_no_extension, file_extension = (os.path.basename(filename)).split(".")
                file_name_no_extension = \
                    file_name_no_extension.replace('Pyramid_S_', '') \
                        if 'Pyramid_S_' in file_name_no_extension \
                            else file_name_no_extension
                
                if file_extension in PYRAMID_DATA_FORMATS:  
                    try:
                        pyramid_s_img = [da.from_zarr(filename + "/" + str(i)) for i in range(Num_pyramid_layers)]
                        self.viewer.add_image(
                            data=pyramid_s_img,
                            contrast_limits=[0, MAX_CONTRAST_LIMIT],
                            colormap="gray",
                            multiscale=True,
                            visible=True, 
                            blending='translucent',
                            name=img_name,
                            )
                        update_open_history(Path(filename))
                    except Exception as e:
                        print(e)
                    
    def _open_dapi_file(self):
        """Add DAPI file from the menubar."""
        dlg = QFileDialog()
        hist = get_open_history()
        dlg.setHistory(hist)
        dlg.setNameFilter("Raw Data (*.dout *.dapi *.tif *.tiff)")
        dlg.setDirectory(hist[0])

        filenames, _ = dlg.getOpenFileNames(
            parent=self,
            caption=trans._('Select DAPI file(s)...'),
            directory=hist[0],
            filter=("Raw Data (*.dout *.dapi *.tif *.tiff)"),
            options=(
                QFileDialog.DontUseNativeDialog
                if in_ipython()
                else QFileDialog.Options()
            ),
        )

        if (filenames != []) and (filenames is not None):
            for filename in filenames:
                if (filename != '') and (filename is not None):
                    
                    file_extension = (os.path.basename(filename)).split(".")[-1]
                    filename_no_path = os.path.basename(filename)
                    filename_no_extension = filename_no_path.replace("."+file_extension, '')
                    dapi_gene_cut_reshape = None            
                    try:
                        if file_extension == "dout" or file_extension == "dapi":
                            with open(filename, "rb") as fid:
                                dapi_gene = (fromfile(fid, uint16))                    
                                offset = len(dapi_gene) - (IMG_SIZE_Y*IMG_SIZE_Y)
                                dapi_gene_cut = dapi_gene[offset:]
                                dapi_gene_cut_reshape = dapi_gene_cut.reshape(IMG_SIZE_Y, IMG_SIZE_Y)
                        
                        elif file_extension == "tif":                    
                            dapi_gene_cut = imread(filename)                    
                            dapi_gene_cut_reshape = dapi_gene_cut.reshape(IMG_SIZE_Y, IMG_SIZE_Y)
                        
                        else:
                            dapi_gene_cut_reshape = None 
                        
                        self.viewer.add_image(dapi_gene_cut_reshape, name=filename_no_extension)
                        update_open_history(Path(filename))
                        
                    except Exception as ex:
                        print(ex)
                        try:
                            dapi_gene_cut = volread(filename, format="tiff")
                            dapi_gene_cut_reshape = dapi_gene_cut.reshape(IMG_SIZE_Y, IMG_SIZE_Y)
                        except Exception as ex:
                            print(ex)


class List_Existing_Files(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        hlay = QHBoxLayout(self)
        
        # self.treeview = QTreeView()
        # self.listview = QListView()
        # hlay.addWidget(self.treeview)
        # hlay.addWidget(self.listview)
        
        file_dlg = self.dlg = QFileDialog()
        file_dlg.setFileMode(QFileDialog.Directory)
        hlay.addWidget(self.dlg)
    
    def load_experiment_files(self):
        hist = get_open_history()
        self.dlg.setHistory(hist)
        self.dlg.setDirectory(hist[0])

        selected_directory = self.dlg.getExistingDirectory(
            parent=self,
            caption=trans._('Select lsgd/lout file(s)...'),
            directory=hist[0],
            options=(QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks),
        )
        
        all_raw_files = glob(os.path.join(selected_directory, "*.lsgd"))

        if (all_raw_files != '') and (all_raw_files is not None):
            file_name_no_extension, file_extension = (os.path.basename(selected_directory)).split(".")
            print(file_name_no_extension)


@magicgui(
    auto_call=False, 
    main_window=False,
    Folder_Path={'mode': 'd'}, 
    call_button="Load Experiment Path", 
    result_widget=False,
    )
def load_experiment_path(
    viewer: Viewer,
    Folder_Path=Path(),   
    ):
    
    """ Load a path with existing captured data """
    
    print(Folder_Path)
    
    # dlg = QFileDialog()
    # hist = get_open_history()
    # dlg.setHistory(hist)
    # dlg.setDirectory(hist[0])
    
    # Folder_Path_Default = None 

    # experiment_path = dlg.getExistingDirectory(
    #     parent=Folder_Path_Default,
    #     caption=trans._('Select the folder containing all zarr files...'),
    #     directory=hist[0],
    #     options=(QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks),
    #     )
        
    # all_raw_files = glob(os.path.join(experiment_path, "*.lsgd"))
    
    # return Folder_Path
    
    
@magicgui(auto_call=False, main_window=False, result_widget=True, \
    call_button="Experiment Info", \
        layout='vertical', \
            # experiment_name={'bind': experiment_name}, \
            #     experiment_full_path={'bind': experiment_full_path} \
                    )
def show_experiment_name(experiment_name, experiment_full_path):
    print(experiment_name)
    print(experiment_full_path)
    return (f"Current experiment: {experiment_name}")


@magicgui
def filepicker(
    filename : Annotated[Path, {'mode': 'r', 'filter': 'Images (*.lsgd *.lout)'}]
):
    print("Testing filepicker")


def show_files_in_dir(Folder_Path=Path()):
    # files = sorted(glob(str(Folder_Path) + "*.lsgd"))
    files = sorted(glob(str(Folder_Path) + "*.*"))
    names = [os.path.basename(f) for f in files]
    print(names)
    # for n in names:
    #     list_widget.addItem(n)
    
    print("... show_files_in_dir ... ")
    
    return names
    

class show_all_files:
    
    def __init__(self, current_folder):
        print("Initializing show_all_files")
        self.current_folder = current_folder
        self.list_widget = QListWidget()
        
        all_files_in_folder = show_files_in_dir(self.current_folder)
        self.list_widget.addItems(all_files_in_folder)
    
    def refresh_content(self, updated_folder):
        print("Updating show_all_files")
        all_files_in_folder = show_files_in_dir(updated_folder)
        self.list_widget.addItems(all_files_in_folder)
        

if __name__ == '__main__':
    
    print("Testing Rebus Napari Menues")
    viewer = Viewer()
      
    rebus_menu = RebusMenu(viewer, viewer.window)
    viewer.window.main_menu.addMenu(rebus_menu)
    
    # experiment_path_widget = load_experiment_path
    # experiment_path_widget_viewer = viewer.window.add_dock_widget(experiment_path_widget, name='Load Experiment Path', area='right') 
    
    # list_widget = QListWidget()
    # for n in names:
    #     list_widget.addItem(n)

    # list_widget.currentItemChanged.connect(experiment_path_widget_viewer)
    
    # show_files_widgets = List_Existing_Files()
    # show_files_widgets_viewer = viewer.window.add_dock_widget(show_files_widgets, name='List Files', area='right') 
    
    # list_widget = QListWidget()
    # viewer.window.add_dock_widget(list_widget, name='List Files', area='right') 
    
    # show_files_in_dir_widget = show_files_in_dir(list_widget, Path(str(experiment_path_widget.Folder_Path.value.root)))    
    # # list_widget.addItems(show_files_in_dir_widget)
    
    # experiment_path_widget.changed.connect(show_files_in_dir_widget)
    # list_widget.currentItemChanged.connect(experiment_path_widget)
    # experiment_path_widget.call_button.clicked.connect(show_files_in_dir_widget)
    
    # show_files_in_dir.changed.connect(experiment_path_widget)
    
    # show_files_widg = show_all_files(current_folder=Path(str(experiment_path_widget.Folder_Path.value)))
    # viewer.window.add_dock_widget(show_files_widg.list_widget, name='List Files', area='right') 
    # experiment_path_widget.call_button.clicked.connect(show_files_widg.refresh_content(show_files_widg.current_folder))
    
    FolderBrowser_widget = FolderBrowser(viewer)
    viewer.window.add_dock_widget(FolderBrowser_widget)

    run_napari()