
import napari 
import os 
import sys 

THIS_SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(THIS_SCRIPT_DIR, "esper_explore_widgets"))
sys.path.insert(0, os.path.join(THIS_SCRIPT_DIR, "SVBeamMeasure"))
sys.path.insert(0, THIS_SCRIPT_DIR)

import esper_explore_widgets
import esper_explore_widgets.svbeam
import esper_explore_widgets.napari_file_explore_widget

import SVBeamMeasure.ArgoDataLib 
import SVBeamMeasure.GaussFitLib 

import esper_explore_widgets.fov_utils
from esper_explore_widgets.napari_file_explore_widget import FolderBrowser
from esper_explore_widgets.svbeam import SVBeamMeasureUI 

  
def main():
    
    viewer = napari.Viewer()
            
    FolderBrowser_widget = FolderBrowser(viewer)
    viewer.window.add_dock_widget(FolderBrowser_widget, name='Load Experiment Files', area='right', menu=viewer.window.window_menu)    
    viewer.window.add_dock_widget(SVBeamMeasureUI, name='SVBeamMeasure', area='right', menu=viewer.window.window_menu)    
        
    napari.run()


if __name__ == '__main__':
    main()
