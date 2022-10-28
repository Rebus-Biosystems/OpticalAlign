import sys
import os 

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app import main
from esper_explore_widgets import napari_file_explore_widget
from SVBeamMeasure import ArgoDataLib, GaussFitLib, svbeam

if __name__ == '__main__':
    main()
