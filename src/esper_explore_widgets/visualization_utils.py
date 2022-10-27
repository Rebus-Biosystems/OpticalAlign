import argparse
import glob
import json
import os
import pickle
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

import dask.array as da
import napari
import numpy as np
import pandas as pd
import zarr
from cv2 import COLOR_BGR2GRAY, cvtColor, imread
from magicgui import magic_factory, magicgui, widgets
from magicgui.widgets import LineEdit
from napari import Viewer, gui_qt
from napari import run as run_napari
from napari.layers.points.points import Points as Napari_points
from napari.types import ImageData, LayerDataTuple, PointsData
from napari.utils.events import Event
from napari.utils.io import imsave
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtGui import QCursor, QMouseEvent, QTextCursor, QTextDocument
from qtpy.QtWidgets import (QApplication, QColorDialog, QFileDialog,
                            QPlainTextEdit, QTextEdit, QToolTip, QVBoxLayout,
                            QWidget)
from tqdm import tqdm

THIS_SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(THIS_SCRIPT_DIR, "../common"))
sys.path.insert(0, os.path.join(THIS_SCRIPT_DIR, "../"))
sys.path.insert(0, os.path.join(THIS_SCRIPT_DIR, "../../"))

import profile_plotter_napari
import rebus_widgets
from napari._qt.menus._util import NapariMenu, populate_menu
from napari._qt.menus.file_menu import FileMenu
from napari._qt.qt_main_window import Window, _QtMainWindow
from napari.utils.translations import trans
from rebus_napari_menu import RebusFileMenu, RebusMenu

from common import experiment_utils
from esper_process.cv_algorithms.fov_registration import load_obj_json
from esper_process.cv_algorithms.generate_images import (
    generate_vertices, get_fov_names, get_shifts_from_registration,
    save_fov_vertices, unique_values)

table_dtypes = {'CBR':float, 'global_RNA_x':float, 'global_RNA_y':float}

# PARAMETERS
multiprocess = False      
SHOW_FOV_GRID = False  
LOAD_NUCLEI_NOT_PYRAMID = False          
image_size_default = 2048
REF_CYCLE = 1
DEFAULT_LEFT_RIGHT_SHIFT=1234
ZOOM_FOV = 0.6 

MIN_CRA_COL_VALUE = 1.5
MAX_CRA_COL_VALUE = 10
MIN_BACKGROUND = 300
MAX_BACKGROUND = 20000
MIN_INTENSITY = 300
MAX_INTENSITY = 30000
MIN_INTENSITY_DIFF = 300
MAX_INTENSITY_DIFF = 30000
MIN_DISTANCE_TO_CELL = -1
MAX_DISTANCE_TO_CELL = 100

Widget_type = 'FloatSpinBox' #'FloatSlider'

SHOW_NUCLEI_LOCATIONS = False 
SHOW_NUCLEI = True    
SKIP_NUCLEI = not SHOW_NUCLEI
DISK_SIZE = 1
DISK_SIZE_150 = 3
DISK_SIZE_141 = 3 
DISK_SIZE_NUCLEI = 4
SCALE_141 = 4
IMAGFE_SIZE_DEFAULT = 4928
DEFAULT_VISIBLE_DOTS = False  
THREAD_COUNT = 8

SHOW_141 = False               
SHOW_SIDE_BY_SIDE = False      
SHIFT_X_BIAS = 0
SHIFT_Y_BIAS = 0          
    
max_data_Size = 25000 
VISUALIZE_FEATURE_COLS = ['global_RNA_x', 'global_RNA_y', 'CBR', 'x', 'y', 'Background', 'Intensity', 'distance_to_cell']
NAN_VALUE = float("NaN")
REMOVE_GENES_NOT_INTEREST = ["None", "none"]
unpopulated_gene_channel = ['L473r']

CUT_OVERLAP = False     
Recon_size_default = 4928 

Num_pyramid_layers = 16
MAX_CONTRAST_LIMIT = 2500


class napari_camera: #(EventedModel):
    """Camera object modeling position and view of the camera.

    Attributes
    ----------
    center : 3-tuple
        Center of rotation for the camera.
        In 2D viewing the last two values are used.
    zoom : float
        Scale from canvas pixels to world pixels.
    angles : 3-tuple
        Euler angles of camera in 3D viewing (rx, ry, rz), in degrees.
        Only used during 3D viewing.
        Note that Euler angles's intrinsic degeneracy means different
        sets of Euler angles may lead to the same view.
    perspective : float
        Perspective (aka "field of view" in vispy) of the camera (if 3D).
    interactive : bool
        If the camera interactivity is enabled or not.
    """

    # fields
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    zoom: float = 1.0
    angles: Tuple[float, float, float] = (0.0, 0.0, 90.0)
    perspective: float = 0
    # interactive: bool = True

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_cutting_coorids(Recon_size = Recon_size_default):    
    pre_recon_size = int(Recon_size/4)
    x_start = int(pre_recon_size/2)-1
    x_end = int(pre_recon_size/2) + pre_recon_size
    y_start, y_end = x_start, x_end
    return x_start, x_end, y_start, y_end


def get_stitch_columns(table_files):
    data_cols = pd.read_csv(table_files, nrows=0).columns
    use_cols = list(set(data_cols.to_list()) & set(VISUALIZE_FEATURE_COLS))
    if 'global_RNA_x' in data_cols:
        col_x = 'global_RNA_x'
        col_y = 'global_RNA_y'
    else:
        col_x = 'x'
        col_y = 'y'

    return use_cols, col_x, col_y

class FOV_Vertices:
    def __init__(self, fov_names, vertices):
        self.fov_names = fov_names
        self.vertices = vertices


def load_obj(filename):
    file_handle = open(filename, 'rb')
    fov_handle = pickle.load(file_handle)
    file_handle.close()
    return fov_handle


def _update_thumbnail_color(points_layer):
        """Update thumbnail with current points and colors."""
        colormapped = np.zeros(points_layer._thumbnail_shape)
        colormapped[..., 3] = 1
        view_data = points_layer._view_data
        if len(view_data) > 0:
            # Get the zoom factor required to fit all data in the thumbnail.
            de = points_layer._extent_data
            min_vals = [de[0, i] for i in points_layer._dims_displayed]
            shape = np.ceil(
                [de[1, i] - de[0, i] + 1 for i in points_layer._dims_displayed]
            ).astype(int)
            zoom_factor = np.divide(
                points_layer._thumbnail_shape[:2], shape[-2:]
            ).min()

            # Maybe subsample the points.
            if len(view_data) > points_layer._max_points_thumbnail:
                thumbnail_indices = np.random.randint(
                    0, len(view_data), points_layer._max_points_thumbnail
                )
                points = view_data[thumbnail_indices]
            else:
                points = view_data
                thumbnail_indices = points_layer._indices_view

            # Calculate the point coordinates in the thumbnail data space.
            thumbnail_shape = np.clip(
                np.ceil(zoom_factor * np.array(shape[:2])).astype(int),
                1,  # smallest side should be 1 pixel wide
                points_layer._thumbnail_shape[:2],
            )
            coords = np.floor(
                (points[:, -2:] - min_vals[-2:] + 0.5) * zoom_factor
            ).astype(int)
            coords = np.clip(coords, 0, thumbnail_shape - 1)

            # Draw single pixel points in the colormapped thumbnail.
            colormapped = np.zeros(tuple(thumbnail_shape) + (4,))
            colormapped[..., 3] = 1
            colors = points_layer._face.colors[thumbnail_indices]
            colormapped[coords[:, 0], coords[:, 1]] = colors

        colormapped[..., 3] *= points_layer.opacity
        
        return colormapped



# class RebusFileMenu(FileMenu):
#     def __init__(self, window: 'Window'):
#         self._win = window
#         super().__init__(window)
#         ACTIONS = [
#         {
#             'text': trans._('Open File(s)...'),
#             'slot': window._qt_viewer._open_files_dialog,
#             'shortcut': 'Ctrl+O',
#         },
#         ]   
#         populate_menu(self, ACTIONS)

   
def visualize_table(experiment_folders, gene_list, args):

    """ Visualize the RNA dots, stitched pyramids and DAPI using Napari """

    viewer = Viewer()
      
    rebus_menu = RebusMenu(viewer, viewer.window)
    viewer.window.main_menu.addMenu(rebus_menu)
    
    nuclei_stitched = experiment_folders["stitching_save_folder"]     
    nuclei_stitched_filepath = os.path.join(nuclei_stitched, "nuclei_stitched_Cycle_1.tif")
    Segmented_nuclei_stitched_filepath = os.path.join(nuclei_stitched, "Segmented_nuclei_stitched_Cycle_1.tif")
    bbox_nuclei_stitched = os.path.join(nuclei_stitched, "nuclei_stitched_bbox_cycle_1.tif")
    zarr_nuclei_stitched_filepath = os.path.join(nuclei_stitched, "nuclei_stitched_Cycle_1.zarr")

    all_pyramid_s_images = glob.glob(os.path.join(experiment_folders["Pyramid_S_Images"], "Pyramid_S_*")) 
    all_pyramid_dapi_images = glob.glob(os.path.join(experiment_folders["Pyramid_DAPI"], "Pyramid_DAPI_*")) 

    cellxgene_folder = os.path.join(experiment_folders['CellxGene'], 'process_params.csv')
    default_points_color_save = os.path.join(experiment_folders['CellxGene'], 'points_color_save.csv')

    esperiment = experiment_utils.EsperExperiment.load_or_create(args.rootdir)
    experiment_name = os.path.basename(args.rootdir)
    experiment_full_path = Path(args.rootdir)

    if len(all_pyramid_dapi_images)>0:
        all_pyramid_s_images += all_pyramid_dapi_images

    try:
        vertices_loaded = False
        
        vertices_filename = os.path.join(experiment_folders["Pyramid_metadata"], 'vertices_metadata.obj')    
        if os.path.isfile(vertices_filename):    
            vertices_obj = load_obj(vertices_filename)
            fov_names = vertices_obj.fov_names
            vertices = vertices_obj.vertices
        else:
            print(f"No such file exist {vertices_filename} ... Generating Vertices")
            fov_names = get_fov_names(esperiment)
            vertices = generate_vertices_onthefly(experiment_folders, fov_names)
            _ = save_fov_vertices(FOV_Vertices, experiment_folders, fov_names, vertices)
        
        initial_text_color = "yellow"
        initial_text_size = 8
        initial_edge_width = 5
        properties = {"fov_name": fov_names}
        text_params = {"text": "{fov_name}", "size": initial_text_size, "color": initial_text_color, "anchor": "center", "translation": [0, 0]}
        vertices_loaded = True 
    
    except Exception as e:
        print(f"Exception {e}")

    if os.path.isfile(bbox_nuclei_stitched):
        bbox_nuclei_stitched_img = imread(bbox_nuclei_stitched)
        viewer.add_image(bbox_nuclei_stitched_img, name="bbox_nuclei_stitched", scale=(4,4), opacity=0.5, visible=False)

    if not args.SKIP_NUCLEI:
    
        if not os.path.isfile(nuclei_stitched_filepath):
            nuclei_stitched_filepath = os.path.join(nuclei_stitched, "nuclei_stitched_coarse_cycle_1.tif")

        if os.path.isfile(nuclei_stitched_filepath):
            nuclei_stitched = imread(nuclei_stitched_filepath)
            if len(nuclei_stitched.shape)>1:
                nuclei_stitched = cvtColor(nuclei_stitched, COLOR_BGR2GRAY)

            viewer.add_image(nuclei_stitched, name="nuclei_stitched_low_res", \
                scale=(4,4), colormap="gray", blending="additive", opacity=0.5, visible=False, \
                    )
        
        if os.path.isfile(Segmented_nuclei_stitched_filepath):
            segmented_nuclei_stitched = imread(Segmented_nuclei_stitched_filepath)
            if len(segmented_nuclei_stitched.shape)>1:
                segmented_nuclei_stitched = cvtColor(segmented_nuclei_stitched, COLOR_BGR2GRAY)

            viewer.add_image(segmented_nuclei_stitched, name="segmented_nuclei_stitched_low_res", \
                scale=(4,4), colormap="gray", blending="additive", opacity=0.5, visible=True, \
                    )
        
        if os.path.isdir(zarr_nuclei_stitched_filepath) and LOAD_NUCLEI_NOT_PYRAMID:
            nuclei_stitched_zarr = zarr.convenience.load(zarr_nuclei_stitched_filepath)
            
            zarr_data_shap = nuclei_stitched_zarr.shape 
            y_start, y_end = 0, zarr_data_shap[1]
            x_start, x_end = 0, zarr_data_shap[0]
            if max(zarr_data_shap)>max_data_Size:
                # Divide by 4 slices 
                x_middle, y_middle = int(zarr_data_shap[0]/2), int(zarr_data_shap[1]/2)
                x_end, y_end = int(zarr_data_shap[0]), int(zarr_data_shap[1])

                for i in range(2):
                    for j in range(2):
                        viewer.add_image(
                            nuclei_stitched_zarr[i*x_middle:(i+1)*x_middle, j*y_middle:(j+1)*y_middle],
                            name="nuclei_stitched_full_res",
                            colormap="gray",
                            blending="additive",
                            translate=(i*x_middle, j*y_middle),
                            contrast_limits=(0,25000),
                        )

            else:
                viewer.add_image(
                    nuclei_stitched_zarr,
                    name="nuclei_stitched_full_res",
                    colormap="gray",
                    blending="additive",
                )  

    if args.multiprocess:
        points_collection = []
        args_tuples = [\
        (experiment_folders, target, args)\
            for target in gene_list\
                ]
        
        with Pool(args.threadcount) as p:
            with tqdm(total=len(gene_list)) as pbar:
                for results in p.imap_unordered(add_target_points, args_tuples):
                    pbar.update()
                    points_collection.append(results)

        # (target, coords_selected, coords_selected_df, results_filtered_df)
        for point in points_collection:
            this_target = point[0]
            target_points = point[2]
            results_filtered_df = point[3]
            data_Shape = target_points.shape
            if data_Shape[0]>2: 
                print(this_target)
                rgb = get_rgb_random()
                layer = viewer.add_points(
                                    target_points,
                                    features=results_filtered_df,
                                    opacity=0.95,
                                    face_color=rgb,
                                    edge_color=rgb,
                                    symbol="disc",
                                    size=DISK_SIZE,
                                    name=this_target,
                                    visible = DEFAULT_VISIBLE_DOTS
                                    )
                layer.name = this_target 
            else:
                print(f"Target points for {this_target} is empty dataframe, not displaying on Napari")

    else:

        for target in gene_list[0:]:
            print(target)
            
            target_filename = os.path.join(experiment_folders["GeneTables_folder"], target + ".csv")

            try:

                if os.path.isfile(target_filename):
                    cols_to_load, col_x, col_y = get_stitch_columns(target_filename)
                    results_filtered_df = pd.read_csv(target_filename, low_memory=True, header=0)
                    # results_filtered_df.dropna(axis=0, how='any', thresh=1, subset=None, inplace=True)
                    results_filtered_df.dropna(axis=0, how='any', inplace=True)
                    coords_selected_df, results_selected_df = filter_data_features(col_x, col_y, results_filtered_df)

                    rgb = get_rgb_random()
                    layer = viewer.add_points(
                        coords_selected_df,
                        features=results_selected_df,
                        opacity=0.95,
                        face_color=rgb,
                        edge_color=rgb,
                        symbol="disc",
                        size=DISK_SIZE,
                        name=target,
                        visible = DEFAULT_VISIBLE_DOTS
                                )
                    layer.name = target 
                        
                else:
                    print(f"{target} file does not exist in the {target_filename}" )
            
            except Exception as e:
                print(f"Exception {e}")
                print("Oops!", e.__class__, "occurred.")

    for filename in all_pyramid_s_images:
        zarr_img_filename = os.path.splitext(os.path.basename(filename))[0]
        
        print(zarr_img_filename, filename)
        if "Pyramid_S_" in zarr_img_filename:
            img_name = zarr_img_filename.replace("Pyramid_S_",'')
        elif "Pyramid_DAPI_" in zarr_img_filename:
            img_name = zarr_img_filename.replace("Pyramid_",'')
        else:
            img_name = zarr_img_filename
        
        try:
            pyramid_s_img = [da.from_zarr(filename + "/" + str(i)) for i in range(Num_pyramid_layers)]
            viewer.add_image(
                data=pyramid_s_img,
                contrast_limits=[0, MAX_CONTRAST_LIMIT],
                colormap="gray",
                multiscale=True,
                visible=False, 
                blending='additive',
                name=img_name,
                )
        except Exception as e:
            print(f"Exception {e}")
            print("Oops!", e.__class__, "occurred.")
            print(f"Could not load pyramid file {zarr_img_filename}")
    
    try:
        if vertices_loaded:
            shape_layer = viewer.add_shapes(
                    vertices,
                    name="FOVs",
                    shape_type="polygon",
                    edge_width=initial_edge_width,
                    edge_color="yellow",
                    face_color="transparent",
                    properties=properties,
                    text=text_params,
                    )
            container_vertices, _ = rebus_widgets.vertices_ui_widgets(viewer, initial_text_size, initial_edge_width, shape_layer)

    except Exception as e:
        print(e)
        print("Did not load FOV vertices due to Exception")

    Rebus_widget_Napari = rebus_widgets.Rebus_Widgets(viewer, experiment_folders)
    feature_panel = rebus_widgets.feature_panel
    change_points_color = rebus_widgets.change_points_color
    change_points_size = rebus_widgets.change_points_size 
    save_spot_colors = rebus_widgets.save_spot_colors
    load_spot_colors = rebus_widgets.load_spot_colors
    save_params_panel = rebus_widgets.save_params
    save_params_panel.Folder_Path.value = Path(cellxgene_folder)
    save_spot_colors.Folder_Path.value = Path(default_points_color_save)
    load_spot_colors.Folder_Path.value = Path(default_points_color_save)

    @magicgui(auto_call=False, main_window=False, fov={"choices": fov_names}, call_button="Move to FOV")
    def move_to_fov_region(viewer: Viewer, layer: napari.layers.Shapes, fov):
        if 'FOVs' in layer.name:
            fov_cooords = layer.data
            fov_list = list(layer.text.values)
            if fov in fov_list:
                fov_name_index = fov_list.index(fov)
                this_fov_coords = fov_cooords[fov_name_index]
                viewer.camera.center = (0, \
                    0.5*(this_fov_coords[1][0] + this_fov_coords[0][0]), \
                        0.5*(this_fov_coords[3][1] + this_fov_coords[0][1]) \
                            )
                viewer.camera.zoom = ZOOM_FOV
    
    save_selected_region = rebus_widgets.save_selected_region
    save_selected_region.experiment_folders.value = experiment_folders

    @magicgui(auto_call=False, main_window=False, call_button="Create FOVs", \
        experiment_folders={'bind': experiment_folders}, fov_names={'bind': fov_names})
    def load_or_create_fovs(viewer: Viewer, experiment_folders, fov_names, create_new=True):
        if create_new: 
            vertices = generate_vertices_onthefly(experiment_folders, fov_names)
            initial_text_color = "red"
            initial_text_size = 8
            initial_edge_width = 5
            properties = {"fov_name": fov_names}
            text_params = {"text": "{fov_name}", "size": initial_text_size, "color": initial_text_color, "anchor": "center", "translation": [0, 0]}
            shape_layer_reload = viewer.add_shapes(
                vertices,
                shape_type="polygon",
                edge_width=initial_edge_width,
                edge_color="red",
                face_color="transparent",
                properties=properties,
                text=text_params,
                name='All_FOVs_Reload',
            )
        
        return True 

    exp_name_widget = LineEdit(value=experiment_name)
    exp_path_eidget = LineEdit(value=experiment_full_path)
    exp_name_widget.max_width = 600 
    exp_path_eidget.max_width = 600 

    container_experiment = widgets.Container(widgets=[exp_name_widget, exp_path_eidget])
    show_experiment_name_widget = \
        viewer.window.add_dock_widget(\
            container_experiment, \
                name='Experiment Info', \
                    area='left', \
                        menu=viewer.window.window_menu \
        )        
    
    @magicgui(auto_call=True, main_window=False, call_button="Show_on_Mouse")
    def mouseover_status_changed(viewer: napari.Viewer):
        """Update status bar.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        # print(viewer.layers.selection.active.position)
        try:
            current_mouse_roll = viewer.status
            layer_name = current_mouse_roll.split(" ")[0]
            layer_value = int(current_mouse_roll.split(",")[-1])
            print(viewer.status)
            print(viewer.cursor.position)
            print(layer_name, layer_value)
        except Exception as e:
            print(e)

        return None 
           

    @viewer.mouse_move_callbacks.append 
    def get_pdisplay(viewer, event):
        if 'Control' in event.modifiers:
            
            try:
                current_mouse_roll = viewer.status
                layer_name = current_mouse_roll.split(" ")[0]
                layer_value = int(current_mouse_roll.split(",")[-1])
                viewer.text_overlay.text = layer_name + "  " + str(layer_value)
                viewer.text_overlay.visible  = True 
                viewer.scale_bar.ticks  = True 
                viewer.scale_bar.visible  = True 
                viewer.tooltip.text = str(layer_value)
                
                mouse_x, mouse_y = rebus_widgets._get_mouse_coords_in_screen(viewer)
                textEdit = QTextEdit()
                cursor1 = QCursor()
                pos3 = cursor1.pos()
                cursor1 = QTextCursor(textEdit.document())
                cursor1.setPosition(0)
                QToolTip.showText(pos3, str(layer_value))
                
            except Exception as e:
                print(e)
                
        else:
            viewer.text_overlay.visible  = False 

    
    save_camera_viewer = rebus_widgets.save_camera_viewer
    load_camera_view = rebus_widgets.load_camera_view
    display_layer_stuff = rebus_widgets.display_layer_stuff
    save_screenshot = rebus_widgets.save_screenshot
    image_dynamic_range_ui_widgets, _ = rebus_widgets.image_dynamic_range_ui_widgets(viewer, MAX_CONTRAST_LIMIT=MAX_CONTRAST_LIMIT)
    container_points = widgets.Container(widgets=[feature_panel, save_params_panel, change_points_color, change_points_size, save_spot_colors, load_spot_colors])
    
    if vertices_loaded:
        container_region_and_camera = \
            widgets.Container(widgets=[\
                container_vertices, load_or_create_fovs, move_to_fov_region, \
                    save_selected_region, save_camera_viewer, load_camera_view, save_screenshot\
                        ]\
                            )
    else:
        container_region_and_camera = widgets.Container(widgets=[save_selected_region, save_camera_viewer, load_camera_view, save_screenshot])

    plot_profiler_widget = profile_plotter_napari.PlotProfile(viewer)
    
    viewer.layers.events.inserted.connect(feature_panel.reset_choices)
    viewer.layers.events.changed.connect(feature_panel.reset_choices)  
    viewer.layers.events.changed.connect(feature_panel._on_change)  
    viewer.layers.events.removed.connect(feature_panel.reset_choices)  
    
    viewer.layers.selection.active.events.connect(change_points_color.reset_choices)
    viewer.layers.selection.active.events.connect(change_points_color.reset_call_count)
    viewer.layers.events.changed.connect(change_points_color.reset_choices)  
    
    viewer.layers.selection.active.events.connect(change_points_size.reset_choices)
    viewer.layers.selection.active.events.connect(change_points_size.reset_call_count)
    viewer.layers.events.changed.connect(change_points_size.reset_choices)  
    
    viewer.layers.selection.active.events.connect(save_spot_colors.reset_choices)
    viewer.layers.selection.active.events.connect(save_spot_colors.reset_call_count)
    viewer.layers.events.changed.connect(save_spot_colors.reset_choices)  
    
    viewer.layers.selection.active.events.connect(load_spot_colors.reset_choices)
    viewer.layers.selection.active.events.connect(load_spot_colors.reset_call_count)
    viewer.layers.events.changed.connect(load_spot_colors.reset_choices)  

    viewer.layers.selection.active.events.connect(display_layer_stuff.reset_choices)
    viewer.layers.selection.active.events.connect(display_layer_stuff.reset_call_count)
    viewer.layers.selection.active.events.connect(display_layer_stuff.layer._emit_parent)
    viewer.layers.selection.active.events.connect(display_layer_stuff.parent_changed)

    @viewer.layers.selection.events.changed.connect
    def update_combox_selection(event):
        # display_layer_stuff.reset_choices()
        save_spot_colors.reset_choices()
        load_spot_colors.reset_choices()
        change_points_size.reset_choices()
        change_points_color.reset_choices()
        feature_panel.reset_choices()
        current_selected = viewer.layers.selection._current
        viewer.layers.selection._update_active()        
        if current_selected:
            if current_selected._type_string=='points':
                change_points_color.layer.value = current_selected
                change_points_size.layer.value = current_selected
                feature_panel.layer.value = current_selected

    dw_points = viewer.window.add_dock_widget(container_points, name='Spot Maps', menu=viewer.window.window_menu)
    
    dw_regions = viewer.window.add_dock_widget(container_region_and_camera, name='ROIs', menu=viewer.window.window_menu)    
    viewer.window._qt_window.tabifyDockWidget(dw_points, dw_regions) 

    dw_plotp_tofile = viewer.window.add_dock_widget(plot_profiler_widget, area='right', name='Plot Profile', menu=viewer.window.window_menu)
    viewer.window._qt_window.tabifyDockWidget(dw_regions, dw_plotp_tofile)

    dw_dynamic_range = viewer.window.add_dock_widget(image_dynamic_range_ui_widgets, area='right', name='Image Layers', menu=viewer.window.window_menu)
    viewer.window._qt_window.tabifyDockWidget(dw_plotp_tofile, dw_dynamic_range) 

    run_napari()

    end_processing_time = time.time() 

    return end_processing_time 


def generate_vertices_onthefly(experiment_folders, fov_names):
    reg_file = os.path.join(experiment_folders["registration_dir"], "global_transform_all.json")
    if os.path.isfile(reg_file):
        read_success, global_transform_all = load_obj_json(reg_file)
    # left_right_shift, top_bot_shift = get_shifts_from_registration(experiment_folders, esperiment)
    FOV_Shift_filename = os.path.join(experiment_folders["registration_dir"], 'FOV_shift.csv')
    if os.path.isfile(FOV_Shift_filename):
        FOV_shifts = pd.read_csv(FOV_Shift_filename, header=0, index_col=0)
        FOV_shifts_dict = (FOV_shifts).to_dict()
        left_right_shift = float(FOV_shifts_dict['0']['left_right_shift'])
        top_bot_shift = float(FOV_shifts_dict['0']['top_bot_shift'])

    left_crop = int((image_size_default - left_right_shift)/2)
    top_crop = int((image_size_default - top_bot_shift)/2)
    vertices = generate_vertices(fov_names, global_transform_all, image_size_default, image_size_default, REF_CYCLE, left_crop, top_crop)
    
    return vertices


def get_rgb_random():
    r = random.uniform(0,1)
    g = random.uniform(0,1)
    b = random.uniform(0.2,1)
    rgb = [r,g,b]
    return rgb


def add_target_points(args_tuples):

    experiment_folders = args_tuples[0]
    target = args_tuples[1]
    args = args_tuples[2]

    
    target_filename = os.path.join(experiment_folders["GeneTables_folder"], target + ".csv")
    coords_selected_df = pd.DataFrame()
    results_selected_df = pd.DataFrame()        

    if os.path.isfile(target_filename):

        cols_to_load, col_x, col_y = get_stitch_columns(target_filename)
        results_selected_df = pd.read_csv(target_filename, low_memory=True, header=0)
        # results_selected_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        results_selected_df.dropna(axis=0, how='any', inplace=True)
        coords_selected_df, results_filtered_df = filter_data_features(col_x, col_y, results_selected_df, args.CUT_OVERLAP)

        # if 'Background' in results_filtered_df.columns:
        #     results_filtered_df['Intensity_Diff'] = results_filtered_df['Intensity'] - results_filtered_df['Background'] 
                
        # Drop the dupilicates
        coords_selected_df = coords_selected_df.drop_duplicates(subset=[col_x,col_y])
        results_filtered_df = results_filtered_df.drop_duplicates(subset=[col_x,col_y])
        coords_selected_arr = coords_selected_df.to_numpy()
        coords_selected = coords_selected_arr
            
    else:
        print(f"{target} file does not exist in the {target_filename}" )
    
    out_tuple = (target, coords_selected, coords_selected_df, results_filtered_df)

    return out_tuple


def visualize_table_141(experiment_folders, gene_list):
    
    """ Visualize the RNA dots using Napari, CBR method of spot detection"""

    with gui_qt():

        viewer = Viewer()
        
        for target in gene_list:
                print(target)
                
                target_filename = os.path.join(experiment_folders["process_output_dir"], target+".csv")
                if os.path.isfile(target_filename):
                    results_filtered_df = pd.read_csv(target_filename, low_memory=False)

                    coords_selected_df = pd.DataFrame()
                    coords_selected_df['x'] = results_filtered_df['Spot location (X)']
                    coords_selected_df['y'] = results_filtered_df['Spot location (Y)']

                    coords_selected_arr = coords_selected_df.to_numpy()
                    coords_selected = coords_selected_arr

                    r = random.uniform(0,1)
                    g = random.uniform(0,1)
                    b = random.uniform(0.2,1)
                    rgb = [r,g,b]
                    layer = viewer.add_points(
                                # coords, 
                                coords_selected,
                                opacity=0.95,
                                face_color=rgb,
                                edge_color=rgb,
                                symbol="disc",
                                size=DISK_SIZE,
                                name=target,
                                visible = DEFAULT_VISIBLE_DOTS
                                )
                    layer.name = target + "_141"
                
                else:
                    print(f"{target} file does not exist in the {target_filename}" )
               
    return None 
 

def visualize_tables_side_by_side(experiment_folders, gene_list):

    """ Visualize the RNA dots using Napari, CBR method of spot detection compared with 1.4.1"""

    print(gene_list)
    from esper_process.cv_algorithms.fov_registration import (
        MODULO, FOV_Graph, estimate_shifts, flatten_list, get_num_rowcol_mod)
    
    def unique_values(x): return list(set(x))

    esperiment = experiment_utils.EsperExperiment.load_or_create(args.rootdir, make_folders=False)
    fov_names = unique_values(flatten_list([esperiment.cycles_by_number[cycle].FOVs.keys() for cycle in esperiment.cycles_by_number]))    
    
    FOV_Shift_filename = os.path.join(experiment_folders["registration_dir"], 'FOV_shift.csv')
    if os.path.isfile(FOV_Shift_filename):
        FOV_shifts = pd.read_csv(FOV_Shift_filename, header=0, index_col=0)
        FOV_shifts_dict = (FOV_shifts).to_dict()
        left_right_shift = float(FOV_shifts_dict['0']['left_right_shift'])
        top_bot_shift = float(FOV_shifts_dict['0']['top_bot_shift'])
    else:
        left_right_shift, top_bot_shift = \
            estimate_shifts(esperiment.cycles_by_number[1], N_RANDOM_ESTIMATE=10, method="cross-correlation")
    
    esperiment.left_right_shift = left_right_shift
    esperiment.top_bot_shift = top_bot_shift
    IMAGE_Y_SIZE = image_size_default
    IMAGE_X_SIZE = IMAGE_Y_SIZE
    esperiment.IMAGE_Y_SIZE = IMAGE_Y_SIZE
    esperiment.IMAGE_X_SIZE = IMAGE_X_SIZE    

    fov_graph = FOV_Graph(fov_names)
    fov_maps = fov_graph.fov_maps
    tissue_Graph = FOV_Graph(fov_names)
    total_num_rows = tissue_Graph.num_rows
    total_num_cols = tissue_Graph.num_cols

    stitched_img_size_row, stitched_img_size_col = get_num_rowcol_mod(esperiment, total_num_rows, total_num_cols, MODULO)
    print((MODULO - stitched_img_size_row%MODULO), (MODULO - stitched_img_size_col%MODULO))

    # stitched_img_size_row = esperiment.IMAGE_X_SIZE + (total_num_rows+2)*((int(left_right_shift)+1))
    # stitched_img_size_col = esperiment.IMAGE_Y_SIZE + (total_num_cols+2)*(int(top_bot_shift)+1)
    # stitched_img_size_row += (MODULO - stitched_img_size_row%MODULO)
    # stitched_img_size_col += (MODULO - stitched_img_size_col%MODULO)
    
    gene_list = sorted(gene_list)

    for _index, target in enumerate(gene_list):
        
        if _index==0:
            print(target)

            # viewer = Viewer()
            viewer = napari.viewer.Viewer()

            nuclei_stitched = experiment_folders["stitching_save_folder"]     
            Segmented_nuclei_stitched_filepath = os.path.join(nuclei_stitched, "Segmented_nuclei_stitched_Cycle_1.tif")
            nuclei_stitched_filepath = os.path.join(nuclei_stitched, "nuclei_stitched_Cycle_1.tif") #
            zarr_nuclei_stitched_filepath = os.path.join(nuclei_stitched, "nuclei_stitched_Cycle_1.zarr")
            
            if not os.path.isfile(nuclei_stitched_filepath):
                nuclei_stitched_filepath = os.path.join(nuclei_stitched, "nuclei_stitched_coarse_cycle_1.tif")

            if os.path.isfile(nuclei_stitched_filepath):
                nuclei_stitched = imread(nuclei_stitched_filepath)
                if len(nuclei_stitched.shape)>1:
                    nuclei_stitched = cvtColor(nuclei_stitched, COLOR_BGR2GRAY)

                viewer.add_image(nuclei_stitched, name="nucle_stitched_low_res", \
                    scale=(4,4), colormap="gray", blending="additive", opacity=0.5\
                        )
            
            try:
                get_target_points_15(experiment_folders, target, viewer)
                
            except Exception as e:
                print(f"Exception {e}")
                print("Oops!", e.__class__, "occurred.")
                print("Could not get or add 1.5 points")
            
            try:
                get_target_points_14(experiment_folders, left_right_shift, top_bot_shift, IMAGE_Y_SIZE, IMAGE_X_SIZE, target, viewer)                
            except Exception as e:
                print(f"Exception {e}")
                print("Oops!", e.__class__, "occurred.")
                print("Could not get or add 1.4.1 points")
                        
            @magicgui(
                auto_call=False,
                call_button=True, 
                shift_x={"widget_type": "FloatSpinBox", "min": -10000, "max": 10000},
                shift_y={"widget_type": "FloatSpinBox", "min": -10000, "max": 10000},
                )
            def Points_shift(layer: PointsData, shift_x: float = 0, shift_y: float = 0) -> LayerDataTuple:
                layer[:,0] += shift_y
                layer[:,1] += shift_x
                return (layer, {'name': 'X_Y Shifted Points', 'face_color':'yellow', 'edge_color':'yellow', 'symbol':'disc', 'size':DISK_SIZE_141}, 'points')

            @magicgui(gene={"choices": gene_list}, auto_call=True)
            def gene_list_updater(gene: str):
                print("Loading points for target: ", gene)
                get_target_points_15(experiment_folders, gene, viewer)
                get_target_points_14(experiment_folders, left_right_shift, top_bot_shift, IMAGE_Y_SIZE, IMAGE_X_SIZE, gene, viewer)
            
            dw_gene_list = viewer.window.add_dock_widget(gene_list_updater)
            viewer.layers.events.inserted.connect(gene_list_updater.reset_choices) #TODO: Disabled this, make sure all works 
            viewer.layers.events.changed.connect(gene_list_updater.reset_choices)  
            # viewer.layers.events.changed.connect(gene_list_updater.from_callable)  
            viewer.layers.events.changed.connect(gene_list_updater._on_change)  
            # viewer.window._qt_window.tabifyDockWidget(dw_slider_shift_correction, dw_gene_list)

            @magicgui(auto_call=True, SNR={'widget_type': Widget_type, 'min': MIN_CRA_COL_VALUE, 'max': MAX_CRA_COL_VALUE})
            def CBR_slider_features(layer: napari.layers.Points, SNR=MIN_CRA_COL_VALUE):
                _features = layer.features
                if 'CBR' in _features.columns:
                    layer.shown = _features['CBR'] > SNR
            # dw_csbr_slider = viewer.window.add_dock_widget(CBR_slider_features)
            # viewer.layers.events.inserted.connect(CBR_slider_features.reset_choices)
            # viewer.layers.events.changed.connect(CBR_slider_features.reset_choices)
            # viewer.window._qt_window.tabifyDockWidget(dw_slider_shift_correction, dw_csbr_slider)

            @magicgui(auto_call=True, Intennsity={'widget_type': 'FloatSpinBox', 'min': MIN_INTENSITY, 'max': MAX_INTENSITY})
            def Intensity_slider_features(layer: napari.layers.Points, Intennsity=MIN_INTENSITY):
                _features = layer.features
                if 'Intensity' in _features.columns:
                    layer.shown = _features['Intensity'] > Intennsity
            # dw_intensity_slider = viewer.window.add_dock_widget(Intensity_slider_features)
            # viewer.layers.events.inserted.connect(Intensity_slider_features.reset_choices)
            # viewer.layers.events.changed.connect(Intensity_slider_features.reset_choices)
            # viewer.window._qt_window.tabifyDockWidget(dw_slider_shift_correction, dw_intensity_slider)

            @magicgui(
                auto_call=False, 
                main_window=True,
                call_button=True, 
                SNR={'widget_type': Widget_type, 'min': MIN_CRA_COL_VALUE, 'max': MAX_CRA_COL_VALUE},
                SNR_Max={'widget_type': Widget_type, 'min': MIN_CRA_COL_VALUE, 'max': MAX_CRA_COL_VALUE},
                Intensity={'widget_type': Widget_type, 'min': MIN_INTENSITY, 'max': MAX_INTENSITY}, 
                Intensity_max={'widget_type': Widget_type, 'min': MIN_INTENSITY, 'max': MAX_INTENSITY}, 
                Background={'widget_type': Widget_type, 'min': MIN_BACKGROUND, 'max': MAX_BACKGROUND},
                Background_Max={'widget_type': Widget_type, 'min': MIN_BACKGROUND, 'max': MAX_BACKGROUND},
                Intensity_Diff={'widget_type': Widget_type, 'min': MIN_INTENSITY_DIFF, 'max': MAX_INTENSITY_DIFF},
                Intensity_Diff_Max={'widget_type': Widget_type, 'min': MIN_INTENSITY_DIFF, 'max': MAX_INTENSITY_DIFF},                
                Distance_to_Cell={'widget_type': Widget_type, 'min': MIN_DISTANCE_TO_CELL, 'max': MAX_DISTANCE_TO_CELL},  
                )
            def Feature_slider(
                layer: napari.layers.Points, 
                SNR=MIN_CRA_COL_VALUE, 
                SNR_Max=MAX_CRA_COL_VALUE,
                Intensity=MIN_INTENSITY, 
                Intensity_max=MAX_INTENSITY, 
                Background=MIN_BACKGROUND,
                Background_Max=MAX_BACKGROUND,
                Intensity_Diff=MIN_INTENSITY_DIFF,
                Intensity_Diff_Max=MAX_INTENSITY_DIFF,                
                Distance_to_Cell=MAX_DISTANCE_TO_CELL, 
                ):

                _features = layer.features
                if 'CBR' in _features.columns:
                    layer_shown_cbr = (_features['CBR'] > SNR) & (_features['CBR']<SNR_Max)
                    layer.shown = layer_shown_cbr
                if 'Intensity' in _features.columns:
                    layer_shown_intensity = (_features['Intensity'] > Intensity) & (_features['Intensity'] < Intensity_max)
                    layer.shown = (layer.shown) & (layer_shown_intensity) 
                if 'distance_to_cell' in _features.columns:
                    layer_shown_dtc = (_features['distance_to_cell'] <= Distance_to_Cell) 
                    layer.shown = (layer.shown) & (layer_shown_dtc) 
                if 'Intensity_Diff' in _features.columns:
                    layer_shown_id = (_features['Intensity_Diff'] > Intensity_Diff) & (_features['Intensity_Diff'] < Intensity_Diff_Max)
                    layer.shown = (layer.shown) & (layer_shown_id) 
                if 'Background' in _features.columns:
                    layer_shown_bg = (_features['Background'] > Background) & (_features['Background'] < Background_Max)
                    layer.shown = (layer.shown) & (layer_shown_bg) 

            container = widgets.Container(widgets=[Feature_slider, Points_shift])
            viewer.window.add_dock_widget(container)

            run_napari()


def get_target_points_14(experiment_folders, left_right_shift, top_bot_shift, IMAGE_Y_SIZE, IMAGE_X_SIZE, target, viewer):
    target_filename_141 = os.path.join(experiment_folders["process_output_dir"], target+".csv")
    if os.path.isfile(target_filename_141):
        results_filtered_df = pd.read_csv(target_filename_141, low_memory=False, header=0)
        # results_filtered_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        results_filtered_df.dropna(axis=0, how='any', inplace=True)

        coords_selected_df = pd.DataFrame()
        coords_selected_df['x'] = results_filtered_df['Spot location (Y)']
        coords_selected_df['y'] = results_filtered_df['Spot location (X)']

        coords_selected_arr = coords_selected_df.to_numpy()
        coords_selected_141 = coords_selected_arr/SCALE_141

                # This setting kinda worked 
                # shift_x_mean = IMAGE_X_SIZE - left_right_shift /SCALE_141 
                # shift_y_mean = IMAGE_Y_SIZE - top_bot_shift/SCALE_141 

        shift_x_mean = IMAGE_X_SIZE - (IMAGE_X_SIZE - left_right_shift)/2 
        shift_y_mean = IMAGE_Y_SIZE - (IMAGE_Y_SIZE - top_bot_shift)/2 

    else:
        print(f"{target} file does not exist in the {target_filename_141}" )

    coords_selected_141[:,0] += shift_x_mean - SHIFT_X_BIAS
    coords_selected_141[:,1] += shift_y_mean - SHIFT_Y_BIAS

    layer = viewer.add_points(
                coords_selected_141,
                opacity=0.95,
                face_color="cyan",
                edge_color="cyan",
                symbol="disc",
                size=DISK_SIZE_141,
                name=target,
                )
    layer.name = target + "_141"
    
    layer.refresh()
    layer.refresh_colors(update_color_mapping=True)


def get_target_points_15(experiment_folders, target, viewer):
    target_filename = os.path.join(experiment_folders["GeneTables_folder"], target+".csv")
    if os.path.isfile(target_filename):
        cols_to_load, col_x, col_y = get_stitch_columns(target_filename)
        results_filtered_df = pd.read_csv(target_filename, low_memory=False, header=0)
        # results_filtered_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        results_filtered_df.dropna(axis=0, how='any', inplace=True)
        coords_selected_df, results_selected_df = filter_data_features(col_x, col_y, results_filtered_df)
        coords_selected_arr = coords_selected_df.to_numpy()
        coords_selected = coords_selected_arr

        # SNR_array = results_selected_df['CBR']
        # Intensity_array = results_selected_df['Intensity']
        # Background_array = results_selected_df['Background']
        # Intensity_diff_array = results_selected_df['Intensity_Diff']
        # distance_to_cell_array = results_selected_df['distance_to_cell']

        _ = add_points_with_filter(viewer, coords_selected, results_selected_df, target, color_of="magenta", size_of=DISK_SIZE_150)
          
    else:
        print(f"{target} file does not exist in the {target_filename}" )


def add_points_with_filter(viewer, coords_selected, features_array, target_name, color_of="magenta", size_of=DISK_SIZE_150):
    
    try:
        layer_151 = viewer.add_points(
                            coords_selected,
                            features=features_array,
                            opacity=0.95,
                            face_color=color_of,
                            edge_color=color_of,
                            symbol="disc",
                            size=size_of,
                            name=target_name,
                            )
        layer_151.name = target_name + "_150"

        return True 
    except:
        return False 


def filter_data_features(col_x, col_y, results_filtered_df, _CUT_OVERLAP=CUT_OVERLAP):

    SKIP_CUT = False   
    
    if 'CBR' in results_filtered_df.columns:
        results_filtered_df = results_filtered_df[pd.to_numeric(results_filtered_df['CBR'], errors='coerce').notnull()]
        # results_filtered_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        results_filtered_df.dropna(axis=0, how='any', inplace=True)

        if 'Background' in results_filtered_df.columns and 'Intensity_Diff' not in results_filtered_df.columns:
            results_filtered_df['Intensity_Diff'] = results_filtered_df['Intensity'] - results_filtered_df['Background'] 

        if SKIP_CUT:
            results_filtered_df['CBR'] = results_filtered_df['CBR'].astype(float)
            results_filtered_df = results_filtered_df.sort_values(by=['CBR'])
            results_filtered_df = results_filtered_df.loc[results_filtered_df['CBR'] > MIN_CRA_COL_VALUE]
            results_filtered_df = results_filtered_df.loc[results_filtered_df['CBR'] < MAX_CRA_COL_VALUE]
                        
            if 'Background' in results_filtered_df.columns:
                results_filtered_df = results_filtered_df.loc[results_filtered_df['Background'] > MIN_BACKGROUND]
                results_filtered_df = results_filtered_df.loc[results_filtered_df['Background'] < MAX_BACKGROUND]
                            
            if 'Intensity' in results_filtered_df.columns:
                results_filtered_df = results_filtered_df.loc[results_filtered_df['Intensity'] > MIN_INTENSITY]
                results_filtered_df = results_filtered_df.loc[results_filtered_df['Intensity'] < MAX_INTENSITY]

            if 'Background' in results_filtered_df.columns:
                results_filtered_df = results_filtered_df.loc[results_filtered_df['Intensity_Diff'] > MIN_INTENSITY_DIFF]
                results_filtered_df = results_filtered_df.loc[results_filtered_df['Intensity_Diff'] < MAX_INTENSITY_DIFF]
                            
            if 'distance_to_cell' in results_filtered_df.columns:
                results_filtered_df = results_filtered_df.loc[results_filtered_df['distance_to_cell'] > MIN_DISTANCE_TO_CELL]
                results_filtered_df = results_filtered_df.loc[results_filtered_df['distance_to_cell'] < MAX_DISTANCE_TO_CELL]

    if _CUT_OVERLAP:
        x_start, x_end, y_start, y_end = get_cutting_coorids() 
        if 'x' in results_filtered_df.columns:
            results_filtered_df = results_filtered_df.loc[results_filtered_df['x'] > x_start]
            results_filtered_df = results_filtered_df.loc[results_filtered_df['x'] < x_end]
        if 'y' in results_filtered_df.columns:
            results_filtered_df = results_filtered_df.loc[results_filtered_df['y'] > y_start]
            results_filtered_df = results_filtered_df.loc[results_filtered_df['y'] < y_end]              

    coords_selected_df = pd.DataFrame()
    if col_x in results_filtered_df.columns:
        results_filtered_df = results_filtered_df[pd.to_numeric(results_filtered_df[col_y], errors='coerce').notnull()]
        # results_filtered_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        results_filtered_df.dropna(axis=0, how='any', inplace=True)
        coords_selected_df[col_x] = results_filtered_df[col_y]
    if col_y in results_filtered_df.columns:
        results_filtered_df = results_filtered_df[pd.to_numeric(results_filtered_df[col_x], errors='coerce').notnull()]
        # results_filtered_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        results_filtered_df.dropna(axis=0, how='any', inplace=True)
        coords_selected_df[col_y] = results_filtered_df[col_x]

    return coords_selected_df, results_filtered_df


def gene_list_cleanup(gene_list, REMOVE_GENES_NOT_INTEREST):
    gene_list = list(set(gene_list))
    for el in REMOVE_GENES_NOT_INTEREST:
        if el in gene_list:
            gene_list.remove(el)
    
    gene_list_to_remove = []
    for genes_to_remove in unpopulated_gene_channel:
        for gene_name in gene_list:
            if genes_to_remove in gene_name:
                gene_list_to_remove.append(gene_name)
                
    gene_list = [gene_name for gene_name in gene_list if gene_name not in gene_list_to_remove]
    
    print(gene_table_exist, laser_to_ch_dict, gene_list)
    
    return gene_list


def parse_args():
    """
    Parse arguments.
    """
    # args
    parser = argparse.ArgumentParser(
        description="""
        Visualize the spot tablese.
        """,
        epilog="""
        This code will process a given experiment folder to visualize the spot tables.
        """,
    )

    parser.add_argument(
        "--rootdir",
        help="The input directory containing the experiment data. ",
        metavar="dir",
        default="",
    )

    parser.add_argument(
        "--CUT_OVERLAP",
        help="Whether cut the overlap x-y coordinates.",
        action="store_true",
        default=CUT_OVERLAP,
    )

    parser.add_argument(
        "--Recon_size_default",
        help="Cutting the overlap, getting the default size",
        type=int,
        default=Recon_size_default,
    )

    parser.add_argument(
        "--SKIP_NUCLEI",
        help="Whether cut the overlap x-y coordinates.",
        action="store_true",
        default=SKIP_NUCLEI,
    )

    parser.add_argument(
        "--threadcount",
        help="Using a multiprocessing in Python.",
        type=int,
        default=THREAD_COUNT,
    )

    parser.add_argument(
        "--multiprocess",
        help="Using a multiprocessing in Python.",
        action="store_true",
        default=multiprocess,
    )

    parser.add_argument(
        "--SHOW_141",
        help="Whether to show the spot results from 1.4.1.",
        action="store_true",
        default=SHOW_141,
    )

    parser.add_argument(
        "--SHOW_SIDE_BY_SIDE",
        help="Whether to show the spot results from 1.4.1.",
        action="store_true",
        default=SHOW_SIDE_BY_SIDE,
    )
    

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    
    """ Main function : visualize the spot tables"""
    
    start_processing_time = time.time() 
    print("Running spot table visualization")
    args = parse_args()
    root_dir = args.rootdir    
    experiment_folders = experiment_utils.initialize_output_folders(root_dir)
    gene_table_exist, gene_list, laser_to_ch_dict = experiment_utils.get_gene_table_params(experiment_folders)    
    print(gene_table_exist, laser_to_ch_dict, gene_list)
    
    gene_list = gene_list_cleanup(gene_list, REMOVE_GENES_NOT_INTEREST)
    gene_list = sorted(gene_list[0:])

    if args.SHOW_141:
        _ = visualize_table_141(experiment_folders, gene_list)
    elif args.SHOW_SIDE_BY_SIDE:
        _ = visualize_tables_side_by_side(experiment_folders, gene_list)
    else:
        end_processing_time = visualize_table(experiment_folders, gene_list, args)

        total_processing_time = end_processing_time - start_processing_time
        total_processing_time_per_gene = total_processing_time/len(gene_list)
        print(f"Total processing time = {total_processing_time} and per image = {total_processing_time_per_gene}")

    
