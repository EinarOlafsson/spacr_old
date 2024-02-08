import subprocess
import sys
import os
import json
import shutil
import platform
import getpass

def get_paths(env_name):
    conda_executable = "conda.exe" if sys.platform == "win32" else "conda"
    python_executable = "python.exe" if sys.platform == "win32" else "python"
    pip_executable = "pip.exe" if sys.platform == "win32" else "pip"

    conda_path = shutil.which(conda_executable)
    
    if not conda_path:
        if sys.platform == "win32":
            conda_path = "C:\\ProgramData\\Anaconda3\\Scripts\\conda.exe"
        else:
            home_directory = os.path.expanduser('~')
            conda_path = os.path.join(home_directory, 'anaconda3', 'bin', conda_executable)

    if not os.path.exists(conda_path):
        if sys.platform == "win32":
            username = getpass.getuser()
            conda_path = f"C:\\Users\\{username}\\Anaconda3\\Scripts\\conda.exe"

    if not conda_path or not os.path.exists(conda_path):
        print("Conda is not found in the system PATH")
        return None, None, None, None

    conda_dir = os.path.dirname(os.path.dirname(conda_path))
    env_path = os.path.join(conda_dir, 'envs', env_name)
    
    if sys.platform == "win32":
        pip_path = os.path.join(env_path, 'Scripts', pip_executable)
        python_path = os.path.join(env_path, python_executable)
    else:
        python_path = os.path.join(env_path, 'bin', python_executable)
        pip_path = os.path.join(env_path, 'bin', pip_executable)

    return conda_path, python_path, pip_path, env_path

# create new kernel
def add_kernel(env_name, display_name):
    _, python_path, _, _ = get_paths(env_name)
    if not python_path:
        print(f"Failed to locate the Python executable for '{env_name}'")
        return

    try:
        subprocess.run([python_path, '-m', 'ipykernel', 'install', '--user', '--name', env_name, '--display-name', display_name])
        print(f"Kernel for '{env_name}' with display name '{display_name}' added successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to add kernel. Error: {e}")
        print(f"kernel can be added manualy with: python -m ipykernel install --user --name {env_name} --display-name {display_name}")

def create_environment(conda_PATH, env_name):
    print(f"Creating environment {env_name}...")
    subprocess.run([conda_PATH, "create", "-n", env_name, "python=3.9", "-y"])

def has_nvidia_gpu():
    try:
        if sys.platform == "win32":
            # For Windows, use systeminfo
            result = subprocess.run("systeminfo", capture_output=True, text=True)
            return "NVIDIA" in result.stdout
        else:
            # For Linux and macOS, use nvidia-smi
            subprocess.run("nvidia-smi", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
    except subprocess.CalledProcessError:
        # nvidia-smi not found or failed, assuming no NVIDIA GPU
        return False

# Install dependencies in a specified kernel environment.
def install_dependencies_in_kernel(dependencies, env_name):
    
    conda_PATH, _, pip_PATH, _ = get_paths(env_name)

    # Check if conda is available
    if not conda_PATH:
        raise EnvironmentError("Conda executable not found.")
    
    # Get the current Conda configuration for channels
    result = subprocess.run([conda_PATH, "config", "--show", "channels"], capture_output=True, text=True)
    channels = result.stdout

    # Check if 'conda-forge' is in the channels list
    if 'conda-forge' not in channels:
        # If 'conda-forge' is not in the channels, add it
        subprocess.run([conda_PATH, "config", "--add", "channels", "conda-forge"])
        print("Added conda-forge to channels.")
    
    # Update conda
    print("Updating Conda...")
    subprocess.run([conda_PATH, "update", "-n", "base", "-c", "defaults", "conda", "-y"])

    # Check for NVIDIA GPU
    if has_nvidia_gpu():
        print("NVIDIA GPU found. Installing PyTorch with GPU support.")
        subprocess.run([pip_PATH, "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])
    else:
        print("No NVIDIA GPU found. Installing PyTorch for CPU.")
        subprocess.run([pip_PATH, "install", "torch", "torchvision", "torchaudio"])

    # Install torch, torchvision, torchaudio with pip
    #print("Installing torch")
    #subprocess.run([pip_PATH, "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])
                    
    # Install cellpose
    print("Installing cellpose")
    subprocess.run([pip_PATH, "install", "cellpose"])

    # Install remaining dependencies with conda
    for package in dependencies:
        print(f"Installing {package}")
        subprocess.run([conda_PATH, "install", "-n", env_name, package, "-y"])

    pip_packages = ["numpy==1.24.0", "numba==0.58.0"]
    
    for package in pip_packages:
    	print(f"Installing {package}")
    	subprocess.run([pip_PATH, "install", package])

    print("Dependencies installation complete.")

env_name = "spacr_data_generation"

conda_PATH, python_PATH, pip_PATH, env_PATH = get_paths(env_name)

dependencies = ["pandas", "ipykernel", "mahotas","scikit-learn", "scikit-image", "seaborn", "matplotlib", "xgboost", "moviepy", "ipywidgets"]

if not os.path.exists(env_PATH):
	print(f'System type: {sys.platform}')
	print(f'PATH to conda: {conda_PATH}')
	print(f'PATH to python: {python_PATH}')
	print(f'PATH to pip: {pip_PATH}')
	print(f'PATH to new environment: {env_PATH}')
	
	create_environment(conda_PATH, env_name)
	install_dependencies_in_kernel(dependencies, env_name)
	add_kernel(env_name, env_name)
	print(f"Environment '{env_name}' created and added as a Jupyter kernel.")
	print(f"Refresh the page, set {env_name} as the kernel and run cell again")
	sys.exit()

################################################################################################################################################################################

import os, gc, re, cv2, csv, math, time, torch, json, traceback

print('Torch available:', torch.cuda.is_available())
print('CUDA version:',torch.version.cuda)

import string, shutil, random, logging, sqlite3, cellpose, imageio

# Image and array processing
from cellpose import models
#from cellpose import dynamics
from torch.cuda.amp import autocast
import pandas as pd
import numpy as np
from PIL import Image, ImageTk, ImageOps
import tifffile

# other
from queue import Queue
import tkinter as tk
from tkinter import Tk, Label, Button
from concurrent.futures import ThreadPoolExecutor
import threading  # Make sure this line is here
from pathlib import Path
import xgboost as xgb

import moviepy.editor as mpy
import ipywidgets as widgets
from ipywidgets import IntProgress, interact, interact_manual, Button, HBox, IntSlider
from IPython.display import display, clear_output, HTML

# Data visualization
#%matplotlib inline
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from collections import defaultdict

# scikit-image
from skimage import exposure, measure, morphology, filters
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries, clear_border, watershed
from skimage.morphology import opening, disk, closing, dilation, square
from skimage.exposure import rescale_intensity
from skimage.measure import label, regionprops_table, regionprops, shannon_entropy, find_contours
from skimage.feature import graycomatrix, graycoprops, peak_local_max
from mahotas.features import zernike_moments

# scikit-learn
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

# SciPy
import scipy.ndimage as ndi
from scipy.stats import pearsonr
from scipy.interpolate import UnivariateSpline
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_erosion, binary_dilation as binary_erosion, binary_dilation, distance_transform_edt, generate_binary_structure
from scipy.spatial.distance  import cdist
from scipy.stats import zscore

# parallel processing
import multiprocessing as mp
from multiprocessing import Lock
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) # Ignore RuntimeWarning
warnings.filterwarnings("ignore")

def list_folders(src):
    # List to hold the names of folders
    folders = []
    # Check if the source directory exists
    if not os.path.exists(src):
        print(f"The directory {src} does not exist.")
        return folders
    # Iterate over all items in the directory
    for item in os.listdir(src):
        # Construct the full path
        item_path = os.path.join(src, item)
        # Check if the item is a directory
        if os.path.isdir(item_path):
            folders.append(item)
    # Sort the list of folders alphabetically
    sorted_folders = sorted(folders)
    return sorted_folders

def z_to_mip(src, regex, batch_size=100, pick_slice=False, skip_mode='01'):
    regular_expression = re.compile(regex)
    images_by_key = defaultdict(list)
    stack_path = os.path.join(src, 'stack')
    if not os.path.exists(stack_path) or (os.path.isdir(stack_path) and len(os.listdir(stack_path)) == 0):
        all_filenames = [filename for filename in os.listdir(src) if filename.endswith('.tif')]
        print(f'All_files:{len(all_filenames)} in {src}')
        for i in range(0, len(all_filenames), batch_size):
            batch_filenames = all_filenames[i:i+batch_size]
            for filename in batch_filenames:
                match = regular_expression.match(filename)
                if match:
                    try:
                        try:
                            plate = match.group('plateID')
                        except:
                            plate = os.path.basename(src)
                        well = match.group('wellID')
                        field = match.group('fieldID')
                        channel = match.group('chanID')
                        mode = None

                        if pick_slice:
                            try:
                                mode = match.group('AID')
                            except IndexError:
                                sliceid = '00'

                            if mode == skip_mode:
                                continue
                                
                        key = (plate, well, field, channel, mode)
                        with Image.open(os.path.join(src, filename)) as img:
                            images_by_key[key].append(np.array(img))
                    except IndexError:
                        print(f"Could not extract information from filename {filename} using provided regex")
                else:
                    print(f"Filename {filename} did not match provided regex")

            if pick_slice:
                for key in images_by_key:
                    plate, well, field, channel, mode = key
                    max_intensity_slice = max(images_by_key[key], key=lambda x: np.percentile(x, 90))
                    #max_intensity_slice = max(images_by_key[key], key=lambda x: np.sum(x))
                    mip_image = Image.fromarray(max_intensity_slice)

                    output_dir = os.path.join(src, channel)
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = f'{plate}_{well}_{field}.tif'
                    output_path = os.path.join(output_dir, output_filename)
                    
                    if os.path.exists(output_path):                        
                        print(f'WARNING: A file with the same name already exists at location {output_filename}')
                    else:
                        mip_image.save(output_path)
            else:
                for key, images in images_by_key.items():
                    mip = np.max(np.stack(images), axis=0)
                    mip_image = Image.fromarray(mip)
                    plate, well, field, channel = key[:4]
                    output_dir = os.path.join(src, channel)
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = f'{plate}_{well}_{field}.tif'
                    output_path = os.path.join(output_dir, output_filename)

                    if os.path.exists(output_path):                        
                        print(f'WARNING: A file with the same name already exists at location {output_filename}')
                    else:
                        mip_image.save(output_path)

            images_by_key.clear()

        # Move original images to a new directory
        valid_exts = ['.tif', '.png']
        newpath = os.path.join(src, 'orig')
        os.makedirs(newpath, exist_ok=True)
        for filename in os.listdir(src):
            if os.path.splitext(filename)[1] in valid_exts:
                move = os.path.join(newpath, filename)
                if os.path.exists(move):
                    print(f'WARNING: A file with the same name already exists at location {move}')
                else:
                    shutil.move(os.path.join(src, filename), move)
    return

def move_to_chan_folder(src, regex, timelapse=False):
    src = Path(src)
    valid_exts = ['.tif', '.png']

    if not (src / 'stack').exists():
        for file in src.iterdir():
            if file.is_file():
                name, ext = file.stem, file.suffix
                if ext in valid_exts:
                    metadata = re.match(regex, file.name)                    
                    try:
                        plateID = metadata.group('plateID')
                    except:
                        plateID = src.name
                    wellID = metadata.group('wellID')
                    fieldID = metadata.group('fieldID')
                    chanID = metadata.group('chanID')
                    timeID = metadata.group('timeID')
                    newname = f"{plateID}_{wellID}_{fieldID}_{timeID if timelapse else ''}{ext}"
                    newpath = src / chanID
                    move = newpath / newname
                    if move.exists():
                        print(f'WARNING: A file with the same name already exists at location {move}')
                    else:
                        newpath.mkdir(exist_ok=True)
                        shutil.move(file, move)
    return
# Generate random colour cmap
def random_cmap(num_objects=100):
    #num_objects = len(unique_labels[unique_labels != 0])
    random_colors = np.random.rand(num_objects+1, 4)
    random_colors[:, 3] = 1
    random_colors[0, :] = [0, 0, 0, 1]
    random_cmap = mpl.colors.ListedColormap(random_colors)
    return random_cmap

def plot_arrays(src, figuresize=50, cmap='inferno', nr=1, normalize=True, q1=1, q2=99):
    mask_cmap = random_cmap()
    paths = []
    for file in os.listdir(src):
        if file.endswith('.npy'):
            path = os.path.join(src, file)
            paths.append(path)
    paths = random.sample(paths, nr)
    for path in paths:
        print(f'Image path:{path}')
        img = np.load(path)
        if normalize:
            img = normalize_to_dtype(array=img, q1=q1, q2=q2)
        dim = img.shape
        if len(img.shape)>2:
            array_nr = img.shape[2]
            fig, axs = plt.subplots(1, array_nr,figsize=(figuresize,figuresize))
            for channel in range(array_nr):
                i = np.take(img, [channel], axis=2)
                axs[channel].imshow(i, cmap=plt.get_cmap(cmap))
                axs[channel].set_title('Channel '+str(channel),size=24)
                axs[channel].axis('off')
        else:
            fig, ax = plt.subplots(1, 1,figsize=(figuresize,figuresize))
            ax.imshow(img, cmap=plt.get_cmap(cmap))
            ax.set_title('Channel 0',size=24)
            ax.axis('off')
        fig.tight_layout()
        plt.show()
    return

def remove_multiobject_cells(stack, mask_dim, cell_dim, nucleus_dim, pathogen_dim, object_dim):
    cell_mask = stack[:, :, mask_dim]
    nucleus_mask = stack[:, :, nucleus_dim]
    pathogen_mask = stack[:, :, pathogen_dim]
    object_mask = stack[:, :, object_dim]

    for cell_label in np.unique(cell_mask)[1:]:
        cell_region = cell_mask == cell_label
        labels_in_cell = np.unique(object_mask[cell_region])
        if len(labels_in_cell) > 2:
            cell_mask[cell_region] = 0
            nucleus_mask[cell_region] = 0
            for pathogen_label in labels_in_cell[1:]:  # Skip the first label (0)
                pathogen_mask[pathogen_mask == pathogen_label] = 0

    stack[:, :, cell_dim] = cell_mask
    stack[:, :, nucleus_dim] = nucleus_mask
    stack[:, :, pathogen_dim] = pathogen_mask
    return stack

def remove_advanced_filter_cells(stack, mask_dim, cell_dim, nucleus_dim, pathogen_dim, object_dim):
    cell_mask = stack[:, :, mask_dim]
    nucleus_mask = stack[:, :, nucleus_dim]
    pathogen_mask = stack[:, :, pathogen_dim]
    object_mask = stack[:, :, object_dim]

    for cell_label in np.unique(cell_mask)[1:]: 
        cell_region = cell_mask == cell_label
        cell_area = np.sum(cell_region)
        
        # Calculate nucleus and pathogen areas within the cell
        nucleus_area = np.sum(nucleus_mask[cell_region])
        pathogen_area = np.sum(pathogen_mask[cell_region])

        # Advanced filtration based on nucleus and pathogen areas
        if advanced_filtration and (nucleus_area > 0.5 * cell_area or pathogen_area > 0.5 * cell_area):
            cell_mask[cell_region] = 0
            nucleus_mask[cell_region] = 0
            pathogen_mask[cell_region] = 0
            continue

    stack[:, :, cell_dim] = cell_mask
    stack[:, :, nucleus_dim] = nucleus_mask
    stack[:, :, pathogen_dim] = pathogen_mask
    
    return stack

def remove_noninfected(stack, cell_dim, nucleus_dim, pathogen_dim):
    if not cell_dim is None:
    	cell_mask = stack[:, :, cell_dim]
    else:
    	cell_mask = np.zeros_like(stack)
    if not nucleus_dim is None:
    	nucleus_mask = stack[:, :, nucleus_dim]
    else:
    	nucleus_mask = np.zeros_like(stack)
    	
    if not pathogen_dim is None:
    	pathogen_mask = stack[:, :, pathogen_dim]
    else:
    	pathogen_mask = np.zeros_like(stack)
	
    for cell_label in np.unique(cell_mask)[1:]:
        cell_region = cell_mask == cell_label
        labels_in_cell = np.unique(pathogen_mask[cell_region])
        if len(labels_in_cell) <= 1:
            cell_mask[cell_region] = 0
            nucleus_mask[cell_region] = 0
    if not cell_dim is None:
    	stack[:, :, cell_dim] = cell_mask
    if not nucleus_dim is None:
    	stack[:, :, nucleus_dim] = nucleus_mask
    return stack

def remove_border_pathogens(stack, cell_dim, nucleus_dim, pathogen_dim):
    cell_mask = stack[:, :, cell_dim]
    nucleus_mask = stack[:, :, nucleus_dim]
    pathogen_mask = stack[:, :, pathogen_dim]

    cell_labels = np.unique(cell_mask)[1:]  # Get unique cell labels, excluding background
    for cell_label in cell_labels:
        cell_region = cell_mask == cell_label
        pathogens_in_cell = np.unique(pathogen_mask[cell_region])
        pathogens_in_cell = pathogens_in_cell[pathogens_in_cell != 0]  # Exclude background

        # Create a border for the cell using dilation and subtract the original cell mask
        cell_border = binary_dilation(cell_region) & ~cell_region

        for pathogen_label in pathogens_in_cell:
            pathogen_region = pathogen_mask == pathogen_label
            
            # If the pathogen is touching the border of the cell, remove the cell, corresponding nucleus and the pathogen
            if np.any(pathogen_region & cell_border):
                cell_mask[cell_region] = 0
                nucleus_mask[cell_region] = 0
                pathogen_mask[pathogen_region] = 0
                break

    stack[:, :, cell_dim] = cell_mask
    stack[:, :, nucleus_dim] = nucleus_mask
    stack[:, :, pathogen_dim] = pathogen_mask
    return stack

def remove_outside_objects(stack, cell_dim, nucleus_dim, pathogen_dim):
    if not cell_dim is None:
    	cell_mask = stack[:, :, cell_dim]
    else:
    	return stack
    nucleus_mask = stack[:, :, nucleus_dim]
    pathogen_mask = stack[:, :, pathogen_dim]
    pathogen_labels = np.unique(pathogen_mask)[1:]
    for pathogen_label in pathogen_labels:
        pathogen_region = pathogen_mask == pathogen_label
        cell_in_pathogen_region = np.unique(cell_mask[pathogen_region])
        cell_in_pathogen_region = cell_in_pathogen_region[cell_in_pathogen_region != 0]  # Exclude background
        if len(cell_in_pathogen_region) == 0:
            pathogen_mask[pathogen_region] = 0
            corresponding_nucleus_region = nucleus_mask == pathogen_label
            nucleus_mask[corresponding_nucleus_region] = 0
    stack[:, :, cell_dim] = cell_mask
    stack[:, :, nucleus_dim] = nucleus_mask
    stack[:, :, pathogen_dim] = pathogen_mask
    return stack
            
def plot_merged(src, settings):
    
    def generate_mask_random_cmap(mask):  
        unique_labels = np.unique(mask)
        num_objects = len(unique_labels[unique_labels != 0])
        random_colors = np.random.rand(num_objects+1, 4)
        random_colors[:, 3] = 1
        random_colors[0, :] = [0, 0, 0, 1]
        random_cmap = mpl.colors.ListedColormap(random_colors)
        return random_cmap
    
    def get_colours_merged(outline_color):
        if outline_color == 'rgb':
            outline_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # rgb
        elif outline_color == 'bgr':
            outline_colors = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]  # bgr
        elif outline_color == 'gbr':
            outline_colors = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]  # gbr
        elif outline_color == 'rbg':
            outline_colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]  # rbg
        else:
            outline_colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]  # rbg
        return outline_colors
    
    def filter_objects_in_plot(stack, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim, mask_dims, filter_min_max, include_multinucleated, include_multiinfected):

        stack = remove_outside_objects(stack, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim)
        
        for i, mask_dim in enumerate(mask_dims):
            if not filter_min_max is None:
                min_max = filter_min_max[i]
            else:
                min_max = [0, 100000]

            mask = np.take(stack, mask_dim, axis=2)
            props = measure.regionprops_table(mask, properties=['label', 'area'])
            avg_size_before = np.mean(props['area'])
            total_count_before = len(props['label'])
            
            if not filter_min_max is None:
                valid_labels = props['label'][np.logical_and(props['area'] > min_max[0], props['area'] < min_max[1])]  
                stack[:, :, mask_dim] = np.isin(mask, valid_labels) * mask  

            props_after = measure.regionprops_table(stack[:, :, mask_dim], properties=['label', 'area']) 
            avg_size_after = np.mean(props_after['area'])
            total_count_after = len(props_after['label'])

            if mask_dim == cell_mask_dim:
                if include_multinucleated is False and nucleus_mask_dim is not None:
                    stack = remove_multiobject_cells(stack, mask_dim, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim, object_dim=pathogen_mask_dim)
                if include_multiinfected is False and cell_mask_dim is not None and pathogen_mask_dim is not None:
                    stack = remove_multiobject_cells(stack, mask_dim, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim, object_dim=nucleus_mask_dim)
                cell_area_before = avg_size_before
                cell_count_before = total_count_before
                cell_area_after = avg_size_after
                cell_count_after = total_count_after
            if mask_dim == nucleus_mask_dim:
                nucleus_area_before = avg_size_before
                nucleus_count_before = total_count_before
                nucleus_area_after = avg_size_after
                nucleus_count_after = total_count_after
            if mask_dim == pathogen_mask_dim:
                pathogen_area_before = avg_size_before
                pathogen_count_before = total_count_before
                pathogen_area_after = avg_size_after
                pathogen_count_after = total_count_after

        if cell_mask_dim is not None:
            print(f'removed {cell_count_before-cell_count_after} cells, cell size from {cell_area_before} to {cell_area_after}')
        if nucleus_mask_dim is not None:
            print(f'removed {nucleus_count_before-nucleus_count_after} nuclei, nuclei size from {nucleus_area_before} to {nucleus_area_after}')
        if pathogen_mask_dim is not None:
            print(f'removed {pathogen_count_before-pathogen_count_after} pathogens, pathogen size from {pathogen_area_before} to {pathogen_area_after}')

        return stack
        
    def normalize_and_outline(image, remove_background, backgrounds, normalize, normalization_percentiles, overlay, overlay_chans, mask_dims, outline_colors, outline_thickness):
        outlines = []
        if remove_background:
            for chan_index, channel in enumerate(range(image.shape[-1])):
                single_channel = stack[:, :, channel]  # Extract the specific channel
                background = backgrounds[chan_index]
                single_channel[single_channel < background] = 0
                image[:, :, channel] = single_channel
        if normalize:
            image = normalize_to_dtype(array=image, q1=normalization_percentiles[0], q2=normalization_percentiles[1])
        rgb_image = np.take(image, overlay_chans, axis=-1)
        rgb_image = rgb_image.astype(float)
        rgb_image -= rgb_image.min()
        rgb_image /= rgb_image.max()
        if overlay:
            overlayed_image = rgb_image.copy()
            for i, mask_dim in enumerate(mask_dims):
                mask = np.take(stack, mask_dim, axis=2)
                outline = np.zeros_like(mask)
                # Find the contours of the objects in the mask
                for j in np.unique(mask)[1:]:
                    contours = find_contours(mask == j, 0.5)
                    for contour in contours:
                        contour = contour.astype(int)
                        outline[contour[:, 0], contour[:, 1]] = j
                # Make the outline thicker
                outline = dilation(outline, square(outline_thickness))
                outlines.append(outline)
                # Overlay the outlines onto the RGB image
                for j in np.unique(outline)[1:]:
                    overlayed_image[outline == j] = outline_colors[i % len(outline_colors)]
            return overlayed_image, image, outlines
        else:
            return [], image, []
        
    def plot_merged_plot(overlay, image, stack, mask_dims, figuresize, overlayed_image, outlines, cmap, outline_colors, print_object_number):
        if overlay:
            fig, ax = plt.subplots(1, image.shape[-1] + len(mask_dims) + 1, figsize=(4 * figuresize, figuresize))
            ax[0].imshow(overlayed_image)
            ax[0].set_title('Overlayed Image')
            ax_index = 1
        else:
            fig, ax = plt.subplots(1, image.shape[-1] + len(mask_dims), figsize=(4 * figuresize, figuresize))
            ax_index = 0

        # Normalize and plot each channel with outlines
        for v in range(0, image.shape[-1]):
            channel_image = image[..., v]
            channel_image_normalized = channel_image.astype(float)
            channel_image_normalized -= channel_image_normalized.min()
            channel_image_normalized /= channel_image_normalized.max()
            channel_image_rgb = np.dstack((channel_image_normalized, channel_image_normalized, channel_image_normalized))

            # Apply the outlines onto the RGB image
            for outline, color in zip(outlines, outline_colors):
                for j in np.unique(outline)[1:]:
                    channel_image_rgb[outline == j] = mpl.colors.to_rgb(color)

            ax[v + ax_index].imshow(channel_image_rgb)
            ax[v + ax_index].set_title('Image - Channel'+str(v))

        for i, mask_dim in enumerate(mask_dims):
            mask = np.take(stack, mask_dim, axis=2)
            random_cmap = generate_mask_random_cmap(mask)
            ax[i + image.shape[-1] + ax_index].imshow(mask, cmap=random_cmap)
            ax[i + image.shape[-1] + ax_index].set_title('Mask '+ str(i))
            if print_object_number:
                unique_objects = np.unique(mask)[1:]
                for obj in unique_objects:
                    cy, cx = ndi.center_of_mass(mask == obj)
                    ax[i + image.shape[-1] + ax_index].text(cx, cy, str(obj), color='white', fontsize=font, ha='center', va='center')

        plt.tight_layout()
        plt.show()
        return fig
    
    font = settings['figuresize']/2
    outline_colors = get_colours_merged(settings['outline_color'])
    index = 0
    
    mask_dims = [settings['cell_mask_dim'], settings['nucleus_mask_dim'], settings['pathogen_mask_dim']]
    mask_dims = [element for element in mask_dims if element is not None]
    
    if settings['verbose']:
        display(settings)

    for file in os.listdir(src):
        path = os.path.join(src, file)
        stack = np.load(path)
        print(f'Loaded: {path}')
        if not settings['include_noninfected']:
            if settings['pathogen_mask_dim'] is not None and settings['cell_mask_dim'] is not None:
                stack = remove_noninfected(stack, settings['cell_mask_dim'], settings['nucleus_mask_dim'], settings['pathogen_mask_dim'])

        if settings['include_multiinfected'] is not None or settings['include_multinucleated'] is not None or settings['filter_min_max'] is not None:
            stack = filter_objects_in_plot(stack, settings['cell_mask_dim'], settings['nucleus_mask_dim'], settings['pathogen_mask_dim'], mask_dims, settings['filter_min_max'], settings['include_multinucleated'], settings['include_multiinfected'])

        image = np.take(stack, settings['channel_dims'], axis=2)

        overlayed_image, image, outlines = normalize_and_outline(image, settings['remove_background'], settings['backgrounds'], settings['normalize'], settings['normalization_percentiles'], settings['overlay'], settings['overlay_chans'], mask_dims, outline_colors, settings['outline_thickness'])
        
        if index < settings['nr']:
            index += 1
            fig = plot_merged_plot(settings['overlay'], image, stack, mask_dims, settings['figuresize'], overlayed_image, outlines, settings['cmap'], outline_colors, settings['print_object_number'])
        else:
            return
def save_filtered_cells_to_csv(src, cell_mask_dim=4, nucleus_mask_dim=5, pathogen_mask_dim=6, include_multinucleated=True, include_multiinfected=True, include_noninfected=False, include_border_pathogens=False, verbose=False):
    mask_dims = [cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim]
    dest = os.path.join(src, 'measurements')
    os.makedirs(dest, exist_ok=True)
    csv_file = dest+'/filtered_filelist.csv'
    mask_dim = cell_dim
    for file in os.listdir(src+'/merged'):
        path = os.path.join(src+'/merged', file)
        stack = np.load(path, allow_pickle=True)
        if not include_noninfected:
            stack = remove_noninfected(stack, cell_dim, nucleus_dim, pathogen_dim)
        if include_multinucleated is not True:
            stack = remove_multiobject_cells(stack, mask_dim, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim, object_dim=pathogen_mask_dim)
        if include_multiinfected is not True:
            stack = remove_multiobject_cells(stack, mask_dim, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim, object_dim=nucleus_mask_dim)
        if include_border_pathogens is not True:
            stack = remove_border_pathogens(stack, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim)
        for i, mask_dim in enumerate(mask_dims):
            mask = np.take(stack, mask_dim, axis=2)
            unique_labels = np.unique(mask)
            with open(csv_file, 'a', newline='') as csvfile:
                fieldnames = ['filename', 'object_label']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for label in unique_labels[1:]:
                    writer.writerow({'filename': file, 'object_label': label})
    return print(f'filtered file list saved at: {csv_file}')

def merge_file(chan_dirs, stack_dir, file):
    chan1 = cv2.imread(str(file), -1)
    chan1 = np.expand_dims(chan1, axis=2)
    new_file = stack_dir / (file.stem + '.npy')
    if not new_file.exists():
        stack_dir.mkdir(exist_ok=True)
        channels = [chan1]
        for chan_dir in chan_dirs[1:]:
            img = cv2.imread(str(chan_dir / file.name), -1)
            chan = np.expand_dims(img, axis=2)
            channels.append(chan)
        stack = np.concatenate(channels, axis=2)
        np.save(new_file, stack)

def is_dir_empty(dir_path):
    return len(os.listdir(dir_path)) == 0

def merge_channels(src, plot=False):
    src = Path(src)
    stack_dir = src / 'stack'
    chan_dirs = [d for d in src.iterdir() if d.is_dir() and d.name in ['01', '02', '03', '04', '00', '1', '2', '3', '4','0']]
    print(chan_dirs)
    chan_dirs.sort(key=lambda x: x.name)
    print(f'List of folders in src: {[d.name for d in chan_dirs]}. Single channel folders.')
    start_time = time.time()

    # First directory and its files
    dir_files = list(chan_dirs[0].iterdir())
    
    # Create the 'stack' directory if it doesn't exist
    stack_dir.mkdir(exist_ok=True)

    if is_dir_empty(stack_dir):
        with Pool(cpu_count()) as pool:
            merge_func = partial(merge_file, chan_dirs, stack_dir)
            pool.map(merge_func, dir_files)

    avg_time = (time.time() - start_time) / len(dir_files)
    print(f'Average Time: {avg_time:.3f} sec')

    if plot:
        plot_arrays(src+'/stack')

    return

def plot_4D_arrays(src, figuresize=10, cmap='inferno', nr_npz=1, nr=1):
    paths = [os.path.join(src, file) for file in os.listdir(src) if file.endswith('.npz')]
    paths = random.sample(paths, min(nr_npz, len(paths)))
    
    for path in paths:
        with np.load(path) as data:
            stack = data['data']
        num_images = stack.shape[0]
        num_channels = stack.shape[3]

        for i in range(min(nr, num_images)):
            img = stack[i]
            
            # Create subplots
            if num_channels == 1:
                fig, axs = plt.subplots(1, 1, figsize=(figuresize, figuresize))
                axs = [axs]  # Make axs a list to use axs[c] later
            else:
                fig, axs = plt.subplots(1, num_channels, figsize=(num_channels * figuresize, figuresize))
            
            for c in range(num_channels):
                axs[c].imshow(img[:, :, c], cmap=cmap)
                axs[c].set_title(f'Channel {c}', size=24)
                axs[c].axis('off')

            fig.tight_layout()
            plt.show()
    return

def plot_stack(stack):
    # Number of images, image width, image height, and number of channels
    num_images, img_width, img_height, num_channels = stack.shape
    print(stack.dtype, stack.shape)
    for i in range(num_images):
        plt.figure(figsize=(4 * num_channels, 4)) # Create a new figure for each image
        for j in range(num_channels):
            img = stack[i, :, :, j]
            plt.subplot(1, num_channels, j+1) # Create a subplot for each channel
            plt.imshow(img, cmap='inferno') # Display the image
            plt.title(f'Image {i+1}, Channel {j+1}') # Set the title
            plt.axis('off')
        plt.tight_layout()
        plt.show() # Display the figure
        
def process_file(args):
    path, channels = args
    array = np.load(path)
    array = np.take(array, channels, axis=2)
    return array, path

#generate npz from png
def png_to_npz(src, randomize=True, batch_size=1000):
    paths = []
    index = 0
    for file in os.listdir(src):
        if file.endswith('.png'):
            path = os.path.join(src, file)
            paths.append(path)
    if randomize:
        random.shuffle(paths)
    stack_ls = []
    nr_files = len(paths)
    channel_stack_loc = src+'_channel_stack'
    os.makedirs(channel_stack_loc, exist_ok=True)
    batch_index = 0  # Added this to name the output files
    filenames_batch = []  # to hold filenames of the current batch
    for i, path in enumerate(paths):
        array = cv2.imread(path)
        stack_ls.append(array)
        filenames_batch.append(os.path.basename(path))  # store the filename
        print(f'Concatenated: {i+1}/{nr_files} files', end='\r', flush=True)

        if (i+1) % batch_size == 0 or i+1 == nr_files:
            stack = np.stack(stack_ls)
            save_loc = os.path.join(channel_stack_loc, f'stack_{batch_index}.npz')
            np.savez(save_loc, data=stack, filenames=filenames_batch)
            batch_index += 1  # increment this after each batch is saved
            del stack  # delete to free memory
            stack_ls = []  # empty the list for the next batch
            filenames_batch = []  # empty the filenames list for the next batch
    print(f'\nAll files concatenated and saved to:{channel_stack_loc}')
    return channel_stack_loc

def generate_time_lists(file_list):
    file_dict = defaultdict(list)
    for filename in file_list:
        if filename.endswith('.npy'):
            parts = filename.split('_')
            plate = parts[0]
            well = parts[1]
            field = parts[2]
            timepoint = parts[3].split('.')[0]
            key = (plate, well, field)
            file_dict[key].append(filename)
    sorted_file_lists = [sorted(files) for files in file_dict.values()]
    return sorted_file_lists

def concatenate_channel(src, channels, randomize=True, timelapse=False, batch_size=100):
    channels = [item for item in channels if item is not None]
    paths = []
    index = 0
    channel_stack_loc = os.path.join(os.path.dirname(src), 'channel_stack')
    os.makedirs(channel_stack_loc, exist_ok=True)
    if timelapse:
        try:
            time_stack_path_lists = generate_time_lists(os.listdir(src))
            for i, time_stack_list in enumerate(time_stack_path_lists):
                stack_region = []
                filenames_region = []
                for idx, file in enumerate(time_stack_list):
                    path = os.path.join(src, file)
                    if idx == 0:
                        parts = file.split('_')
                        name = parts[0]+'_'+parts[1]+'_'+parts[2]
                    array = np.load(path)
                    array = np.take(array, channels, axis=2)
                    stack_region.append(array)
                    filenames_region.append(os.path.basename(path))
                print(f'Region {i+1}/ {len(time_stack_path_lists)}', end='\r', flush=True)
                stack = np.stack(stack_region)
                save_loc = os.path.join(channel_stack_loc, f'{name}.npz')
                print(save_loc)
                np.savez(save_loc, data=stack, filenames=filenames_region)
                del stack
        except Exception as e:
            print(f"Error processing files, make sure filenames metadata is structured plate_well_field_time.npy")
            print(f"Error: {e}")
    else:
        for file in os.listdir(src):
            if file.endswith('.npy'):
                path = os.path.join(src, file)
                paths.append(path)
        if randomize:
            random.shuffle(paths)
        nr_files = len(paths)
        batch_index = 0  # Added this to name the output files
        stack_ls = []
        filenames_batch = []  # to hold filenames of the current batch
        for i, path in enumerate(paths):
            array = np.load(path)
            array = np.take(array, channels, axis=2)
            stack_ls.append(array)
            filenames_batch.append(os.path.basename(path))  # store the filename
            print(f'Concatenated: {i+1}/{nr_files} files', end='\r', flush=True)

            if (i+1) % batch_size == 0 or i+1 == nr_files:
                unique_shapes = {arr.shape[:-1] for arr in stack_ls}
                if len(unique_shapes) > 1:
                    max_dims = np.max(np.array(list(unique_shapes)), axis=0)
                    print(f'Warning: arrays with multiple shapes found in batch {i+1}. Padding arrays to max X,Y dimentions {max_dims}', end='\r', flush=True)
                    padded_stack_ls = []
                    for arr in stack_ls:
                        pad_width = [(0, max_dim - dim) for max_dim, dim in zip(max_dims, arr.shape[:-1])]
                        pad_width.append((0, 0))
                        padded_arr = np.pad(arr, pad_width)
                        padded_stack_ls.append(padded_arr)
                    stack = np.stack(padded_stack_ls)
                else:
                    stack = np.stack(stack_ls)
                save_loc = os.path.join(channel_stack_loc, f'stack_{batch_index}.npz')
                np.savez(save_loc, data=stack, filenames=filenames_batch)
                batch_index += 1  # increment this after each batch is saved
                del stack  # delete to free memory
                stack_ls = []  # empty the list for the next batch
                filenames_batch = []  # empty the filenames list for the next batch
                padded_stack_ls = []
    print(f'\nAll files concatenated and saved to:{channel_stack_loc}')
    return channel_stack_loc
    
def get_lists_for_normalization(settings):

    # Initialize the lists
    backgrounds = []
    signal_to_noise = []
    signal_thresholds = [] 

    # Iterate through the channels and append the corresponding values if the channel is not None
    for ch in settings['channels']:
        if ch == settings['nucleus_channel']:
            backgrounds.append(settings['nucleus_background'])
            signal_to_noise.append(settings['nucleus_Signal_to_noise'])
            signal_thresholds.append(settings['nucleus_Signal_to_noise']*settings['nucleus_background'])
        elif ch == settings['cell_channel']:
            backgrounds.append(settings['cell_background'])
            signal_to_noise.append(settings['cell_Signal_to_noise'])
            signal_thresholds.append(settings['cell_Signal_to_noise']*settings['cell_background'])
        elif ch == settings['pathogen_channel']:
            backgrounds.append(settings['pathogen_background'])
            signal_to_noise.append(settings['pathogen_Signal_to_noise'])
            signal_thresholds.append(settings['pathogen_Signal_to_noise']*settings['pathogen_background'])
    return backgrounds, signal_to_noise, signal_thresholds
    
def normalize_timelapse(src, lower_quantile=0.01, save_dtype=np.float32):
    paths = [os.path.join(src, file) for file in os.listdir(src) if file.endswith('.npz')]
    output_fldr = os.path.join(os.path.dirname(src), 'norm_channel_stack')
    os.makedirs(output_fldr, exist_ok=True)

    for file_index, path in enumerate(paths):
        with np.load(path) as data:
            stack = data['data']
            filenames = data['filenames']

        normalized_stack = np.zeros_like(stack, dtype=save_dtype)
        file = os.path.basename(path)
        name, _ = os.path.splitext(file)

        for chan_index in range(stack.shape[-1]):
            single_channel = stack[:, :, :, chan_index]
            first_image = single_channel[0]

            global_lower = np.quantile(first_image[first_image != 0], lower_quantile)
            global_upper = np.quantile(first_image[first_image != 0], 0.98)

            for array_index in range(single_channel.shape[0]):
                arr_2d = single_channel[array_index]
                arr_2d_rescaled = exposure.rescale_intensity(arr_2d, in_range=(global_lower, global_upper), out_range='dtype')
                normalized_stack[array_index, :, :, chan_index] = arr_2d_rescaled

                print(f'Progress: files {file_index+1}/{len(paths)}, channels:{chan_index+1}/{stack.shape[-1]}, arrays:{array_index+1}/{single_channel.shape[0]}', end='\r')

        save_loc = os.path.join(output_fldr, f'{name}_norm_timelapse.npz')
        np.savez(save_loc, data=normalized_stack, filenames=filenames)

        del normalized_stack, stack, filenames
        gc.collect()

    print(f'\nSaved normalized stacks in: {output_fldr}')

def normalize_stack(src, backgrounds=[100,100,100], remove_background=False, lower_quantile=0.01, save_dtype=np.float32, signal_to_noise=[5,5,5], signal_thresholds=[1000,1000,1000], correct_illumination=False):

    paths = [os.path.join(src, file) for file in os.listdir(src) if file.endswith('.npz')]
    output_fldr = os.path.join(os.path.dirname(src), 'norm_channel_stack')
    os.makedirs(output_fldr, exist_ok=True)
    time_ls = []
    for file_index, path in enumerate(paths):
        with np.load(path) as data:
            stack = data['data']
            filenames = data['filenames']
        normalized_stack = np.zeros_like(stack, dtype=stack.dtype)
        file = os.path.basename(path)
        name, _ = os.path.splitext(file)
        
        for chan_index, channel in enumerate(range(stack.shape[-1])):
            single_channel = stack[:, :, :, channel]
            background = backgrounds[chan_index]
            signal_threshold = signal_thresholds[chan_index]
            #print(f'signal_threshold:{signal_threshold} in {signal_thresholds} for {chan_index}')
            
            signal_2_noise = signal_to_noise[chan_index]
            if remove_background:
                single_channel[single_channel < background] = 0
            if correct_illumination:
                bg = filters.gaussian(single_channel, sigma=50)
                single_channel = single_channel - bg
            
            #Calculate the global lower and upper quantiles for non-zero pixels
            non_zero_single_channel = single_channel[single_channel != 0]
            global_lower = np.quantile(non_zero_single_channel, lower_quantile)
            for upper_p in np.linspace(0.98, 1.0, num=100).tolist():
                global_upper = np.quantile(non_zero_single_channel, upper_p)
                if global_upper >= signal_threshold:
                    break
            
            #Normalize the pixels in each image to the global quantiles and then dtype.
            arr_2d_normalized = np.zeros_like(single_channel, dtype=single_channel.dtype)
            signal_to_noise_ratio_ls = []
            for array_index in range(single_channel.shape[0]):
                start = time.time()
                arr_2d = single_channel[array_index, :, :]
                non_zero_arr_2d = arr_2d[arr_2d != 0]
                if non_zero_arr_2d.size > 0:
                    lower, upper = np.quantile(non_zero_arr_2d, (lower_quantile, upper_p))
                    signal_to_noise_ratio = upper/lower
                else:
                    signal_to_noise_ratio = 0
                signal_to_noise_ratio_ls.append(signal_to_noise_ratio)
                average_stnr = np.mean(signal_to_noise_ratio_ls) if len(signal_to_noise_ratio_ls) > 0 else 0
                
                #mask_channels = [nucleus_chann_dim, pathogen_chann_dim, cell_chann_dim]
                #mask_channels = [0,2,3]
                
                if signal_to_noise_ratio > signal_2_noise:
                    arr_2d_rescaled = exposure.rescale_intensity(arr_2d, in_range=(lower, upper), out_range=(global_lower, global_upper))
                    arr_2d_normalized[array_index, :, :] = arr_2d_rescaled
                else:
                    arr_2d_normalized[array_index, :, :] = arr_2d
                stop = time.time()
                duration = (stop - start)*single_channel.shape[0]
                time_ls.append(duration)
                average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
                print(f'Progress: files {file_index+1}/{len(paths)}, channels:{chan_index}/{stack.shape[-1]-1}, arrays:{array_index+1}/{single_channel.shape[0]}, Signal:{upper:.1f}, noise:{lower:.1f}, Signal-to-noise:{average_stnr:.1f}, Time/channel:{average_time:.2f}sec', end='\r', flush=True)
            normalized_single_channel = exposure.rescale_intensity(arr_2d_normalized, out_range='dtype')
            normalized_stack[:, :, :, channel] = normalized_single_channel
        save_loc = output_fldr+'/'+name+'_norm_stack.npz'
        normalized_stack = normalized_stack.astype(save_dtype)
        np.savez(save_loc, data=normalized_stack, filenames=filenames)
        del normalized_stack, single_channel, normalized_single_channel, stack, filenames
        gc.collect()
    return print(f'Saved stacks:{output_fldr}')

def merge_touching_objects(mask, threshold=0.25):
    perimeters = {}
    labels = np.unique(mask)
    # Calculating perimeter of each object
    for label in labels:
        if label != 0:  # Ignore background
            edges = morphology.erosion(mask == label) ^ (mask == label)
            perimeters[label] = np.sum(edges)
    # Detect touching objects and find the shared boundary
    shared_perimeters = {}
    dilated = morphology.dilation(mask > 0)
    for label in labels:
        if label != 0:  # Ignore background
            # Find the objects that this object is touching
            dilated_label = morphology.dilation(mask == label)
            touching_labels = np.unique(mask[dilated & (dilated_label != 0) & (mask != 0)])
            for touching_label in touching_labels:
                if touching_label != label:  # Exclude the object itself
                    shared_boundary = dilated_label & morphology.dilation(mask == touching_label)
                    shared_perimeters[(label, touching_label)] = np.sum(shared_boundary)
    # Merge objects if more than 25% of their boundary is touching
    for (label1, label2), shared_perimeter in shared_perimeters.items():
        if shared_perimeter > threshold * min(perimeters[label1], perimeters[label2]):
            mask[mask == label2] = label1  # Merge label2 into label1
    return mask

def check_masks(batch, batch_filenames, output_folder):
    # Create a mask for filenames that are already present in the output folder
    existing_files_mask = [not os.path.isfile(os.path.join(output_folder, filename)) for filename in batch_filenames]

    # Use the mask to filter the batch and batch_filenames
    filtered_batch = [b for b, exists in zip(batch, existing_files_mask) if exists]
    filtered_filenames = [f for f, exists in zip(batch_filenames, existing_files_mask) if exists]

    return np.array(filtered_batch), filtered_filenames

def count_objects_in_mask(mask, filename, object_type):
    unique_values, counts = np.unique(mask, return_counts=True)
    df = pd.DataFrame({'object_type':object_type, 'filename':filename, 'object_number': unique_values, 'Size': counts})
    return df

def measure_eccentricity_and_intensity(mask, image):
    labels = np.unique(mask) #[1:], 0 label is typically the background
    properties = regionprops_table(mask, intensity_image=image, properties=('label', 'area', 'eccentricity', 'mean_intensity'))
    df = pd.DataFrame(properties)
    return df

def timelapse_segmentation(batch, output_folder, path, chans, model, diameter, interpolate=True):

    def calculate_region_props(masks):
        all_props = []
        for i in range(masks.shape[0]):
            print(f'Processing props for timepoint {i}', end='\r', flush=True)
            labeled_mask = label(masks[i], connectivity=1)
            props = regionprops(labeled_mask)
            all_props.append([{
                'centroid': prop.centroid, 
                'area': prop.area, 
                'label': prop.label,
                'eccentricity': prop.eccentricity,
                'orientation': prop.orientation,
                'perimeter': prop.perimeter
            } for prop in props])
        return all_props

    def match_objects_v1(all_props):
        object_mappings = [{} for _ in range(len(all_props))]
        current_id = 1  # Start object IDs from 1

        # Flattening the list of all properties for normalization
        flat_props = [item for sublist in all_props for item in sublist]
        centroids = np.array([p['centroid'] for p in flat_props])
        areas = np.array([p['area'] for p in flat_props])
        eccentricities = np.array([p['eccentricity'] for p in flat_props])
        perimeters = np.array([p['perimeter'] for p in flat_props])
        orientations = np.array([p['orientation'] for p in flat_props])

        # Standardizing properties
        centroid_zs = zscore(centroids, axis=0)
        area_zs = zscore(areas)
        eccentricity_zs = zscore(eccentricities)
        perimeter_zs = zscore(perimeters)
        orientation_zs = zscore(orientations)

        # Weights for each property
        weights = {'centroid': 5, 'area': 3, 'eccentricity': 2, 'perimeter': 2, 'orientation': 1}

        for i in range(1, len(all_props)):
            print(f'Mapping props for timepoint {i}', end='\r', flush=True)
            prev_frame, current_frame = all_props[i-1], all_props[i]
            composite_distances = []

            for prev_obj in prev_frame:
                distances = []
                for curr_obj in current_frame:
                    # Calculate weighted distance for each property, using standardized values
                    distance = 0
                    distance += weights['centroid'] * np.linalg.norm(centroid_zs[flat_props.index(prev_obj)] - centroid_zs[flat_props.index(curr_obj)])
                    distance += weights['area'] * abs(area_zs[flat_props.index(prev_obj)] - area_zs[flat_props.index(curr_obj)])
                    distance += weights['eccentricity'] * abs(eccentricity_zs[flat_props.index(prev_obj)] - eccentricity_zs[flat_props.index(curr_obj)])
                    distance += weights['perimeter'] * abs(perimeter_zs[flat_props.index(prev_obj)] - perimeter_zs[flat_props.index(curr_obj)])
                    distance += weights['orientation'] * abs(orientation_zs[flat_props.index(prev_obj)] - orientation_zs[flat_props.index(curr_obj)])
                    distances.append(distance)
                composite_distances.append(distances)
            # Find the best match for each object in the previous frame
            for prev_index, distances in enumerate(composite_distances):
                current_index = np.argmin(distances)
                prev_label = prev_frame[prev_index]['label']
                curr_label = current_frame[current_index]['label']
                object_mappings[i][curr_label] = object_mappings[i-1].get(prev_label, current_id)
                if curr_label not in object_mappings[i-1].values():
                    current_id += 1
        return object_mappings
        
    def match_objects(all_props):
        object_mappings = [{} for _ in range(len(all_props))]
        current_id = 1
        some_threshold = 0.1

        # Pre-compute all properties in a vectorized form for efficiency
        all_centroids = [np.array([prop['centroid'] for prop in frame_props]) for frame_props in all_props]
        all_areas = [np.array([prop['area'] for prop in frame_props]) for frame_props in all_props]
        all_eccentricities = [np.array([prop['eccentricity'] for prop in frame_props]) for frame_props in all_props]
        all_perimeters = [np.array([prop['perimeter'] for prop in frame_props]) for frame_props in all_props]
        all_orientations = [np.array([prop['orientation'] for prop in frame_props]) for frame_props in all_props]

        # Weights for each property
        weights = np.array([5, 3, 2, 2, 1])  # centroid, area, eccentricity, perimeter, orientation

        for i in range(1, len(all_props)):
            if len(all_props[i-1]) == 0 or len(all_props[i]) == 0:
                continue  # Skip if no objects to match

            # Standardize properties for the current and previous frame
            centroids_zs = zscore(np.vstack((all_centroids[i-1], all_centroids[i])), axis=0)
            areas_zs = zscore(np.hstack((all_areas[i-1], all_areas[i])))
            eccentricities_zs = zscore(np.hstack((all_eccentricities[i-1], all_eccentricities[i])))
            perimeters_zs = zscore(np.hstack((all_perimeters[i-1], all_perimeters[i])))
            orientations_zs = zscore(np.hstack((all_orientations[i-1], all_orientations[i])))

            # Calculate distances between all pairs using standardized properties
            dist_matrix = np.zeros((len(all_props[i-1]), len(all_props[i])))

            # Calculate Euclidean distances for centroids separately due to being multi-dimensional
            centroid_distances = cdist(centroids_zs[:len(all_props[i-1])], centroids_zs[len(all_props[i-1]):], 'euclidean')
            dist_matrix += weights[0] * centroid_distances

            # Calculate absolute differences for other properties
            for prop_index, prev_frame_props in enumerate([areas_zs, eccentricities_zs, perimeters_zs, orientations_zs], start=1):
                dist_matrix += weights[prop_index] * cdist(prev_frame_props[:len(all_props[i-1])][:, np.newaxis], 
                                                           prev_frame_props[len(all_props[i-1]):][:, np.newaxis], 
                                                           'cityblock')

            # Match objects based on minimum distance
            for prev_index in range(dist_matrix.shape[0]):
                current_index = np.argmin(dist_matrix[prev_index])
                min_distance = dist_matrix[prev_index, current_index]

                # This is a simplified example; additional checks might be necessary
                if min_distance < some_threshold:
                    prev_label = all_props[i-1][prev_index]['label']
                    curr_label = all_props[i][current_index]['label']
                    object_mappings[i][curr_label] = object_mappings[i-1].get(prev_label, current_id)
                    if curr_label not in object_mappings[i-1].values():
                        current_id += 1

        return object_mappings

    def relabel_masks(masks, object_mappings):
        relabeled_masks = np.zeros_like(masks)
        for i, mapping in enumerate(object_mappings):
            print(f'Relabeling masks for timepoint {i}', end='\r', flush=True)
            for props in calculate_region_props([masks[i]])[0]:
                if props['label'] in mapping:
                    relabeled_masks[i][masks[i] == props['label']] = mapping[props['label']]
        return relabeled_masks

    def filter_objects(masks):
        unique_labels_per_frame = [np.unique(masks[frame])[1:] for frame in range(masks.shape[0])]  # Exclude background (0)
        consistent_objects = set(unique_labels_per_frame[0])
        for labels in unique_labels_per_frame[1:]:
            consistent_objects.intersection_update(labels)

        filtered_masks = np.zeros_like(masks)
        for obj in consistent_objects:
            filtered_masks[masks == obj] = obj
        return filtered_masks

    def interpolate_missing_objects(masks, object_mappings):
        # Initialize an array to hold the interpolated masks
        interpolated_masks = np.copy(masks)
        for frame in range(1, len(masks) - 1):
            print(f'Interpolating masks for timepoint {i}', end='\r', flush=True)
            for obj_id, mapping in object_mappings[frame].items():
                if obj_id not in object_mappings[frame - 1] or obj_id not in object_mappings[frame + 1]:
                    continue  # Skip if the object is not missing
                # Find masks for the object in the previous and next frames
                prev_mask = (interpolated_masks[frame - 1] == mapping)
                next_mask = (interpolated_masks[frame + 1] == mapping)
                # Calculate the overlapping region
                overlap_mask = prev_mask & next_mask
                # Check if there's an actual overlap to avoid creating objects from thin air
                if np.any(overlap_mask):
                    interpolated_masks[frame][overlap_mask] = mapping
        return interpolated_masks
        
    def save_timelapse_masks(mask_stack, output_folder, folder, path):
        # Ensure the target directory exists
        target_dir = os.path.join(output_folder, folder)
        os.makedirs(target_dir, exist_ok=True)

        # Extract the base name of the original file and prepare the new file name
        file_name = os.path.basename(path)
        name, ext = os.path.splitext(file_name)
        new_name = f"{name}.tif"
        new_path = os.path.join(target_dir, new_name)

        # Save the entire mask stack as a single multi-page (4D) TIFF file
        tifffile.imwrite(new_path, mask_stack, imagej=True)

        print(f"4D masks saved in {new_path}")
        
    def plot_timelapse(data, frame_index=0, cmap='gray'):
        if data.ndim == 4:
            frame_data = data[frame_index]
        else:
            frame_data = data

        if frame_data.ndim == 2:  # Single mask or grayscale image
            plt.figure(figsize=(10, 10))
            plt.imshow(frame_data, cmap=cmap)
            plt.axis('off')
            plt.show()
        elif frame_data.ndim == 3:
            channels = frame_data.shape[-1]
            if channels in [3, 4]:  # RGB or RGBA image
                plt.figure(figsize=(10, 10))
                plt.imshow(frame_data)
                plt.axis('off')
                plt.show()
            else:
                # Adjust the number of subplots based on the number of channels
                cols = min(channels, 4)  # Limit the number of columns to 4 for readability
                rows = np.ceil(channels / cols).astype(int)  # Calculate required rows
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
                if channels == 1:
                    axes = [axes]  # Make axes iterable if only one subplot
                for i, ax in enumerate(axes.flat):
                    if i < channels:
                        ax.imshow(frame_data[:, :, i], cmap=cmap)
                        ax.set_title(f'Channel {i}')
                    ax.axis('off')
                plt.tight_layout()
                plt.show()
                
    def create_interactive_plot(data, cmap='gray'):
        max_index = data.shape[0] - 1 if data.ndim == 4 else 0
        interact(lambda frame_index: plot_timelapse(data, frame_index, cmap),frame_index=IntSlider(min=0, max=max_index, step=1, value=0))
        
    #create_interactive_plot(batch, cmap='inferno')

    masks, flows, styles, diams = model.eval(batch,
                                             diameter=diameter,
                                             channels=chans,
                                             do_3D=True,
                                             anisotropy=None)
                                             
    save_timelapse_masks(mask_stack=masks, output_folder=output_folder, folder='CP_masks', path=path)
    
    create_interactive_plot(masks, cmap='gray')

    props = calculate_region_props(masks)
    mapping = match_objects(props)
    masks = relabel_masks(masks, mapping)
    
    save_timelapse_masks(mask_stack=masks, output_folder=output_folder, folder='relabled', path=path)

    if interpolate:
        masks = interpolate_missing_objects(masks, batch.shape[0])
        save_timelapse_masks(mask_stack=masks, output_folder=output_folder, folder='interpolated', path=path)

    masks_filtered = filter_objects(masks)
    save_timelapse_masks(mask_stack=masks, output_folder=output_folder, folder='filtered', path=path)

    return masks_filtered

def plot_masks_to_image(src, fov='', figuresize=20, cmap='gist_ncar', print_object_number=True):
    def fig2img(fig):
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        return np.asarray(buf)
    image_list = []
    mask_files = [f for f in os.listdir(src) if f.endswith('.npy') and f.startswith(fov)]
    # find the maximum label value across all masks
    max_label = max(np.max(np.load(os.path.join(src, mask_file))) for mask_file in mask_files)
    for mask_file in mask_files:
        mask = np.load(os.path.join(src, mask_file))
        fig, ax = plt.subplots(figsize=(figuresize, figuresize))
        # use vmin and vmax to make the colormap consistent across images
        ax.imshow(mask, cmap='gist_ncar', vmin=0, vmax=max_label)
        ax.set_title('Mask')
        if print_object_number:
            unique_objects = np.unique(mask)[1:]
            for obj in unique_objects:
                cy, cx = ndi.center_of_mass(mask == obj)
                ax.text(cx, cy, str(obj), color='black', fontsize=figuresize)
        image_array = fig2img(fig)
        image_list.append(image_array)
        plt.close(fig)
    return image_list

def concatenate_images_to_movie(dst, image_list, name, fps=1):
    output_movie = os.path.join(dst, name+'_mask_stack.mp4')
    os.makedirs(dst, exist_ok=True)
    clip = mpy.ImageSequenceClip(image_list, fps=fps)
    clip.write_videofile(output_movie, logger=None)  # ProgressbarLogger, PrintLogger, None
    print(f'Progress: Saved movie')

def remove_intensity_objects(image, mask, intensity_threshold, mode):
    # Calculate the mean intensity of each object in the original image
    props = regionprops_table(mask, image, properties=('label', 'mean_intensity'))
    # Find the labels of the objects with mean intensity below the threshold
    if mode == 'low':
        labels_to_remove = props['label'][props['mean_intensity'] < intensity_threshold]
    if mode == 'high':
        labels_to_remove = props['label'][props['mean_intensity'] > intensity_threshold]
    # Remove these objects from the mask
    mask[np.isin(mask, labels_to_remove)] = 0
    return mask

def get_avg_object_size(masks):
    object_areas = []
    for mask in masks:
        # Check if the mask is a 2D or 3D array and is not empty
        if mask.ndim in [2, 3] and np.any(mask):
            properties = measure.regionprops(mask)
            object_areas += [prop.area for prop in properties]
        else:
            if not np.any(mask):
                print(f"Mask is empty. ")
            if not mask.ndim in [2, 3]:
                print(f"Mask is not in the correct format. dim: {mask.ndim}")
            continue

    if object_areas:
        return sum(object_areas) / len(object_areas)
    else:
        return 0  # Return 0 if no objects are found

def mask_object_count(mask):
    unique_labels = np.unique(mask)
    num_objects = len(unique_labels[unique_labels!=0])
    return num_objects

def split_objects_by_intensity(flow_output, img):
    output_labels = np.zeros_like(flow_output)
    for cell_label in np.unique(flow_output):
        if cell_label == 0:  # Skip background
            continue
        cell_mask = (flow_output == cell_label)
        distance = distance_transform_edt(img * cell_mask)
        local_maxi = peak_local_max(distance, indices=False, labels=cell_mask)
        markers = label(local_maxi)
        cell_labels = watershed(-distance, markers, mask=cell_mask)
        output_labels += cell_labels + (output_labels.max() if output_labels.size > 0 else 0)
    return output_labels

def plot_masks(batch, masks, flows, cmap='inferno', figuresize=20, nr=1, file_type='.npz', print_object_number=True):
    if len(batch.shape) == 3:
        batch = np.expand_dims(batch, axis=0)
    if not isinstance(masks, list):
        masks = [masks]
    if not isinstance(flows, list):
        flows = [flows]
    else:
        flows = flows[0]
    if file_type == 'png':
        flows = [f[0] for f in flows]  # assuming this is what you want to do when file_type is 'png'
    font = figuresize/2
    index = 0
    for image, mask, flow in zip(batch, masks, flows):
        unique_labels = np.unique(mask)
        
        num_objects = len(unique_labels[unique_labels != 0])
        random_colors = np.random.rand(num_objects+1, 4)
        random_colors[:, 3] = 1
        random_colors[0, :] = [0, 0, 0, 1]
        random_cmap = mpl.colors.ListedColormap(random_colors)
        
        if index < nr:
            index += 1
            chans = image.shape[-1]
            fig, ax = plt.subplots(1, image.shape[-1] + 2, figsize=(4 * figuresize, figuresize))
            for v in range(0, image.shape[-1]):
                ax[v].imshow(image[..., v], cmap=cmap)  # display first channel
                ax[v].set_title('Image - Channel'+str(v))
            ax[chans].imshow(mask, cmap=random_cmap)
            ax[chans].set_title('Mask')
            if print_object_number:
                unique_objects = np.unique(mask)[1:]
                for obj in unique_objects:
                    cy, cx = ndi.center_of_mass(mask == obj)
                    ax[chans].text(cx, cy, str(obj), color='white', fontsize=font, ha='center', va='center')
            ax[chans+1].imshow(flow, cmap='viridis')
            ax[chans+1].set_title('Flow')
            plt.show()
    return

def filter_cp_masks(masks, flows, refine_masks, filter_size, minimum_size, maximum_size, remove_border_objects, merge, filter_dimm, batch, moving_avg_q1, moving_avg_q3, moving_count, plot, figuresize, split_objects=False):
    mask_stack = []
    for idx, (mask, flow, image) in enumerate(zip(masks, flows[0], batch)):
        if plot:
            num_objects = mask_object_count(mask)
            print(f'Number of objects before filtration: {num_objects}')
            #plot_mask_and_flow(mask, flow, figuresize)
            plot_masks(batch=image, masks=mask, flows=flow, cmap='inferno', figuresize=figuresize, nr=1, file_type='.npz', print_object_number=True)
        
        if filter_size:
            props = measure.regionprops_table(mask, properties=['label', 'area'])  # Measure properties of labeled image regions.
            valid_labels = props['label'][np.logical_and(props['area'] > minimum_size, props['area'] < maximum_size)]  # Select labels of valid size.
            masks[idx] = np.isin(mask, valid_labels) * mask  # Keep only valid objects.
            if plot:
                num_objects = mask_object_count(mask)
                print(f'Number of objects after size filtration >{minimum_size} and <{maximum_size} : {num_objects}')
                plot_masks(batch=image, masks=mask, flows=flow, cmap='inferno', figuresize=figuresize, nr=1, file_type='.npz', print_object_number=True)
        if remove_border_objects:
            mask = clear_border(mask)
            if plot:
                num_objects = mask_object_count(mask)
                print(f'Number of objects after removing border objects, : {num_objects}')
                plot_masks(batch=image, masks=mask, flows=flow, cmap='inferno', figuresize=figuresize, nr=1, file_type='.npz', print_object_number=True)
        if merge:
            mask = merge_touching_objects(mask, threshold=0.25)
            if plot:
                num_objects = mask_object_count(mask)
                print(f'Number of objects after merging adjacent objects, : {num_objects}')
                plot_masks(batch=image, masks=mask, flows=flow, cmap='inferno', figuresize=figuresize, nr=1, file_type='.npz', print_object_number=True)
        if filter_dimm:
            unique_labels = np.unique(mask)
            if len(unique_labels) == 1 and unique_labels[0] == 0:
                continue
            object_intensities = [np.mean(batch[idx, :, :, 1][mask == label]) for label in unique_labels if label != 0]
            object_q1s = [np.percentile(intensities, 25) for intensities in object_intensities if intensities.size > 0]
            object_q3s = [np.percentile(intensities, 75) for intensities in object_intensities if intensities.size > 0]
            if object_q1s:
                object_q1_mean = np.mean(object_q1s)
                object_q3_mean = np.mean(object_q3s)
                moving_avg_q1 = (moving_avg_q1 * moving_count + object_q1_mean) / (moving_count + 1)
                moving_avg_q3 = (moving_avg_q3 * moving_count + object_q3_mean) / (moving_count + 1)
                moving_count += 1
            mask = remove_intensity_objects(batch[idx, :, :, 1], mask, intensity_threshold=moving_avg_q1, mode='low')
            mask = remove_intensity_objects(batch[idx, :, :, 1], mask, intensity_threshold=moving_avg_q3, mode='high')
            if plot:
                num_objects = mask_object_count(mask)
                print(f'Objects after intensity filtration > {moving_avg_q1} and <{moving_avg_q3}: {num_objects}')
                plot_masks(batch=image, masks=mask, flows=flow, cmap='inferno', figuresize=figuresize, nr=1, file_type='.npz', print_object_number=True)
        mask_stack.append(mask)
    return mask_stack
            
def save_object_counts_to_database(arrays, object_type, file_names, db_path, added_string):
    
    def count_objects(mask):
            """Count unique objects in a mask, assuming 0 is the background."""
            unique, counts = np.unique(mask, return_counts=True)
            # Assuming 0 is the background label, remove it from the count
            if unique[0] == 0:
                return len(unique) - 1
            return len(unique)

    records = []
    for mask, file_name in zip(arrays, file_names):
        object_count = count_objects(mask)
        count_type = f"{object_type}{added_string}"  # Combines object_type with added_string for unique count_type
        
        # Append a tuple of (file_name, count_type, object_count) to the records list
        records.append((file_name, count_type, object_count))
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS object_counts (
        file_name TEXT,
        count_type TEXT,
        object_count INTEGER,
        PRIMARY KEY (file_name, count_type)
    )
    ''')
    
    # Batch insert or update the object counts
    cursor.executemany('''
    INSERT INTO object_counts (file_name, count_type, object_count)
    VALUES (?, ?, ?)
    ON CONFLICT(file_name, count_type) DO UPDATE SET
    object_count = excluded.object_count
    ''', records)
    
    # Commit changes and close the database connection
    conn.commit()
    conn.close()
    
def identify_masks(src, object_type, model_name, batch_size, channels, diameter, minimum_size, maximum_size, flow_threshold=30, cellprob_threshold=1, figuresize=25, cmap='inferno', refine_masks=True, filter_size=True, filter_dimm=True, remove_border_objects=False, verbose=False, plot=False, merge=False, save=True, start_at=0, file_type='.npz', net_avg=True, resample=True, timelapse=False):
    
    #Note add logic that handles batches of size 1 as these will break the code batches must all be > 2 images
    gc.collect()
    print('========== generating masks ==========')
    print('Torch available:', torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.Cellpose(gpu=True, model_type=model_name, net_avg=net_avg, device=device)
    if file_type == '.npz':
        paths = [os.path.join(src, file) for file in os.listdir(src) if file.endswith('.npz')]
    else:
        paths = [os.path.join(src, file) for file in os.listdir(src) if file.endswith('.png')]
        if timelapse:
            print(f'timelaps is only compatible with npz files')
            return

    chans = [2, 1] if model_name == 'cyto2' else [0,0] if model_name == 'nuclei' else [2,0] if model_name == 'cyto' else [2, 0]
    
    if verbose == True:
        print(f'source: {src}')
        print(f'Settings: object_type: {object_type}, minimum_size: {minimum_size}, maximum_size:{maximum_size}, figuresize:{figuresize}, cmap:{cmap}, , net_avg:{net_avg}, resample:{resample}')
        print(f'Cellpose settings: Model: {model_name}, batch_size: {batch_size}, channels: {channels}, cellpose_chans: {chans}, diameter:{diameter}, flow_threshold:{flow_threshold}, cellprob_threshold:{cellprob_threshold}')
        print(f'Bool Settings: verbose:{verbose}, plot:{plot}, merge:{merge}, save:{save}, start_at:{start_at}, file_type:{file_type}, timelapse:{timelapse}')

    count_loc = os.path.dirname(src)+'/measurements/measurements.db'
    os.makedirs(os.path.dirname(src)+'/measurements', exist_ok=True)
    create_database(count_loc)
    
    average_sizes = []
    time_ls = []
    moving_avg_q1 = 0
    moving_avg_q3 = 0
    moving_count = 0
    for file_index, path in enumerate(paths):
        if file_type == '.npz':
            if start_at: 
                print(f'starting at file index:{start_at}')
                if file_index < start_at:
                    continue
            output_folder = os.path.join(os.path.dirname(path), object_type+'_mask_stack')
            os.makedirs(output_folder, exist_ok=True)
            overall_average_size = 0
            with np.load(path) as data:
                stack = data['data']
                filenames = data['filenames']
            if timelapse:
                if len(stack) != batch_size:
                    print(f'Changed batch_size:{batch_size} to {len(stack)}, data length:{len(stack)}')
                    batch_size = len(stack)

            for i in range(0, stack.shape[0], batch_size):
                mask_stack = []
                start = time.time()

                if stack.shape[3] == 1:
                    batch = stack[i: i+batch_size, :, :, [0,0]].astype(stack.dtype)
                else:
                    batch = stack[i: i+batch_size, :, :, channels].astype(stack.dtype)
                    
                batch_filenames = filenames[i: i+batch_size].tolist()
                
                if not plot:
                    batch, batch_filenames = check_masks(batch, batch_filenames, output_folder)
                if batch.size == 0:
                    print(f'Processing {file_index}/{len(paths)}: Images/Npz {batch.shape[0]}', end='\r', flush=True)
                    continue
                if batch.max() > 1:
                    batch = batch / batch.max()
                if not timelapse:
                    masks, flows, _, _ = model.eval(x=batch,
                                                    batch_size=batch_size,
                                                    normalize=False,
                                                    channels=chans,
                                                    channel_axis=3,
                                                    diameter=diameter,
                                                    flow_threshold=flow_threshold,
                                                    cellprob_threshold=cellprob_threshold,
                                                    rescale=None,
                                                    resample=resample,
                                                    net_avg=net_avg,
                                                    progress=None)
                    print(f'{object_type}, {batch_filenames}, {count_loc}')
                    save_object_counts_to_database(masks, object_type, batch_filenames, count_loc, added_string='_before_filtration')
                    
                    mask_stack = filter_cp_masks(masks, flows, refine_masks, filter_size, minimum_size, maximum_size, remove_border_objects, merge, filter_dimm, batch, moving_avg_q1, moving_avg_q3, moving_count, plot, figuresize)
                    
                    save_object_counts_to_database(mask_stack, object_type, batch_filenames, count_loc, added_string='_after_filtration')
                    
                    if not np.any(mask_stack):
                        average_obj_size = 0
                    else:
                        average_obj_size = get_avg_object_size(mask_stack)

                    average_sizes.append(average_obj_size) 
                    overall_average_size = np.mean(average_sizes) if len(average_sizes) > 0 else 0
                else:
                    #if plot:
                    #    print(f'before relabeling')
                    #    plot_masks(batch, mask_stack, flows, figuresize=figuresize, cmap=cmap, nr=batch_size, file_type='.npz')
                    print(f'========== generating timelapse masks ==========')
                    mask_stack = timelapse_segmentation(batch, output_folder, path, chans, model, diameter)
                    overall_average_size = 0.000
                    #mask_stack = track_objects_over_time(mask_stack, step_size=2, config=config)
                    #if plot:
                    #    print(f'after relabeling')
                    #    plot_masks(batch, mask_stack, flows, figuresize=figuresize, cmap=cmap, nr=batch_size, file_type='.npz')
            
            stop = time.time()
            duration = (stop - start)
            time_ls.append(duration)
            average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
            time_in_min = average_time/60
            time_per_mask = average_time/batch_size
            print(f'Processing {len(paths)}  files with {batch_size} imgs: {(file_index+1)*(batch_size+1)}/{(len(paths))*(batch_size+1)}: Time/batch {time_in_min:.3f} min: Time/mask {time_per_mask:.3f}sec: {object_type} size: {overall_average_size:.3f} px2', end='\r', flush=True)
            if not timelapse:
                if plot:
                    plot_masks(batch, mask_stack, flows, figuresize=figuresize, cmap=cmap, nr=batch_size, file_type='.npz')
        if file_type == '.png':
            stack = cv2.imread(path).astype(np.float32)
            stack = stack[:, :, channels]
            filename = os.path.basename(path)
            start = time.time()
            if stack.max() > 1:
                stack = stack / stack.max()
            mask, flows, _, _ = model.eval(x=stack,
                                            batch_size=batch_size,
                                            normalize=False,
                                            channels=chans,
                                            channel_axis=3,
                                            diameter=diameter,
                                            flow_threshold=flow_threshold,
                                            cellprob_threshold=cellprob_threshold,
                                            rescale=None,
                                            resample=resample,
                                            net_avg=net_avg,
                                            progress=None)
            
            mask_stack = filter_cp_masks(masks, flows, refine_masks, filter_size, minimum_size, maximum_size, remove_border_objects, merge, filter_dimm, batch, moving_avg_q1, moving_avg_q3, moving_count, plot, figuresize)
            average_obj_size = get_avg_object_size(mask_stack)
            average_sizes.append(average_obj_size) # Store the average size
            overall_average_size = np.mean(average_sizes) if len(average_sizes) > 0 else 0 # Calculate the overall average size across all images
            stop = time.time()
            duration = (stop - start)
            time_ls.append(duration)
            average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
            time_in_min = average_time/60
            time_per_mask = average_time/batch_size
            print(f'Processing {len(paths)} images, Completed {(file_index+1)*(batch_size+1)}/{(len(paths))*(batch_size+1)}: Time/batch {time_in_min:.3f} min: Time/mask {time_per_mask:.3f} sec: Average {object_type} size {overall_average_size:.3f}', end='\r', flush=True)
            if not timelapse:
                if plot:
                    plot_masks(stack, mask_stack, flows, figuresize=figuresize, cmap=cmap, nr=batch_size, file_type='.png')
        if save:
            if timelapse:
                fov_ls = []
                for fn in batch_filenames:
                    nme = os.splitext(fn)
                    comp = split.nme('_')
                    fov = comp[0]+'_'+comp[1]+'_'+comp[2]
                    if fov not in fov_ls:
                        fov_ls.append(fov)
                for fov in fov_ls:
                    image_list = plot_masks_to_image(f'{src}/{name}_{fov}_mask_stack', fov=fov)
                    concatenate_images_to_movie(image_list, 'cell', fps=2)
                
            if file_type == '.npz':
                for mask_index, mask in enumerate(mask_stack):
                    output_filename = os.path.join(output_folder, batch_filenames[mask_index])
                    np.save(output_filename, mask)

            else:
                output_filename = os.path.join(output_folder, filename)
                np.save(output_filename, mask)
            mask_stack = []
            batch_filenames = []
        gc.collect()
    return

def normalize_to_dtype(array, q1=2,q2=98, percentiles=None):
    nimg = array.shape[2]
    new_stack = np.empty_like(array)
    for i,v in enumerate(range(nimg)):
        img = np.squeeze(array[:, :, v])
        non_zero_img = img[img > 0]
        if non_zero_img.size > 0: # check if there are non-zero values
            img_min = np.percentile(non_zero_img, q1)  # change percentile from 0.02 to 2
            img_max = np.percentile(non_zero_img, q2)  # change percentile from 0.98 to 98
            img = rescale_intensity(img, in_range=(img_min, img_max), out_range='dtype')
        else:  # if there are no non-zero values, just use the image as it is
            if percentiles==None:
                img_min, img_max = img.min(), img.max()
            else:
                img_min, img_max = percentiles[i]
            img = rescale_intensity(img, in_range=(img_min, img_max), out_range='dtype')
        img = np.expand_dims(img, axis=2)
        new_stack[:, :, v] = img[:, :, 0]
    return new_stack

def get_percentiles(array, q1=2, q2=98):
    nimg = array.shape[2]
    percentiles = []
    for v in range(nimg):
        img = np.squeeze(array[:, :, v])
        non_zero_img = img[img > 0]
        if non_zero_img.size > 0: # check if there are non-zero values
            img_min = np.percentile(non_zero_img, q1)  # change percentile from 0.02 to 2
            img_max = np.percentile(non_zero_img, q2)  # change percentile from 0.98 to 98
            percentiles.append([img_min, img_max])
        else:  # if there are no non-zero values, just use the image as it is
            img_min, img_max = img.min(), img.max()
            percentiles.append([img_min, img_max])
    return percentiles

def crop_center(img, cell_mask, new_width, new_height, normalize=(2,98)):
    # Convert all non-zero values in mask to 1
    cell_mask[cell_mask != 0] = 1
    mask_3d = np.repeat(cell_mask[:, :, np.newaxis], img.shape[2], axis=2).astype(img.dtype) # Create 3D mask
    img = np.multiply(img, mask_3d).astype(img.dtype) # Multiply image with mask to set pixel values outside of the mask to 0
    #centroid = np.round(ndi.measurements.center_of_mass(cell_mask)).astype(int) # Compute centroid of the mask
    centroid = np.round(ndi.center_of_mass(cell_mask)).astype(int) # Compute centroid of the mask
    # Pad the image and mask to ensure the crop will not go out of bounds
    pad_width = max(new_width, new_height)
    img = np.pad(img, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='constant')
    cell_mask = np.pad(cell_mask, ((pad_width, pad_width), (pad_width, pad_width)), mode='constant')
    # Update centroid coordinates due to padding
    centroid += pad_width
    # Compute bounding box
    start_y = max(0, centroid[0] - new_height // 2)
    end_y = min(start_y + new_height, img.shape[0])
    start_x = max(0, centroid[1] - new_width // 2)
    end_x = min(start_x + new_width, img.shape[1])
    # Crop to bounding box
    img = img[start_y:end_y, start_x:end_x, :]
    return img

def add_prefix_to_columns(df, prefix):
    df.columns = [f"{prefix}_{col}" for col in df.columns]
    return df

def plot_cropped_arrays(stack, figuresize=20,cmap='inferno'):
    """Plot arrays"""
    start = time.time()
    dim = stack.shape 
    channel=min(dim)
    if len(stack.shape) == 2:
        f, a = plt.subplots(1, 1,figsize=(figuresize,figuresize))
        a.imshow(stack, cmap=plt.get_cmap(cmap))
        a.set_title('Channel one',size=18)
        a.axis('off')
        f.tight_layout()
        plt.show()
    if len(stack.shape) > 2:
        anr = stack.shape[2]
        f, a = plt.subplots(1, anr,figsize=(figuresize,figuresize))
        for channel in range(anr):
            a[channel].imshow(stack[:,:,channel], cmap=plt.get_cmap(cmap))
            a[channel].set_title('Channel '+str(channel),size=18)
            a[channel].axis('off')
            f.tight_layout()
        plt.show()
    stop = time.time()
    duration = stop - start
    print('plot_cropped_arrays', duration)
    return

def dialate_objects(object_array, iterations=5, ):
    struct = ndi.generate_binary_structure(2, 2) # Define the structure for dilation
    for value in np.unique(object_array): # Iterate through each unique value (object) in the array
        if value == 0: # Assuming 0 is the background, so skip it
            continue
        mask = (object_array == value) # Create a mask for the current object
        dilated = ndi.binary_dilation(mask, structure=struct, iterations=5) # Apply dilation to the object
        object_array[dilated] = value # Update the array with the dilated object
    return object_array

def create_database(db_path):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        #print(f"SQLite version: {sqlite3.version}")
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()

def secondary_object_regionprops(i, name, sec_obj_mask, sec_obj_ids, file_name, prim_obj_id, channel_arrays, properties, quartiles):
    sec_obj_props_list = []
    sec_obj_id_list = []
    for sec_obj_id in sec_obj_ids:
        if sec_obj_id == 0:
            continue
        sec_obj_region = sec_obj_mask == sec_obj_id
        sec_obj_region = label(sec_obj_region)
        chanels = np.split(channel_arrays, 3, axis=-1)
        sec_obj_props = extended_regionprops_table(channel_arrays, sec_obj_region, properties)
        # Append 'prim_obj_id' and 'file_name' directly to 'sec_obj_props'
        sec_obj_props['prim_obj_id'] = prim_obj_id
        sec_obj_props['file_name'] = file_name
        sec_obj_props_prefixed = add_prefix_to_columns(pd.DataFrame(sec_obj_props), name+'_channal_'+str(i))
        sec_obj_props_list.append(sec_obj_props_prefixed)
        sec_obj_id_list.append(sec_obj_id)
    if len(sec_obj_props_list) > 0:
        sec_obj_props_list_combined = pd.concat(sec_obj_props_list)
        sec_obj_props_list_combined[name+'_label'] = sec_obj_id_list
        sec_obj_props_list_combined = sec_obj_props_list_combined.reset_index(drop=True)
    else:
        sec_obj_props_list_combined = pd.DataFrame(columns=add_prefix_to_columns(pd.DataFrame(), name).columns)

    return sec_obj_props_list_combined

def get_components(cell_mask, nuclei_mask, pathogen_mask):
    # Create mappings from each cell to its nuclei, pathogens, and cytoplasms
    cell_to_nucleus = defaultdict(list)
    cell_to_pathogen = defaultdict(list)
    # Get unique cell labels
    cell_labels = np.unique(cell_mask)
    # Iterate over each cell label
    for cell_id in cell_labels:
        if cell_id == 0:
            continue
        # Find corresponding component labels
        nucleus_ids = np.unique(nuclei_mask[cell_mask == cell_id])
        pathogen_ids = np.unique(pathogen_mask[cell_mask == cell_id])
        # Update dictionaries, ignoring 0 (background) labels
        cell_to_nucleus[cell_id] = nucleus_ids[nucleus_ids != 0].tolist()
        cell_to_pathogen[cell_id] = pathogen_ids[pathogen_ids != 0].tolist()
    # Convert dictionaries to dataframes
    nucleus_df = pd.DataFrame(list(cell_to_nucleus.items()), columns=['cell_id', 'nucleus'])
    pathogen_df = pd.DataFrame(list(cell_to_pathogen.items()), columns=['cell_id', 'pathogen'])
    # Explode lists
    nucleus_df = nucleus_df.explode('nucleus')
    pathogen_df = pathogen_df.explode('pathogen')
    return nucleus_df, pathogen_df

def calculate_zernike(mask, df, degree=8):
    zernike_features = []
    for region in regionprops(mask):
        zernike_moment = zernike_moments(region.image, degree)
        zernike_features.append(zernike_moment.tolist())

    if zernike_features:
        feature_length = len(zernike_features[0])
        for feature in zernike_features:
            if len(feature) != feature_length:
                raise ValueError("All Zernike moments must be of the same length")
        
        zernike_df = pd.DataFrame(zernike_features, columns=[f'zernike_{i}' for i in range(feature_length)])
        return pd.concat([df.reset_index(drop=True), zernike_df], axis=1)
    else:
        return df

def morphological_measurements(cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask, settings, zernike=True, degree=8):
    
    morphological_props = ['label', 'area', 'area_filled', 'area_bbox', 'convex_area', 'major_axis_length', 'minor_axis_length', 
                           'eccentricity', 'solidity', 'extent', 'perimeter', 'euler_number', 'equivalent_diameter_area', 'feret_diameter_max']
    
    prop_ls = []
    ls = []
    
    # Create mappings from each cell to its nuclei, pathogens, and cytoplasms
    if settings['cell_mask_dim'] is not None:
        cell_to_nucleus, cell_to_pathogen = get_components(cell_mask, nuclei_mask, pathogen_mask)
        cell_props = pd.DataFrame(regionprops_table(cell_mask, properties=morphological_props))
        cell_props = calculate_zernike(cell_mask, cell_props, degree=degree)
        prop_ls = prop_ls + [cell_props]
        ls = ls + ['cell']
    else:
        prop_ls = prop_ls + [pd.DataFrame()]
        ls = ls + ['cell']

    if settings['nuclei_mask_dim'] is not None:
        nucleus_props = pd.DataFrame(regionprops_table(nuclei_mask, properties=morphological_props))
        nucleus_props = calculate_zernike(nuclei_mask, nucleus_props, degree=degree)
        if settings['cell_mask_dim'] is not None:
            nucleus_props = pd.merge(nucleus_props, cell_to_nucleus, left_on='label', right_on='nucleus', how='left')
        prop_ls = prop_ls + [nucleus_props]
        ls = ls + ['nucleus']
    else:
        prop_ls = prop_ls + [pd.DataFrame()]
        ls = ls + ['nucleus']
    
    if settings['pathogen_mask_dim'] is not None:
        pathogen_props = pd.DataFrame(regionprops_table(pathogen_mask, properties=morphological_props))
        pathogen_props = calculate_zernike(pathogen_mask, pathogen_props, degree=degree)
        if settings['cell_mask_dim'] is not None:
            pathogen_props = pd.merge(pathogen_props, cell_to_pathogen, left_on='label', right_on='pathogen', how='left')
        prop_ls = prop_ls + [pathogen_props]
        ls = ls + ['pathogen']
    else:
        prop_ls = prop_ls + [pd.DataFrame()]
        ls = ls + ['pathogen']

    if settings['cytoplasm']:
        cytoplasm_props = pd.DataFrame(regionprops_table(cytoplasm_mask, properties=morphological_props))
        prop_ls = prop_ls + [cytoplasm_props]
        ls = ls + ['cytoplasm']
    else:
        prop_ls = prop_ls + [pd.DataFrame()]
        ls = ls + ['cytoplasm']

    df_ls = []
    for i,df in enumerate(prop_ls):
        df.columns = [f'{ls[i]}_{col}' for col in df.columns]
        df = df.rename(columns={col: 'label' for col in df.columns if 'label' in col})
        df_ls.append(df)
 
    return df_ls[0], df_ls[1], df_ls[2], df_ls[3]

def intensity_percentiles(region):
    percentiles = np.percentile(region.intensity_image.flatten(), [5, 10, 25, 50, 75, 85, 95])
    return percentiles    

def extended_regionprops_table(labels, image, intensity_props):
    regions = regionprops(labels, image)
    props = regionprops_table(labels, image, properties=intensity_props)
    percentiles = [5, 10, 25, 50, 75, 85, 95]
    for p in percentiles:
        props[f'percentile_{p}'] = [
            np.percentile(region.intensity_image.flatten()[~np.isnan(region.intensity_image.flatten())], p)
            for region in regions]
    return pd.DataFrame(props)

def periphery_intensity(label_mask, image):
    periphery_intensity_stats = []
    boundary = find_boundaries(label_mask)
    for region in np.unique(label_mask)[1:]:  # skip the background label
        region_boundary = boundary & (label_mask == region)
        intensities = image[region_boundary]
        if intensities.size == 0:
            periphery_intensity_stats.append((region, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        else:
            periphery_intensity_stats.append((region, np.mean(intensities), np.percentile(intensities,5), np.percentile(intensities,10),
                                              np.percentile(intensities,25), np.percentile(intensities,50),
                                              np.percentile(intensities,75), np.percentile(intensities,85), 
                                              np.percentile(intensities,95)))
    return periphery_intensity_stats

def outside_intensity(label_mask, image, distance=5):
    outside_intensity_stats = []
    for region in np.unique(label_mask)[1:]:  # skip the background label
        region_mask = label_mask == region
        dilated_mask = binary_dilation(region_mask, iterations=distance)
        outside_mask = dilated_mask & ~region_mask
        intensities = image[outside_mask]
        if intensities.size == 0:
            outside_intensity_stats.append((region, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        else:
            outside_intensity_stats.append((region, np.mean(intensities), np.percentile(intensities,5), np.percentile(intensities,10),
                                              np.percentile(intensities,25), np.percentile(intensities,50),
                                              np.percentile(intensities,75), np.percentile(intensities,85), 
                                              np.percentile(intensities,95)))
    return outside_intensity_stats

def calculate_average_intensity(distance_map, single_channel_image, num_bins):
    radial_distribution = np.zeros(num_bins)
    for i in range(num_bins):
        min_distance = i * (distance_map.max() / num_bins)
        max_distance = (i + 1) * (distance_map.max() / num_bins)
        bin_mask = (distance_map >= min_distance) & (distance_map < max_distance)
        radial_distribution[i] = single_channel_image[bin_mask].mean()
    return radial_distribution

def calculate_radial_distribution(cell_mask, nuclei_mask, pathogen_mask, channel_arrays, num_bins):
    nucleus_radial_distributions = {}
    pathogen_radial_distributions = {}

    # get unique cell labels
    cell_labels = np.unique(cell_mask)
    cell_labels = cell_labels[cell_labels != 0]  # ignore background

    for cell_label in cell_labels:
        cell_region = cell_mask == cell_label

        nucleus_labels = np.unique(nuclei_mask[cell_region])
        nucleus_labels = nucleus_labels[nucleus_labels != 0]  # ignore background

        pathogen_labels = np.unique(pathogen_mask[cell_region])
        pathogen_labels = pathogen_labels[pathogen_labels != 0]  # ignore background

        for nucleus_label in nucleus_labels:
            nucleus_region = nuclei_mask == nucleus_label
            nucleus_boundary = find_boundaries(nucleus_region, mode='outer')
            distance_map = distance_transform_edt(~nucleus_boundary) * cell_region
            for channel_index in range(channel_arrays.shape[2]):  # iterating over channels
                radial_distribution = calculate_average_intensity(distance_map, channel_arrays[:, :, channel_index], num_bins)
                nucleus_radial_distributions[(cell_label, nucleus_label, channel_index)] = radial_distribution

        for pathogen_label in pathogen_labels:
            pathogen_region = pathogen_mask == pathogen_label
            pathogen_boundary = find_boundaries(pathogen_region, mode='outer')
            distance_map = distance_transform_edt(~pathogen_boundary) * cell_region
            for channel_index in range(channel_arrays.shape[2]):  # iterating over channels
                radial_distribution = calculate_average_intensity(distance_map, channel_arrays[:, :, channel_index], num_bins)
                pathogen_radial_distributions[(cell_label, pathogen_label, channel_index)] = radial_distribution

    return nucleus_radial_distributions, pathogen_radial_distributions

def create_dataframe(radial_distributions, object_type):
    df = pd.DataFrame()
    for key, value in radial_distributions.items():
        cell_label, object_label, channel_index = key
        for i in range(len(value)):
            col_name = f'{object_type}_rad_dist_channel_{channel_index}_bin_{i}'
            df.loc[object_label, col_name] = value[i]
        df.loc[object_label, 'cell_id'] = cell_label
    # Reset the index and rename the column that was previously the index
    df = df.reset_index().rename(columns={'index': 'label'})
    return df

def calculate_correlation_object_level(channel_image1, channel_image2, mask, settings):
    thresholds = settings['manders_thresholds']
    
    corr_data = {}
    for i in np.unique(mask)[1:]:
        object_mask = (mask == i)
        object_channel_image1 = channel_image1[object_mask]
        object_channel_image2 = channel_image2[object_mask]
        total_intensity1 = np.sum(object_channel_image1)
        total_intensity2 = np.sum(object_channel_image2)
        
        if len(object_channel_image1) < 2 or len(object_channel_image2) < 2:
            pearson_corr = np.nan
        else:
            pearson_corr, _ = pearsonr(object_channel_image1, object_channel_image2)
        
        corr_data[i] = {f'label_correlation': i,
                        f'Pearson_correlation': pearson_corr}
        
        for thresh in thresholds:
            chan1_thresh = np.percentile(object_channel_image1, thresh)
            chan2_thresh = np.percentile(object_channel_image2, thresh)
        
            # boolean mask where both signals are present
            overlap_mask = (object_channel_image1 > chan1_thresh) & (object_channel_image2 > chan2_thresh)
            M1 = np.sum(object_channel_image1[overlap_mask]) / total_intensity1 if total_intensity1 > 0 else 0
            M2 = np.sum(object_channel_image2[overlap_mask]) / total_intensity2 if total_intensity2 > 0 else 0
        
            corr_data[i].update({f'M1_correlation_{thresh}': M1,
                                 f'M2_correlation_{thresh}': M2})
            
    return pd.DataFrame(corr_data.values())

def estimate_blur(image):
    # Compute the Laplacian of the image
    lap = cv2.Laplacian(image, cv2.CV_64F)
    # Compute and return the variance of the Laplacian
    return lap.var()

def calculate_homogeneity(label, channel, distances=[2,4,8,16,32,64]):
    homogeneity_values = []
    # Iterate through the regions in label_mask
    for region in regionprops(label):
        region_image = (region.image * channel[region.slice]).astype(int)
        homogeneity_per_distance = []
        for d in distances:
            rescaled_image = rescale_intensity(region_image, out_range=(0, 255)).astype('uint8')
            glcm = graycomatrix(rescaled_image, [d], [0], symmetric=True, normed=True)
            homogeneity_per_distance.append(graycoprops(glcm, 'homogeneity')[0, 0])
        homogeneity_values.append(homogeneity_per_distance)
    # Create a DataFrame with the homogeneity values and appropriate column names
    columns = [f'homogeneity_distance_{d}' for d in distances]
    homogeneity_df = pd.DataFrame(homogeneity_values, columns=columns)
    
    return homogeneity_df

def intensity_measurements(cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask, channel_arrays, settings, sizes=[3, 6, 12, 24], periphery=True, outside=True):
    
    radial_dist=settings['radial_dist']
    calculate_correlation=settings['calculate_correlation']
    homogeneity=settings['homogeneity']
    distances=settings['homogeneity_distances']
    
    intensity_props = ["label", "centroid_weighted", "centroid_weighted_local", "max_intensity", "mean_intensity", "min_intensity"]
    col_lables = ['region_label', 'mean', '5_percentile', '10_percentile', '25_percentile', '50_percentile', '75_percentile', '85_percentile', '95_percentile']
    cell_dfs, nucleus_dfs, pathogen_dfs, cytoplasm_dfs = [], [], [], []
    ls = ['cell','nucleus','pathogen','cytoplasm']
    labels = [cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask]
    dfs = [cell_dfs, nucleus_dfs, pathogen_dfs, cytoplasm_dfs]
    
    for i in range(0,channel_arrays.shape[-1]):
        channel = channel_arrays[:, :, i]
        for j, (label, df) in enumerate(zip(labels, dfs)):
            mask_intensity_df = extended_regionprops_table(label, channel, intensity_props) 
            mask_intensity_df['shannon_entropy'] = shannon_entropy(channel, base=2)
            
            if homogeneity:
                # Calculate homogeneity features
                homogeneity_df = calculate_homogeneity(label, channel, distances)
                # Merge the DataFrames along the columns
                mask_intensity_df = pd.concat([mask_intensity_df.reset_index(drop=True), homogeneity_df], axis=1)
            
            if periphery:
                if ls[j] == 'nucleus' or ls[j] == 'pathogen':
                    periphery_intensity_stats = periphery_intensity(label, channel)
                    mask_intensity_df = pd.concat([mask_intensity_df, pd.DataFrame(periphery_intensity_stats, columns=[f'periphery_{stat}' for stat in col_lables])],axis=1)
            
            if outside:
                if ls[j] == 'nucleus' or ls[j] == 'pathogen':
                    outside_intensity_stats = outside_intensity(label, channel)
                    mask_intensity_df = pd.concat([mask_intensity_df, pd.DataFrame(outside_intensity_stats, columns=[f'outside_{stat}' for stat in col_lables])], axis=1)
            
            # Calculate blur for each object and add it to the dataframe (the next two lines are new)
            blur_col = [estimate_blur(channel[label == region_label]) for region_label in mask_intensity_df['label']]
            mask_intensity_df[f'{ls[j]}_channel_{i}_blur'] = blur_col
            
            mask_intensity_df.columns = [f'{ls[j]}_channel_{i}_{col}' if col != 'label' else col for col in mask_intensity_df.columns]
            df.append(mask_intensity_df)
    
    if radial_dist:
        nucleus_radial_distributions, pathogen_radial_distributions = calculate_radial_distribution(cell_mask, nuclei_mask, pathogen_mask, channel_arrays, num_bins=6)
        nucleus_df = create_dataframe(nucleus_radial_distributions, 'nucleus')
        pathogen_df = create_dataframe(pathogen_radial_distributions, 'pathogen')
        dfs[1].append(nucleus_df)
        dfs[2].append(pathogen_df)
        
    if calculate_correlation:
        for i in range(channel_arrays.shape[-1]):
            for j in range(i+1, channel_arrays.shape[-1]):
                chan_i = channel_arrays[:, :, i]
                chan_j = channel_arrays[:, :, j]
                for m, mask in enumerate(labels):
                    coloc_df = calculate_correlation_object_level(chan_i, chan_j, mask, settings)
                    coloc_df.columns = [f'{ls[m]}_channel_{i}_channel_{j}_{col}' for col in coloc_df.columns]
                    dfs[m].append(coloc_df)
    
    return pd.concat(cell_dfs, axis=1), pd.concat(nucleus_dfs, axis=1), pd.concat(pathogen_dfs, axis=1), pd.concat(cytoplasm_dfs, axis=1)

def check_integrity(df):
    df.columns = [col + f'_{i}' if df.columns.tolist().count(col) > 1 and i != 0 else col for i, col in enumerate(df.columns)]
    label_cols = [col for col in df.columns if 'label' in col]
    # Combine all 'label' columns into a list, and create a new 'label_list' column
    df['label_list'] = df[label_cols].values.tolist()
    # Create a new 'label' column from the first entry in the 'label_list' column
    df['object_label'] = df['label_list'].apply(lambda x: x[0])
    # Drop the original 'label' columns and the 'label_list' column
    df = df.drop(columns=label_cols)
    df['label_list'] = df['label_list'].astype(str)
    return df

def map_wells(file_name):
    try:
        parts = file_name.split('_')
        plate = 'p' + parts[0][5:]
        field = 'f' + str(int(parts[2]))
        well = parts[1]
        row = 'r' + str(string.ascii_uppercase.index(well[0]) + 1)
        column = 'c' + str(int(well[1:]))
        prcf = '_'.join([plate, row, column, field])
    except Exception as e:
        print(f"Error processing filename: {file_name}")
        print(f"Error: {e}")
        plate, row, column, field, prcf = 'error','error','error','error','error'
    return plate, row, column, field, prcf

def safe_int_convert(value, default=0):
    try:
        return int(value)
    except ValueError:
        return default

def map_wells_png(file_name):
    try:
        root, ext = os.path.splitext(file_name)
        parts = root.split('_')
        plate = 'p' + parts[0][5:]
        field = 'f' + str(safe_int_convert(parts[2]))
        well = parts[1]
        row = 'r' + str(string.ascii_uppercase.index(well[0]) + 1)
        column = 'c' + str(safe_int_convert(well[1:]))
        cell_id = 'o' + str(safe_int_convert(parts[-1], default='none'))
        prcfo = '_'.join([plate, row, column, field, cell_id])
    except Exception as e:
        print(f"Error processing filename: {file_name}")
        print(f"Error: {e}")
        plate, row, column, field, cell_id, prcfo = 'error', 'error', 'error', 'error', 'error', 'error'
    return plate, row, column, field, cell_id, prcfo
    
def merge_and_save_to_database(morph_df, intensity_df, table_type, source_folder, file_name, experiment):
    morph_df = check_integrity(morph_df)
    intensity_df = check_integrity(intensity_df)
    if len(morph_df) > 0 and len(intensity_df) > 0:
        merged_df = pd.merge(morph_df, intensity_df, on='object_label', how='outer')
        merged_df = merged_df.rename(columns={"label_list_x": "label_list_morphology", "label_list_y": "label_list_intensity"})
        merged_df['file_name'] = file_name
        merged_df['path_name'] = os.path.join(source_folder, file_name + '.npy')
        merged_df[['plate', 'row', 'col', 'field', 'prcf']] = merged_df['file_name'].apply(lambda x: pd.Series(map_wells(x)))
        cols = merged_df.columns.tolist()  # get the list of all columns
        
        if table_type == 'cell' or table_type == 'cytoplasm':
            column_list = ['object_label', 'plate', 'row', 'col', 'field', 'prcf', 'file_name', 'path_name']
        elif table_type == 'nucleus' or table_type == 'pathogen':
            column_list = ['object_label', 'cell_id', 'plate', 'row', 'col', 'field', 'prcf', 'file_name', 'path_name']
        else:
            raise ValueError(f"Invalid table_type: {table_type}")
            
        # Check if all columns in column_list are in cols
        missing_columns = [col for col in column_list if col not in cols]

        if len(missing_columns) == 1 and missing_columns[0] == 'cell_id':
            missing_columns = False
            column_list = ['object_label', 'plate', 'row', 'col', 'field', 'prcf', 'file_name', 'path_name']
        
        if missing_columns:
            raise ValueError(f"Columns missing in DataFrame: {missing_columns}")
        
        for i, col in enumerate(column_list):
            cols.insert(i, cols.pop(cols.index(col)))
            
        merged_df = merged_df[cols]  # rearrange the columns
        
        if len(merged_df) > 0:
            try:
                conn = sqlite3.connect(f'{source_folder}/measurements/measurements.db', timeout=5)
                merged_df.to_sql(table_type, conn, if_exists='append', index=False)
            except sqlite3.OperationalError as e:
                print("SQLite error:", e)

def exclude_objects(cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask, include_uninfected=True):
    # Remove cells with no nucleus or cytoplasm (or pathogen)
    filtered_cells = np.zeros_like(cell_mask) # Initialize a new mask to store the filtered cells.
    for cell_label in np.unique(cell_mask): # Iterate over all cell labels in the cell mask.
        if cell_label == 0: # Skip background
            continue

        cell_region = cell_mask == cell_label # Get a mask for the current cell.
        
        # Check existence of nucleus, cytoplasm and pathogen in the current cell.
        has_nucleus = np.any(nuclei_mask[cell_region])
        has_cytoplasm = np.any(cytoplasm_mask[cell_region])
        has_pathogen = np.any(pathogen_mask[cell_region])
        
        if include_uninfected:
            if has_nucleus and has_cytoplasm:
                filtered_cells[cell_region] = cell_label
        else:
            if has_nucleus and has_cytoplasm and has_pathogen:
                filtered_cells[cell_region] = cell_label
                
    # Remove objects outside of cells
    nuclei_mask = nuclei_mask * (filtered_cells > 0)
    pathogen_mask = pathogen_mask * (filtered_cells > 0)
    cytoplasm_mask = cytoplasm_mask * (filtered_cells > 0)

    return filtered_cells, nuclei_mask, pathogen_mask, cytoplasm_mask

def filter_object(mask, min_value):
    count = np.bincount(mask.ravel())
    to_remove = np.where(count < min_value)
    mask[np.isin(mask, to_remove)] = 0
    return mask

def merge_overlapping_objects(mask1, mask2):
    # Label unique objects in pathogen_mask
    labeled_1 = label(mask1)
    # Determine the number of unique pathogens
    num_1 = np.max(labeled_1)
    # Iterate over unique objects in pathogen_mask
    for m1_id in range(1, num_1 + 1):
        # Create a mask for the current pathogen object
        current_1_mask = labeled_1 == m1_id
        # Identify the overlapping cell labels
        overlapping_2_labels = np.unique(mask2[current_1_mask])
        # Ignore background (label 0) and filter out unique labels
        overlapping_2_labels = overlapping_2_labels[overlapping_2_labels != 0]

        # Check if the current pathogen object overlaps with more than one cell object
        if len(overlapping_2_labels) > 1:
            # Calculate the overlap percentage with each cell object
            overlap_percentages = [np.sum(current_1_mask & (mask2 == m2_label)) / np.sum(current_1_mask) * 100 for m2_label in overlapping_2_labels]
            # Find the label with the largest overlap
            max_overlap_label = overlapping_2_labels[np.argmax(overlap_percentages)]
            max_overlap_percentage = max(overlap_percentages)

            # If the max overlap is 90% or more, remove the remainder of the mask1 object that overlaps with other mask2 objects
            if max_overlap_percentage >= 90:
                for m2_label in overlapping_2_labels:
                    if m2_label != max_overlap_label:
                        mask1[(current_1_mask) & (mask2 == m2_label)] = 0
            else:
                # Merge the overlapping cell objects into one
                for m2_label in overlapping_2_labels[1:]:
                    mask2[mask2 == m2_label] = overlapping_2_labels[0]
                    
    return mask1, mask2
    
def generate_names(file_name, cell_id, cell_nuclei_ids, cell_pathogen_ids, source_folder, crop_mode='cell'):
    non_zero_cell_ids = cell_id[cell_id != 0]
    cell_id_str = "multi" if non_zero_cell_ids.size > 1 else str(non_zero_cell_ids[0]) if non_zero_cell_ids.size == 1 else "none"

    cell_nuclei_ids = cell_nuclei_ids[cell_nuclei_ids != 0]
    cell_nuclei_id_str = "multi" if cell_nuclei_ids.size > 1 else str(cell_nuclei_ids[0]) if cell_nuclei_ids.size == 1 else "none"

    cell_pathogen_ids = cell_pathogen_ids[cell_pathogen_ids != 0]
    cell_pathogen_id_str = "multi" if cell_pathogen_ids.size > 1 else str(cell_pathogen_ids[0]) if cell_pathogen_ids.size == 1 else "none"

    fldr = f"{source_folder}/data/"
    img_name = ""

    if crop_mode == 'nucleus':
        img_name = f"{file_name}_{cell_id_str}_{cell_nuclei_id_str}.png"
        fldr += "single_nucleus/" if cell_nuclei_ids.size == 1 else "multiple_nuclei/" if cell_nuclei_ids.size > 1 else "no_nucleus/"
        fldr += "single_pathogen/" if cell_pathogen_ids.size == 1 else "multiple_pathogens/" if cell_pathogen_ids.size > 1 else "uninfected/"

    elif crop_mode == 'pathogen':
        img_name = f"{file_name}_{cell_id_str}_{cell_pathogen_id_str}.png"
        fldr += "single_nucleus/" if cell_nuclei_ids.size == 1 else "multiple_nuclei/" if cell_nuclei_ids.size > 1 else "no_nucleus/"
        fldr += "infected/" if cell_pathogen_ids.size >= 1 else "uninfected/"

    elif crop_mode == 'cell' or crop_mode == 'cytoplasm':
        img_name = f"{file_name}_{cell_id_str}.png"
        fldr += "single_nucleus/" if cell_nuclei_ids.size == 1 else "multiple_nuclei/" if cell_nuclei_ids.size > 1 else "no_nucleus/"
        fldr += "single_pathogen/" if cell_pathogen_ids.size == 1 else "multiple_pathogens/" if cell_pathogen_ids.size > 1 else "uninfected/"

    table_name = fldr.replace("/", "_")
    
    return img_name, fldr, table_name

def find_bounding_box(crop_mask, _id, buffer=10):
    object_indices = np.where(crop_mask == _id)

    # Determine the bounding box coordinates
    y_min, y_max = object_indices[0].min(), object_indices[0].max()
    x_min, x_max = object_indices[1].min(), object_indices[1].max()

    # Add buffer to the bounding box coordinates
    y_min = max(y_min - buffer, 0)
    y_max = min(y_max + buffer, crop_mask.shape[0] - 1)
    x_min = max(x_min - buffer, 0)
    x_max = min(x_max + buffer, crop_mask.shape[1] - 1)

    # Create a new mask with the same dimensions as crop_mask
    new_mask = np.zeros_like(crop_mask)

    # Fill in the bounding box area with the _id
    new_mask[y_min:y_max+1, x_min:x_max+1] = _id

    return new_mask

def measure_crop_core(index, time_ls, file, settings):
    start = time.time() 
    try:
        source_folder = os.path.dirname(settings['input_folder'])
        file_name = os.path.splitext(file)[0]
        data = np.load(os.path.join(settings['input_folder'], file))
        data_type = data.dtype
        if settings['save_measurements']:
            os.makedirs(source_folder+'/measurements', exist_ok=True)
            create_database(source_folder+'/measurements/measurements.db')    

        if settings['plot_filtration']:
            #before = data[:, :, len(settings['channels'])+1:]
            plot_cropped_arrays(data)

        channel_arrays = data[:, :, settings['channels']].astype(data_type)        
        if settings['cell_mask_dim'] is not None:
            cell_mask = data[:, :, settings['cell_mask_dim']].astype(data_type)
            if settings['cell_min_size'] is not None and settings['cell_min_size'] != 0:
                cell_mask = filter_object(cell_mask, settings['cell_min_size']) # Filter out small cells
        else:
            cell_mask = np.zeros_like(data[:, :, 0])
            settings['cytoplasm'] = False
            settings['include_uninfected'] = True

        if settings['nuclei_mask_dim'] is not None:
            nuclei_mask = data[:, :, settings['nuclei_mask_dim']].astype(data_type)
            if settings['cell_mask_dim'] is not None:
                nuclei_mask, cell_mask = merge_overlapping_objects(mask1=nuclei_mask, mask2=cell_mask)
            if settings['nucleus_min_size'] is not None and settings['nucleus_min_size'] != 0:
                nuclei_mask = filter_object(nuclei_mask, settings['nucleus_min_size']) # Filter out small nuclei
        else:
            nuclei_mask = np.zeros_like(data[:, :, 0])

        if settings['pathogen_mask_dim'] is not None:
            pathogen_mask = data[:, :, settings['pathogen_mask_dim']].astype(data_type)
            if settings['merge_edge_pathogen_cells']:
                if settings['cell_mask_dim'] is not None:
                    pathogen_mask, cell_mask = merge_overlapping_objects(mask1=pathogen_mask, mask2=cell_mask)
            if settings['pathogen_min_size'] is not None and settings['pathogen_min_size'] != 0:
                pathogen_mask = filter_object(pathogen_mask, settings['pathogen_min_size']) # Filter out small pathogens
        else:
            pathogen_mask = np.zeros_like(data[:, :, 0])
                        
        # Create cytoplasm mask
        if settings['cytoplasm']:
            if settings['cell_mask_dim'] is not None:
                if settings['nuclei_mask_dim'] is not None and settings['pathogen_mask_dim'] is not None:
                    cytoplasm_mask = np.where(np.logical_or(nuclei_mask != 0, pathogen_mask != 0), 0, cell_mask)
                elif settings['nuclei_mask_dim'] is not None:
                    cytoplasm_mask = np.where(np.logical_or(nuclei_mask != 0), 0, cell_mask)
                elif settings['pathogen_mask_dim'] is not None:
                    cytoplasm_mask = np.where(np.logical_or(pathogen_mask != 0), 0, cell_mask)
                else:
                    cytoplasm_mask = np.zeros_like(cell_mask)
        else:
            cytoplasm_mask = np.zeros_like(cell_mask)
        
        if settings['cell_min_size'] is not None and settings['cell_min_size'] != 0:
            cell_mask = filter_object(cell_mask, settings['cell_min_size']) # Filter out small cells
        if settings['cytoplasm_min_size'] is not None and settings['nucleus_min_size'] != 0:
            nuclei_mask = filter_object(nuclei_mask, settings['nucleus_min_size']) # Filter out small nuclei
        if settings['pathogen_min_size'] is not None and settings['pathogen_min_size'] != 0:
            pathogen_mask = filter_object(pathogen_mask, settings['pathogen_min_size']) # Filter out small pathogens
        if settings['cytoplasm_min_size'] is not None and settings['cytoplasm_min_size'] != 0:
            cytoplasm_mask = filter_object(cytoplasm_mask, settings['cytoplasm_min_size']) # Filter out small cytoplasms

        if settings['include_uninfected'] == False:
            cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask = exclude_objects(cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask, include_uninfected=False)
        
        # Update data with the new masks
        if settings['cell_mask_dim'] is not None:
            data[:, :, settings['cell_mask_dim']] = cell_mask.astype(data_type)
        if settings['nuclei_mask_dim'] is not None:
            data[:, :, settings['nuclei_mask_dim']] = nuclei_mask.astype(data_type)
        if settings['pathogen_mask_dim'] is not None:
            data[:, :, settings['pathogen_mask_dim']] = pathogen_mask.astype(data_type)
        if settings['cytoplasm']:
            data = np.concatenate((data, cytoplasm_mask[:, :, np.newaxis]), axis=2)
        
        if settings['plot_filtration']:
            #after = data[:, :, len(settings['channels'])+1:]
            plot_cropped_arrays(data)

        if settings['save_measurements']:
		
            cell_df, nucleus_df, pathogen_df, cytoplasm_df = morphological_measurements(cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask, settings)
            cell_intensity_df, nucleus_intensity_df, pathogen_intensity_df, cytoplasm_intensity_df = intensity_measurements(cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask, channel_arrays, settings, sizes=[1, 2, 3, 4, 5], periphery=True, outside=True)
            if settings['cell_mask_dim'] is not None:
            	cell_merged_df = merge_and_save_to_database(cell_df, cell_intensity_df, 'cell', source_folder, file_name, settings['experiment'])
            if settings['nuclei_mask_dim'] is not None:
            	nucleus_merged_df = merge_and_save_to_database(nucleus_df, nucleus_intensity_df, 'nucleus', source_folder, file_name, settings['experiment'])
            if settings['pathogen_mask_dim'] is not None:
            	pathogen_merged_df = merge_and_save_to_database(pathogen_df, pathogen_intensity_df, 'pathogen', source_folder, file_name, settings['experiment'])
            if settings['cytoplasm']:
            	cytoplasm_merged_df = merge_and_save_to_database(cytoplasm_df, cytoplasm_intensity_df, 'cytoplasm', source_folder, file_name, settings['experiment'])

        if settings['save_png'] or settings['save_arrays'] or settings['plot']:
            
            if isinstance(settings['dialate_pngs'], bool):
                dialate_pngs = [settings['dialate_pngs'], settings['dialate_pngs'], settings['dialate_pngs']]
            if isinstance(settings['dialate_pngs'], list):
                dialate_pngs = settings['dialate_pngs']

            if isinstance(settings['dialate_png_ratios'], float):
                dialate_png_ratios = [settings['dialate_png_ratios'], settings['dialate_png_ratios'], settings['dialate_png_ratios']]
            if isinstance(settings['dialate_png_ratios'], list):
                dialate_png_ratios = settings['dialate_png_ratios']
                
            if isinstance(settings['crop_mode'], str):
                crop_mode = [settings['crop_mode']]
            if isinstance(settings['crop_mode'], list):
                crop_ls = settings['crop_mode']
                size_ls = settings['png_size']
                for crop_idx, crop_mode in enumerate(crop_ls):
                    width, height = size_ls[crop_idx]
                    if crop_mode == 'cell':
                        crop_mask = cell_mask.copy()
                        dialate_png = dialate_pngs[crop_idx]
                        dialate_png_ratio = dialate_png_ratios[crop_idx]
                    elif crop_mode == 'nucleus':
                        crop_mask = nuclei_mask.copy()
                        dialate_png = dialate_pngs[crop_idx]
                        dialate_png_ratio = dialate_png_ratios[crop_idx]
                    elif crop_mode == 'pathogen':
                        crop_mask = pathogen_mask.copy()
                        dialate_png = dialate_pngs[crop_idx]
                        dialate_png_ratio = dialate_png_ratios[crop_idx]
                    elif crop_mode == 'cytoplasm':
                        crop_mask = cytoplasm_mask.copy()
                        dialate_png = False
                    else:
                        print(f'Value error: Posseble values for crop_mode are: cell, nucleus, pathogen, cytoplasm')

                    objects_in_image = np.unique(crop_mask)
                    objects_in_image = objects_in_image[objects_in_image != 0]
                    img_paths = []

                    for _id in objects_in_image:
                        region = crop_mask == _id
                        region_cell_ids = np.atleast_1d(np.unique(cell_mask * region))
                        region_nuclei_ids = np.atleast_1d(np.unique(nuclei_mask * region))
                        region_pathogen_ids = np.atleast_1d(np.unique(pathogen_mask * region))
                        
                        if settings['use_bounding_box']:
                            region = find_bounding_box(crop_mask, _id, buffer=10)
                        
                        img_name, fldr, table_name = generate_names(file_name=file_name, cell_id=region_cell_ids, cell_nuclei_ids=region_nuclei_ids, cell_pathogen_ids=region_pathogen_ids, source_folder=source_folder, crop_mode=crop_mode)
                        
                        if dialate_png:
                            region_area = np.sum(region)
                            approximate_diameter = np.sqrt(region_area)
                            dialate_png_px = int(approximate_diameter * dialate_png_ratio) 
                            struct = generate_binary_structure(2, 2)
                            region = binary_dilation(region, structure=struct, iterations=dialate_png_px)
                        
                        if settings['save_png']:
                            png_folder = f"{fldr}{crop_mode}_png/"
                            img_path = os.path.join(png_folder, img_name)
                            
                            png_channels = data[:, :, settings['png_dims']].astype(data_type)
                            
                            if settings['normalize_by'] == 'fov':
                                percentiles_list = get_percentiles(png_channels, settings['normalize_percentiles'][0],q2=settings['normalize_percentiles'][1])
                            
                            png_channels = crop_center(png_channels, region, new_width=width, new_height=height)

                            if isinstance(settings['normalize'], list):
                                if settings['normalize_by'] == 'png':
                                    png_channels = normalize_to_dtype(png_channels, q1=settings['normalize'][0],q2=settings['normalize'][1])
                                if settings['normalize_by'] == 'fov':
                                    png_channels = normalize_to_dtype(png_channels, q1=settings['normalize'][0],q2=settings['normalize'][1], percentiles=percentiles_list)
					
                            os.makedirs(png_folder, exist_ok=True)

                            if png_channels.shape[2] == 2:
                                dummy_channel = np.zeros_like(png_channels[:,:,0])  # Create a 2D zero array with same shape as one channel
                                png_channels = np.dstack((png_channels, dummy_channel))
                                cv2.imwrite(img_path, png_channels)
                            else:
                                cv2.imwrite(img_path, png_channels)

                        	#if settings['save_measurements']:
                            img_paths.append(img_path)
                            if len(img_paths) == len(objects_in_image):
                                png_df = pd.DataFrame(img_paths, columns=['png_path'])
                                png_df['file_name'] = png_df['png_path'].apply(lambda x: os.path.basename(x))

                                if crop_mode == 'cell':
                                    png_df[['plate', 'row', 'col', 'field', 'cell_id', 'prcfo']] = png_df['file_name'].apply(lambda x: pd.Series(map_wells_png(x)))

                                elif crop_mode == 'nucleus':
                                    png_df[['plate', 'row', 'col', 'field', 'nucleus_id', 'prcfo']] = png_df['file_name'].apply(lambda x: pd.Series(map_wells_png(x)))

                                elif crop_mode == 'pathogen':
                                    png_df[['plate', 'row', 'col', 'field', 'pathogen_id', 'prcfo']] = png_df['file_name'].apply(lambda x: pd.Series(map_wells_png(x)))

                                elif crop_mode == 'cytoplasm':
                                    png_df[['plate', 'row', 'col', 'field', 'cytoplasm_id', 'prcfo']] = png_df['file_name'].apply(lambda x: pd.Series(map_wells_png(x)))

                                try:
                                    conn = sqlite3.connect(f'{source_folder}/measurements/measurements.db', timeout=5)
                                    png_df.to_sql('png_list', conn, if_exists='append', index=False)
                                    conn.commit()
                                except sqlite3.OperationalError as e:
                                    print(f"SQLite error: {e}", flush=True)
                            
                            if settings['plot']:
                                plot_cropped_arrays(png_channels)

                        if settings['save_arrays']:
                            row_idx, col_idx = np.where(region)
                            region_array = data[row_idx.min():row_idx.max()+1, col_idx.min():col_idx.max()+1, :]
                            array_folder = f"{fldr}/region_array/"            
                            os.makedirs(array_folder, exist_ok=True)
                            np.save(os.path.join(array_folder, img_name), region_array)
                            if plot:
                                plot_cropped_arrays(region_array)

                        if not settings['save_arrays'] and not settings['save_png'] and settings['plot']:
                            row_idx, col_idx = np.where(region)
                            region_array = data[row_idx.min():row_idx.max()+1, col_idx.min():col_idx.max()+1, :]
                            plot_cropped_arrays(region_array)

        cells = np.unique(cell_mask)
    except Exception as e:
        print('main',e)
        cells = 0
        traceback.print_exc()
    
    end = time.time()
    duration = end-start
    time_ls.append(duration)
    average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
    return average_time, cells

def save_settings_to_db(settings):
    # Convert the settings dictionary into a DataFrame
    settings_df = pd.DataFrame(list(settings.items()), columns=['setting_key', 'setting_value'])

    # Convert all values in the 'setting_value' column to strings
    settings_df['setting_value'] = settings_df['setting_value'].apply(str)
    display(settings_df)
    # Determine the directory path
    src = os.path.dirname(settings['input_folder'])
    directory = f'{src}/measurements'

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Database connection and saving the settings DataFrame
    conn = sqlite3.connect(f'{directory}/measurements.db', timeout=5)
    settings_df.to_sql('settings', conn, if_exists='replace', index=False)  # Replace the table if it already exists
    conn.close()

def measure_crop(settings):
    #general settings
    settings['merge_edge_pathogen_cells'] = True
    settings['radial_dist'] = True
    settings['calculate_correlation'] = True
    settings['manders_thresholds'] = [15,85,95]
    settings['homogeneity'] = True
    settings['homogeneity_distances'] = [8,16,32]
    settings['save_arrays'] = False
    
    if settings['cell_mask_dim'] is None:
    	settings['include_uninfected'] = True
    
    if settings['cell_mask_dim'] is not None and settings['pathogen_min_size'] is not None:
    	settings['cytoplasm'] = True
    elif settings['cell_mask_dim'] is not None and settings['nucleus_min_size'] is not None:
    	settings['cytoplasm'] = True
    else:
    	settings['cytoplasm'] = False
    
    settings['center_crop'] = True

    int_setting_keys = ['cell_mask_dim', 'nuclei_mask_dim', 'pathogen_mask_dim', 'cell_min_size', 'nucleus_min_size', 'pathogen_min_size', 'cytoplasm_min_size']
    
    if isinstance(settings['normalize'], bool) and settings['normalize']:
        print(f'WARNING: to notmalize single object pngs set normalize to a list of 2 integers, e.g. [1,99] (lower and upper percentiles)')
        return

    if settings['normalize_by'] not in ['png', 'fov']:
        print("Warning: normalize_by should be either 'png' to notmalize each png to its own percentiles or 'fov' to normalize each png to the fov percentiles ")
        return

    if not all(isinstance(settings[key], int) or settings[key] is None for key in int_setting_keys):
        print(f"WARNING: {int_setting_keys} must all be integers")
        return

    if not isinstance(settings['channels'], list):
        print(f"WARNING: channels should be a list of integers representing channels e.g. [0,1,2,3]")
        return

    if not isinstance(settings['crop_mode'], list):
        print(f"WARNING: crop_mode should be a list with at least one element e.g. ['cell'] or ['cell','nucleus'] or [None]")
        return
    
    save_settings_to_db(settings)

    files = [f for f in os.listdir(settings['input_folder']) if f.endswith('.npy')]
    max_workers = settings['max_workers'] or mp.cpu_count()-4
    print(f'using {max_workers} cpu cores')

    with mp.Manager() as manager:
        time_ls = manager.list()
        with mp.Pool(max_workers) as pool:
            result = pool.starmap_async(measure_crop_core, [(index, time_ls, file, settings) for index, file in enumerate(files)])

            # Track progress in the main process
            while not result.ready():  # Run the loop until all tasks have finished
                time.sleep(1)  # Wait for a short amount of time to avoid excessive printing
                files_processed = len(time_ls)
                files_to_process = len(files)
                average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
                time_left = (((files_to_process-files_processed)*average_time)/max_workers)/60
                print(f'Progress: {files_processed}/{files_to_process} Time/img {average_time:.3f}sec, Time Remaining {time_left:.3f} min.', end='\r', flush=True)
            result.get()  # This will block until all tasks are finished

def load_and_concatenate_arrays(src, channels, cell_chann_dim, nucleus_chann_dim, pathogen_chann_dim):
    folder_paths = [os.path.join(src+'/stack')]
    if cell_chann_dim is not None or os.path.exists(os.path.join(src, 'norm_channel_stack', 'cell_mask_stack')):
        folder_paths = folder_paths + [os.path.join(src, 'norm_channel_stack','cell_mask_stack')]
    if nucleus_chann_dim is not None or os.path.exists(os.path.join(src, 'norm_channel_stack', 'nuclei_mask_stack')):
        folder_paths = folder_paths + [os.path.join(src, 'norm_channel_stack','nuclei_mask_stack')]
    if pathogen_chann_dim is not None or os.path.exists(os.path.join(src, 'norm_channel_stack', 'pathogen_mask_stack')):
        folder_paths = folder_paths + [os.path.join(src, 'norm_channel_stack','pathogen_mask_stack')]
	
    output_folder = src+'/merged'
    reference_folder = folder_paths[0]
    os.makedirs(output_folder, exist_ok=True)
    
    count=0
    all_imgs = len(os.listdir(reference_folder))
    
    # Iterate through each file in the reference folder
    for filename in os.listdir(reference_folder):
        
        stack_ls = []
        array_path = []
        
        if filename.endswith('.npy'):
            count+=1
            # Initialize the concatenated array with the array from the reference folder
            concatenated_array = np.load(os.path.join(reference_folder, filename))
            if channels is not None:
                concatenated_array = np.take(concatenated_array, channels, axis=2)
                stack_ls.append(concatenated_array)
            # For each of the other folders, load the array and concatenate it
            for folder in folder_paths[1:]:
                array_path = os.path.join(folder, filename)
                if os.path.isfile(array_path):
                    array = np.load(array_path)
                    if array.ndim == 2:
                        array = np.expand_dims(array, axis=-1)  # add an extra dimension if the array is 2D
                    stack_ls.append(array)   
            unique_shapes = {arr.shape[:-1] for arr in stack_ls}
            if len(unique_shapes) > 1:
                #max_dims = np.max(np.array(list(unique_shapes)), axis=0)
                # Determine the maximum length of tuples in unique_shapes
                max_tuple_length = max(len(shape) for shape in unique_shapes)
                # Pad shorter tuples with zeros to make them all the same length
                padded_shapes = [shape + (0,) * (max_tuple_length - len(shape)) for shape in unique_shapes]
                # Now create a NumPy array and find the maximum dimensions
                max_dims = np.max(np.array(padded_shapes), axis=0)
                print(f'Warning: arrays with multiple shapes found. Padding arrays to max X,Y dimentions {max_dims}', end='\r', flush=True)
                padded_stack_ls = []
                for arr in stack_ls:
                    pad_width = [(0, max_dim - dim) for max_dim, dim in zip(max_dims, arr.shape[:-1])]
                    pad_width.append((0, 0))
                    padded_arr = np.pad(arr, pad_width)
                    padded_stack_ls.append(padded_arr)
                # Concatenate the padded arrays along the channel dimension (last dimension)
                stack = np.concatenate(padded_stack_ls, axis=-1)
 
            else:
                stack = np.concatenate(stack_ls, axis=-1)
            #stack = np.concatenate((concatenated_array, array), axis=-1)
            #Save the concatenated array to the output folder
            output_path = os.path.join(output_folder, filename)
            np.save(output_path, stack)
        print(f'Files merged: {count}/{all_imgs}', end='\r', flush=True)
    return

def preprocess_img_data(src, metadata_type='cellvoyager', custom_regex=None, img_format='.tif', bitdepth='uint16', cmap='inferno', figuresize=15, normalize=False, nr=1, plot=False, mask_channels=[0,1,2], batch_size=[100,100,100], timelapse=False, remove_background=False, backgrounds=100, lower_quantile=0.01, save_dtype=np.float32, correct_illumination=False, randomize=True, generate_movies=False, all_to_mip=False, fps=2, pick_slice=False, skip_mode='01', settings={}):
    
    print(f'========== settings ==========')
    print(f'source == {src}')
    print(f'Bitdepth: {bitdepth}: cmap:{cmap}: figuresize:{figuresize}')
    print(f'========== consolidating arrays ==========')
    
    if metadata_type == 'cellvoyager':
        regex = f'(?P<plateID>.*)_(?P<wellID>.*)_T(?P<timeID>.*)F(?P<fieldID>.*)L(?P<laserID>..)A(?P<AID>..)Z(?P<sliceID>.*)C(?P<chanID>.*){img_format}'
    if metadata_type == 'cq1':
        regex = f'W(?P<wellID>.*)F(?P<fieldID>.*)T(?P<timeID>.*)Z(?P<sliceID>.*)C(?P<chanID>.*){img_format}'
    if metadata_type == 'nikon':
        regex = f'(?P<plateID>.*)_(?P<wellID>.*)_T(?P<timeID>.*)F(?P<fieldID>.*)L(?P<laserID>..)A(?P<AID>..)Z(?P<sliceID>.*)C(?P<chanID>.*){img_format}'
    if metadata_type == 'zeis':
        regex = f'(?P<plateID>.*)_(?P<wellID>.*)_T(?P<timeID>.*)F(?P<fieldID>.*)L(?P<laserID>..)A(?P<AID>..)Z(?P<sliceID>.*)C(?P<chanID>.*){img_format}'
    if metadata_type == 'leica':
        regex = f'(?P<plateID>.*)_(?P<wellID>.*)_T(?P<timeID>.*)F(?P<fieldID>.*)L(?P<laserID>..)A(?P<AID>..)Z(?P<sliceID>.*)C(?P<chanID>.*){img_format}'
    if metadata_type == 'custom':
        regex = f'({custom_regex}){img_format}'
    
    print(f'regex == {regex}')
    if not os.path.exists(src+'/stack'):
        if timelapse:
            move_to_chan_folder(src, 
                                regex=regex, 
                                timelapse=timelapse)
        else:
            print(f'========== creating single channel folders ==========')
            #z_to_mip(src, regex=regex)
            z_to_mip(src, regex=regex, batch_size=batch_size, pick_slice=pick_slice, skip_mode=skip_mode)
		
            #Make sure no batches will be of only one image
            all_imgs = len(src+'/stack')
            full_batches = all_imgs // batch_size
            last_batch_size = all_imgs % batch_size
            
            # Check if the last batch is of size 1
            if last_batch_size == 1:
                # If there's only one batch and its size is 1, it's also an issue
                if full_batches == 0:
                    raise ValueError("Only one batch of size 1 detected. Adjust the batch size.")
                # If the last batch is of size 1, merge it with the second last batch
                elif full_batches > 0:
                    raise ValueError("Last batch of size 1 detected. Adjust the batch size.")
    
        print(f'========== generating stack ==========')
        merge_channels(src, plot=False)
        #len(src+'stack')
        if plot:
            print(f'plotting {nr} images from {src}/stack')
            plot_arrays(src+'/stack', figuresize, cmap, nr=nr, normalize=normalize)
        if all_to_mip:
            mip_all(src+'/stack')
            if plot:
                print(f'plotting {nr} images from {src}/stack')
                plot_arrays(src+'/stack', figuresize, cmap, nr=nr, normalize=normalize)
    nr_of_stacks = len(src+'/channel_stack')

    print(f'========== concatinating stacks to npz ==========: {batch_size} stacks per npz in {nr_of_stacks}')
    concatenate_channel(src+'/stack', 
                        channels=mask_channels, 
                        randomize=randomize, 
                        timelapse=timelapse, 
                        batch_size=batch_size)
    if plot:
        print(f'plotting {nr} images from {src}/channel_stack')
        plot_4D_arrays(src+'/channel_stack', figuresize, cmap, nr_npz=1, nr=nr)
    nr_of_chan_stacks = len(src+'/channel_stack')
    
    print(f'========== normalizing concatinated npz ==========: {batch_size} stacks per npz in {nr_of_chan_stacks}')
    
    backgrounds, signal_to_noise, signal_thresholds = get_lists_for_normalization(settings=settings)

    #print(f'backgrounds:{backgrounds}')
    #print(f'signal_to_noise:{signal_to_noise}')
    #print(f'signal_thresholds:{signal_thresholds}')
    
    if not timelapse:
        normalize_stack(src+'/channel_stack',
                    backgrounds=backgrounds,
                    lower_quantile=lower_quantile,
                    save_dtype=save_dtype,
                    signal_thresholds=signal_thresholds,
                    correct_illumination=correct_illumination,
                    signal_to_noise=signal_to_noise, 
                    remove_background=remove_background)
    else:
        normalize_timelapse(src, lower_quantile=lower_quantile, save_dtype=save_dtype)
        
    if plot:
        plot_4D_arrays(src+'/norm_channel_stack', nr_npz=1, nr=nr)
    if generate_movies:
        make_movies(src+'/norm_channel_stack', fps=fps)
    return print(f'========== complete ==========')

def get_diam(mag, obj):
    if obj == 'cell':
        if mag == 20:
            scale = 6
        if mag == 40:
            scale = 4.5
        if mag == 60:
            scale = 3
    elif obj == 'nuclei':
        if mag == 20:
            scale = 3
        if mag == 40:
            scale = 2
        if mag == 60:
            scale = 1.5
    elif obj == 'pathogen':
        if mag == 20:
            scale = 1.5
        if mag == 40:
            scale = 1
        if mag == 60:
            scale = 1.25
    elif obj == 'pathogen_nuclei':
        if mag == 20:
            scale = 0.25
        if mag == 40:
            scale = 0.2
        if mag == 60:
            scale = 0.2
    else:
        raise ValueError("Invalid object type")
    diamiter = mag*scale
    return diamiter

def generate_masks(src, object_type, mag, batch_size, channels, cellprob_threshold, plot, save, verbose, nr=1, start_at=0, merge=False, file_type='.npz', timelapse=False, settings={}):
    
    if object_type == 'cell':
        refine_masks = False
        filter_size = False
        filter_dimm = False
        remove_border_objects = False
        if settings['nucleus_channel'] is None:
            model_name = 'cyto'
        else:
            model_name = 'cyto2'
        diameter = get_diam(mag, obj='cell')
        minimum_size = (diameter**2)/10
        maximum_size = minimum_size*50
        merge = merge
        net_avg=True
        resample=True
    elif object_type == 'nuclei':
        refine_masks = False
        filter_size = False
        filter_dimm = False
        remove_border_objects = False
        model_name = 'nuclei'
        diameter = get_diam(mag, obj='nuclei')
        minimum_size = (diameter**2)/10
        maximum_size = minimum_size*50
        merge = merge
        net_avg=True
        resample=True
    elif object_type == 'pathogen':
        refine_masks = False
        filter_size = False
        filter_dimm = False
        remove_border_objects = False
        model_name = 'cyto'
        diameter = get_diam(mag, obj='pathogen')
        minimum_size = (diameter**2)/5
        maximum_size = minimum_size*50
        merge = merge
        net_avg=True
        resample=True
    elif object_type == 'pathogen_nuclei':
        refine_masks = False
        filter_size = True
        filter_dimm = False
        remove_border_objects = False
        model_name = 'cyto'
        diameter = mag/4
        diameter = get_diam(mag, obj='pathogen_nuclei')
        maximum_size = minimum_size*100
        merge = merge
        net_avg=False
        resample=True
    else:
        print(f'Object type: {object_type} not supported. Supported object types are : cell, nuclei and pathogen')
    if verbose:
        print(f'Mode:{object_type} Mag.:{mag} Diamiter:{diameter} Min.:{minimum_size} Max.:{maximum_size} Merge:{merge}')
    time_ls =  []
    if file_type == '.npz':
        source = src+'/norm_channel_stack'
    if file_type == '.png':
        source = src

    identify_masks(src=source, 
                   object_type=object_type, 
                   model_name=model_name, 
                   batch_size=batch_size, 
                   channels=channels, 
                   diameter=diameter, 
                   minimum_size=minimum_size, 
                   maximum_size=maximum_size, 
                   flow_threshold=30, 
                   cellprob_threshold=cellprob_threshold, 
                   figuresize=25, 
                   cmap='inferno', 
                   refine_masks=refine_masks, 
                   filter_size=filter_size, 
                   filter_dimm=filter_dimm, 
                   remove_border_objects=remove_border_objects, 
                   verbose=verbose, 
                   plot=plot, 
                   merge=merge, 
                   save=save, 
                   start_at=start_at, 
                   file_type=file_type, 
                   net_avg=net_avg, 
                   resample=resample, 
                   timelapse=timelapse)
    
    return print('========== complete ==========')

def make_movies(src, fps):
    for i,npz in enumerate(os.listdir(src)):
        if npz.endswith('.npz'):
            os.makedirs(os.path.dirname(src)+'/movies/', exist_ok=True)
            path = os.path.join(src,npz)
            data = np.load(path)
            images = [img / 65535.0 for img in data['data']]
            images = [(img * 255).astype(np.uint8) for img in images]
            images = [img[..., ::-1] for img in images]
            clip = mpy.ImageSequenceClip(images, fps=fps)
            name = os.path.splitext(npz)[0]
            clip.write_videofile(os.path.dirname(src)+'/movies/'+name+'.mp4', logger=None) #ProgressbarLogger, PrintLogger, None
        print(f'Progress: Movie:{i+1}/{len(os.listdir(src))}', end='\r', flush=True)
    return

def make_mask_movie(src, fps):
    npz_files = [f for f in os.listdir(src) if f.endswith('.npz')]
    os.makedirs(os.path.join(os.path.dirname(src), 'movies'), exist_ok=True)
    
    for i, npz_file in enumerate(npz_files):
        path = os.path.join(src, npz_file)
        data = np.load(path)
        images = data['data'].tolist()
        #images = [(img * 255 / img.max()).astype(np.uint8) for img in images]
        images = [(np.array(img) * 255 / np.array(img).max()).astype(np.uint8) for img in images]
        images = [np.stack([img]*3, axis=-1) for img in images]
        clip = mpy.ImageSequenceClip(images, fps=fps)
        name = os.path.splitext(npz_file)[0]
        output_path = os.path.join(os.path.dirname(src), 'movies', f'{name}.mp4')
        clip.write_videofile(output_path, logger=None)  # ProgressbarLogger, PrintLogger, None
        print(f'Progress: Movie:{i+1}/{len(npz_files)}', end='\r', flush=True)
    return
def mip_all(src, include_first_chan=True):
    # Print a starting message to indicate the beginning of the MIP generation process.
    print('========== generating MIPs ==========')

    # Iterate over each file in the specified directory (src).
    for filename in os.listdir(src):
        # Check if the current file is a NumPy array file (with .npy extension).
        if filename.endswith('.npy'):
            # Load the array from the file.
            array = np.load(os.path.join(src, filename))
            # Normalize the array using custom parameters (q1=2, q2=98).
            array = normalize_to_dtype(array, q1=2, q2=98, percentiles=None)

            if array.ndim != 3: # Check if the array is not 3-dimensional.
                # Log a message indicating a zero array will be generated due to unexpected dimensions.
                print(f"Generating zero array for {filename} due to unexpected dimensions: {array.shape}")
                # Create a zero array with the same height and width as the original array, but with a single depth layer.
                zeros_array = np.zeros((array.shape[0], array.shape[1], 1))
                # Concatenate the original array with the zero array along the depth axis.
                concatenated = np.concatenate([array, zeros_array], axis=2)
            else:
                if include_first_chan:
                    # Compute the MIP for the entire array along the third axis.
                    mip = np.max(array, axis=2)
                else:
                    # Compute the MIP excluding the first layer of the array along the depth axis.
                    mip = np.max(array[:, :, 1:], axis=2)
                # Reshape the MIP to make it 3-dimensional.
                mip = mip[:, :, np.newaxis]
                # Concatenate the MIP with the original array.
                concatenated = np.concatenate([array, mip], axis=2)
            # save
            np.save(os.path.join(src, filename), concatenated)
    return
    
def pivot_counts_table(db_path):
    
    def read_table_to_dataframe(db_path, table_name='object_counts'):
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        # Read the entire table into a pandas DataFrame
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        # Close the connection
        conn.close()
        return df

    def pivot_dataframe(df):
        # Pivot the DataFrame
        pivoted_df = df.pivot(index='file_name', columns='count_type', values='object_count').reset_index()
        # Because the pivot operation can introduce NaN values for missing data,
        # you might want to fill those NaNs with a default value, like 0
        pivoted_df = pivoted_df.fillna(0)
        return pivoted_df
    
    # Read the original 'object_counts' table
    df = read_table_to_dataframe(db_path, 'object_counts')
    # Pivot the DataFrame to have one row per filename and a column for each object type
    pivoted_df = pivot_dataframe(df)
    # Reconnect to the SQLite database to overwrite the 'object_counts' table with the pivoted DataFrame
    conn = sqlite3.connect(db_path)
    # When overwriting, ensure that you drop the existing table or use if_exists='replace' to overwrite it
    pivoted_df.to_sql('object_counts', conn, if_exists='replace', index=False)
    conn.close()

def get_cellpose_channels(mask_channels, nucleus_chann_dim, pathogen_chann_dim, cell_chann_dim):
    cellpose_channels = {}
    if nucleus_chann_dim in mask_channels:
        cellpose_channels['nucleus'] = [0, mask_channels.index(nucleus_chann_dim)]
    if pathogen_chann_dim in mask_channels:
        cellpose_channels['pathogen'] = [0, mask_channels.index(pathogen_chann_dim)]
    if cell_chann_dim in mask_channels:
        cellpose_channels['cell'] = [0, mask_channels.index(cell_chann_dim)]
    return cellpose_channels
    
def preprocess_generate_masks(src, settings={},advanced_settings={}):
	
    settings_dict = {**settings, **advanced_settings}
    settings_df = pd.DataFrame(list(settings_dict.items()), columns=['Key', 'Value'])
    settings_csv = os.path.join(src,'settings','preprocess_generate_masks_settings.csv')
    os.makedirs(os.path.join(src,'settings'), exist_ok=True)
    settings_df.to_csv(settings_csv, index=False)

    channels = settings['channels']
    nucleus_chann_dim = settings['nucleus_channel']
    nucleus_cp_prob = settings['nucleus_CP_prob']
    pathogen_chann_dim = settings['pathogen_channel']
    pathogen_cp_prob = settings['pathogen_CP_prob']
    cell_chann_dim = settings['cell_channel']
    cell_cp_prob = settings['cell_CP_prob']
    metadata_type = settings['metadata_type']
    experiment = settings['experiment']
    magnefication = settings['magnefication']
    
    custom_regex = advanced_settings['custom_regex']
    save = advanced_settings['save']
    plot = advanced_settings['plot']
    examples_to_plot = advanced_settings['examples_to_plot']
    batch_size = advanced_settings['batch_size']
    preprocess = advanced_settings['preprocess']
    masks = advanced_settings['masks']
    timelapse = advanced_settings['timelapse']
    randomize = advanced_settings['randomize']
    remove_background = advanced_settings['remove_background']
    lower_quantile = advanced_settings['lower_quantile']
    merge = advanced_settings['merge']
    normalize_plots = advanced_settings['normalize_plots']
    all_to_mip = advanced_settings['all_to_mip']
    fps = advanced_settings['fps']
    pick_slice = advanced_settings['pick_slice']
    skip_mode = advanced_settings['skip_mode']
    workers = advanced_settings['workers']
    verbose = advanced_settings['verbose']
    
    mask_channels = [nucleus_chann_dim, cell_chann_dim, pathogen_chann_dim]
    mask_channels = [item for item in mask_channels if item is not None]
    
    if preprocess and not masks:
        print(f'WARNING: channels for mask generation are defined when preprocess = True')
    
    if isinstance(merge, bool):
        merge = [merge]*3
    if isinstance(save, bool):
        save = [save]*3

    if preprocess: 
        preprocess_img_data(src,
                            metadata_type=metadata_type,
                            custom_regex=custom_regex,
                            plot=plot,
                            normalize=normalize_plots,
                            mask_channels=mask_channels,
                            batch_size=batch_size,
                            timelapse=timelapse,
                            remove_background=remove_background,
                            lower_quantile=lower_quantile,
                            save_dtype=np.float32,
                            correct_illumination=False,
                            randomize=randomize,
                            generate_movies=timelapse,
                            nr=examples_to_plot,
                            all_to_mip=all_to_mip,
                            fps=fps,
			    pick_slice=pick_slice,
			    skip_mode=skip_mode,
			    settings = settings)
    if masks:

        cellpose_channels = get_cellpose_channels(mask_channels, nucleus_chann_dim, pathogen_chann_dim, cell_chann_dim)

        if cell_chann_dim != None:
            cell_channels = cellpose_channels['cell']
            generate_masks(src,
                           object_type='cell',
                           mag=magnefication,
                           batch_size=batch_size,
                           channels=cell_channels,
                           cellprob_threshold=cell_cp_prob,
                           plot=plot,
                           nr=examples_to_plot,
                           save=save[0],
                           merge=merge[0],
                           verbose=verbose,
                           timelapse=timelapse,
                           file_type='.npz',
                           settings=settings)
            torch.cuda.empty_cache()
        if nucleus_chann_dim != None:
            nucleus_channels = cellpose_channels['nucleus']
            generate_masks(src,
                           object_type='nuclei',
                           mag=magnefication,
                           batch_size=batch_size,
                           channels=nucleus_channels,
                           cellprob_threshold=nucleus_cp_prob,
                           plot=plot,
                           nr=examples_to_plot,
                           save=save[1],
                           merge=merge[1],
                           verbose=verbose,
                           timelapse=timelapse,
                           file_type='.npz',
                           settings=settings)
            torch.cuda.empty_cache()
        if pathogen_chann_dim != None:
            pathogen_channels = cellpose_channels['pathogen']
            generate_masks(src,
                           object_type='pathogen',
                           mag=magnefication,
                           batch_size=batch_size,
                           channels=pathogen_channels,
                           cellprob_threshold=pathogen_cp_prob,
                           plot=plot,
                           nr=examples_to_plot,
                           save=save[2],
                           merge=merge[2],
                           verbose=verbose,
                           timelapse=timelapse,
                           file_type='.npz',
                           settings=settings)
            torch.cuda.empty_cache()
        if os.path.exists(os.path.join(src,'measurements'))
            pivot_counts_table(db_path=os.path.join(src,'measurements', 'measurements.db'))
	#Concatinate stack with masks
        load_and_concatenate_arrays(src, channels, cell_chann_dim, nucleus_chann_dim, pathogen_chann_dim)
        if plot:
            plot_dims = len(channels)
            overlay_channels = [2,1,0]
            cell_mask_dim = nucleus_mask_dim = pathogen_mask_dim = None
            plot_counter = plot_dims
		
            if cell_chann_dim is not None:
                cell_mask_dim = plot_counter
                plot_counter += 1

            if nucleus_chann_dim is not None:
                nucleus_mask_dim = plot_counter
                plot_counter += 1

            if pathogen_chann_dim is not None:
                pathogen_mask_dim = plot_counter
                
            plot_settings = {'include_noninfected':True, 
                             'include_multiinfect':True,
                             'include_multinucleated':True,
                             'remove_background':False,
                             'filter_min_max':None,
                             'channel_dims':channels,
                             'backgrounds':[100,100,100,100],
                             'cell_mask_dim':cell_mask_dim,
                             'nucleus_mask_dim':nucleus_mask_dim,
                             'pathogen_mask_dim':pathogen_mask_dim,
                             'overlay_chans':[0,2,3],
                             'outline_thickness':3,
                             'outline_color':'gbr',
                             'overlay_chans':overlay_channels,
                             'overlay':True,
                             'normalization_percentiles':[1,99],
                             'normalize':True,
                             'print_object_number':True,
                             'nr':examples_to_plot,
                             'figuresize':20,
                             'cmap':'inferno',
                             'verbose':True}

            plot_merged(src=os.path.join(src,'merged'), settings=plot_settings)
            
    torch.cuda.empty_cache()
    gc.collect()
    return

def read_db(db_loc, tables):
    conn = sqlite3.connect(db_loc)
    dfs = []
    for table in tables:
        query = f'SELECT * FROM {table}'
        df = pd.read_sql_query(query, conn)
        dfs.append(df)
    conn.close()
    return dfs

def well_to_row_col(well):
    row_dict = {char:idx+1 for idx, char in enumerate('ABCDEFGHIJKLMNOP')}
    row = row_dict[well[0]]
    col = int(well[1:])
    return pd.Series([row, col])

def load_and_extract_data(db_loc):
    dfs = read_db(db_loc, tables=['pathogen_nuclei'])
    df = dfs[0]
    df[['plate', 'well', 'field', 'obj']] = df['filename'].str.split('_', expand=True)
    df['obj'] = df['obj'].str.replace('.png', '')
    #df[['plate', 'well', 'field', 'obj']] = df[['plate', 'well', 'field', 'obj']].astype(str)
    df[['row', 'col']] = df['well'].apply(well_to_row_col)
    prefix = ['p','r','c','f','o']
    for idx, val in enumerate(['plate', 'row', 'col', 'field', 'obj']):
        df[val] = df[val].apply(lambda x: prefix[idx] + str(x))
    
    unique_plates = df['plate'].unique().tolist()
    unique_wells = df['well'].unique().tolist()
    unique_rows = df['row'].unique().tolist()
    unique_cols = df['col'].unique().tolist()
    print(f'Data from {len(unique_plates)} plates with {len(unique_wells)} wells in {len(unique_rows)} and {len(unique_cols)} columns')
    return df

# Function to apply to each row
def map_values(row, dict_, type_='col'):
    for values, cols in dict_.items():
        if row[type_] in cols:
            return values
    return None

def split_data(df, group_by, object_type):
    
    df['prcfo'] = df['prcf'] + '_' + df[object_type]
    #df = df.drop([object_type], axis=1)
    # Set 'prcfo' as the index for both dataframes
    df = df.set_index(group_by, inplace=False)
    
    # Split the dataframe into numeric and non-numeric parts
    df_numeric = df.select_dtypes(include=np.number)
    df_non_numeric = df.select_dtypes(exclude=np.number)
    
    # Group by index (note that the result will be a GroupBy object)
    grouped_numeric = df_numeric.groupby(df_numeric.index).mean()
    grouped_non_numeric = df_non_numeric.groupby(df_non_numeric.index).first()
    return pd.DataFrame(grouped_numeric), pd.DataFrame(grouped_non_numeric)

def read_and_merge_data(locs, tables, verbose=False, include_multinucleated=False, include_multiinfected=False, include_noninfected=False):
    #Extract plate DataFrames
    all_dfs = []
    for loc in locs:
        db_dfs = read_db(loc, tables)
        all_dfs.append(db_dfs)
    
    #Extract Tables from DataFrames and concatinate rows
    for i, dfs in enumerate(all_dfs):
        if 'cell' in tables:
            cell = dfs[0]
            print(f'plate: {i+1} cells:{len(cell)}')

        if 'nucleus' in tables:
            nucleus = dfs[1]
            print(f'plate: {i+1} nuclei:{len(nucleus)} ')

        if 'pathogen' in tables:
            pathogen = dfs[2]
            
            print(f'plate: {i+1} pathogens:{len(pathogen)}')
        if 'cytoplasm' in tables:
            if not 'pathogen' in tables:
                cytoplasm = dfs[2]
            else:
                cytoplasm = dfs[3]
            print(f'plate: {i+1} cytoplasms: {len(cytoplasm)}')

        if i > 0:
            if 'cell' in tables:
                cells = pd.concat([cells, cell], axis = 0)
            if 'nucleus' in tables:
                nuclei = pd.concat([nuclei, nucleus], axis = 0)
            if 'pathogen' in tables:
                pathogens = pd.concat([pathogens, pathogen], axis = 0)
            if 'cytoplasm' in tables:
                cytoplasms = pd.concat([cytoplasms, cytoplasm], axis = 0)
        else:
            if 'cell' in tables:
                cells = cell.copy()
            if 'nucleus' in tables:
                nuclei = nucleus.copy()
            if 'pathogen' in tables:
                pathogens = pathogen.copy()
            if 'cytoplasm' in tables:
                cytoplasms = cytoplasm.copy()
    
    #Add an o in front of all object and cell lables to convert them to strings
    if 'cell' in tables:
        cells = cells.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        cells = cells.assign(prcfo = lambda x: x['prcf'] + '_' + x['object_label'])
        cells_g_df, metadata = split_data(cells, 'prcfo', 'object_label')
        print(f'cells: {len(cells)}')
        print(f'cells grouped: {len(cells_g_df)}')
    if 'cytoplasm' in tables:
        cytoplasms = cytoplasms.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        cytoplasms = cytoplasms.assign(prcfo = lambda x: x['prcf'] + '_' + x['object_label'])
        cytoplasms_g_df, _ = split_data(cytoplasms, 'prcfo', 'object_label')
        merged_df = cells_g_df.merge(cytoplasms_g_df, left_index=True, right_index=True)
        print(f'cytoplasms: {len(cytoplasms)}')
        print(f'cytoplasms grouped: {len(cytoplasms_g_df)}')
    if 'nucleus' in tables:
        nuclei = nuclei.dropna(subset=['cell_id'])
        nuclei = nuclei.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        nuclei = nuclei.assign(cell_id=lambda x: 'o' + x['cell_id'].astype(int).astype(str))
        nuclei = nuclei.assign(prcfo = lambda x: x['prcf'] + '_' + x['cell_id'])
        nuclei['nuclei_prcfo_count'] = nuclei.groupby('prcfo')['prcfo'].transform('count')
        if include_multinucleated == False:
            #nuclei = nuclei[~nuclei['prcfo'].duplicated()]
            nuclei = nuclei[nuclei['nuclei_prcfo_count']==1]
        nuclei_g_df, _ = split_data(nuclei, 'prcfo', 'cell_id')
        print(f'nuclei: {len(nuclei)}')
        print(f'nuclei grouped: {len(nuclei_g_df)}')
        if 'cytoplasm' in tables:
            merged_df = merged_df.merge(nuclei_g_df, left_index=True, right_index=True)
        else:
            merged_df = cells_g_df.merge(nuclei_g_df, left_index=True, right_index=True)
    if 'pathogen' in tables:
        pathogens = pathogens.dropna(subset=['cell_id'])
        pathogens = pathogens.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        pathogens = pathogens.assign(cell_id=lambda x: 'o' + x['cell_id'].astype(int).astype(str))
        pathogens = pathogens.assign(prcfo = lambda x: x['prcf'] + '_' + x['cell_id'])
        pathogens['pathogen_prcfo_count'] = pathogens.groupby('prcfo')['prcfo'].transform('count')
        if include_noninfected == False:
            pathogens = pathogens[pathogens['pathogen_prcfo_count']>=1]
        if include_multiinfected == False:
            pathogens = pathogens[pathogens['pathogen_prcfo_count']<=1]
        pathogens_g_df, _ = split_data(pathogens, 'prcfo', 'cell_id')
        print(f'pathogens: {len(pathogens)}')
        print(f'pathogens grouped: {len(pathogens_g_df)}')
        merged_df = merged_df.merge(pathogens_g_df, left_index=True, right_index=True)
    
    #Add prc column (plate row column)
    metadata = metadata.assign(prc = lambda x: x['plate'] + '_' + x['row'] + '_' +x['col'])

    #Count cells per well
    cells_well = pd.DataFrame(metadata.groupby('prc')['object_label'].nunique())

    cells_well.reset_index(inplace=True)
    cells_well.rename(columns={'object_label': 'cells_per_well'}, inplace=True)
    metadata = pd.merge(metadata, cells_well, on='prc', how='inner', suffixes=('', '_drop_col'))
    object_label_cols = [col for col in metadata.columns if '_drop_col' in col]
    metadata.drop(columns=object_label_cols, inplace=True)

    #Add prcfo column (plate row column field object)
    metadata = metadata.assign(prcfo = lambda x: x['plate'] + '_' + x['row'] + '_' +x['col']+ '_' +x['field']+ '_' +x['object_label'])
    metadata.set_index('prcfo', inplace=True)

    merged_df = metadata.merge(merged_df, left_index=True, right_index=True)
    
    merged_df = merged_df.dropna(axis=1)
    
    print(f'Generated dataframe with: {len(merged_df.columns)} columns and {len(merged_df)} rows')
    
    obj_df_ls = []
    if 'cell' in tables:
        obj_df_ls.append(cells)
    if 'cytoplasm' in tables:
        obj_df_ls.append(cytoplasms)
    if 'nucleus' in tables:
        obj_df_ls.append(nuclei)
    if 'pathogen' in tables:
        obj_df_ls.append(pathogens)
        
    return merged_df, obj_df_ls

def filter_measurement_data(df, background, min_cells_per_well=1):
    f0 = len(df)
    df = df[df['nucleus_channel_0_mean_intensity'] >= background]
    f1 = len(df)
    df = df[df['cells_per_well'] >= min_cells_per_well]
    fs = len(df)
    df = df[df['nucleus_channel_3_mean_intensity'] >= background]
    f2 = len(df)
    df = df[df['nucleus_eccentricity'] <= 0.98] # eccentricity 0=round 1=parabola
    f3 = len(df)
    df = df[df['pathogen_channel_2_mean_intensity'] >= background]
    f4 = len(df)
    df = df[df['cell_channel_3_mean_intensity'] >= background]
    f5 = len(df)
    print(f'Origional: {f0} Size:{fs} Nucleus:{f1, f2, f3} pathogen:{f4} Cell:{f5}')
    return df

#def analyze_leakage(df):
#    pathogen_metadata_ls = ['rh', 'dgra8']
#    pathogen_loc_ls = [['c2','c3','c4'],['c5','c6','c7']]
#    treatment_ls = ['cm']
#    treatment_loc_ls = [['c2','c3','c4','c5', 'c6', 'c7']]
#    col_names = ['col','col','col']
#    df = annotate_conditions(df, 
#                             cells=['HeLa'],
#                             cell_loc=None,
#                             pathogens=pathogen_metadata_ls,
#                             pathogen_loc=pathogen_loc_ls,
#                             treatments=treatment_ls,
#                             treatment_loc=treatment_loc_ls,
#                             types=col_names)
#
#    plt.figure(figsize=(5,5))
#    sns.barplot(data=df, x='condition', y='pathogen_channel_2_mean_intensity', hue='pathogen', capsize=.1, ci='sd', dodge=False)
#    plt.xlabel('Condition')
#    plt.ylabel('pathogen_DsRed_mean_intensity')
#    plt.show()
#    
#    plt.figure(figsize=(5,5))
#    sns.barplot(data=df, x='condition', y='cytoplasm_channel_2_mean_intensity', hue='pathogen', capsize=.1, ci='sd', dodge=False)
#    plt.xlabel('Condition')
#    plt.ylabel('cytoplasm_DsRed_mean_intensity')
#    plt.show()
#    
#    plt.figure(figsize=(5,5))
#    sns.barplot(data=df, x='condition', y='rec', hue='pathogen', capsize=.1, ci='sd', dodge=False)
#    plt.xlabel('Condition')
#    plt.ylabel('pathogen/cytoplasm (DsRed mean intensity)')
#    plt.show()
#    cols_to_melt = ['pathogen_rad_dist_channel_2_bin_0', 'pathogen_rad_dist_channel_2_bin_1', 
#                    'pathogen_rad_dist_channel_2_bin_2', 'pathogen_rad_dist_channel_2_bin_3', 
#                    'pathogen_rad_dist_channel_2_bin_4', 'pathogen_rad_dist_channel_2_bin_5',
#                    'pathogen_channel_2_outside_5_percentile', 'pathogen_channel_2_outside_10_percentile', 
#                    'pathogen_channel_2_outside_25_percentile', 'pathogen_channel_2_outside_50_percentile', 
#                    'pathogen_channel_2_outside_75_percentile', 'pathogen_channel_2_outside_85_percentile', 
#                    'pathogen_channel_2_outside_95_percentile']
#    
#    id_vars = [col for col in df.columns if col not in cols_to_melt]
#    df_melt = df.melt(id_vars=id_vars, value_vars=cols_to_melt, var_name='Concentration', value_name='Value')
#    df_melt = df_melt[df_melt['pathogen'].isin(['rh', 'dgra8'])]
#
#    plt.figure(figsize=(10, 6))
#    sns.barplot(data=df_melt, x='Concentration', y='Value', hue='pathogen', capsize=.1, ci='sd')
#    plt.title('Bar graph comparing Concentrations for Conditions rh and dgra8')
#    plt.xticks(rotation=90)
#    plt.show()
#    
#    return df

def group_by_well(df):
    numeric_cols = df._get_numeric_data().columns
    non_numeric_cols = df.select_dtypes(include=['object']).columns

    # Apply mean function to numeric columns and first to non-numeric
    df_grouped = df.groupby(['plate', 'row', 'col']).agg({**{col: np.mean for col in numeric_cols}, **{col: 'first' for col in non_numeric_cols}})
    return df_grouped

def annotate_conditions(df, cells=['HeLa'], cell_loc=None, pathogens=['rh'], pathogen_loc=None, treatments=['cm'], treatment_loc=None, types = ['col','col','col']):
    if cell_loc is None:
        df['host_cells'] = cells[0]
    else:
        cells_dict = dict(zip(cells, cell_loc))
        df['host_cells'] = df.apply(lambda row: map_values(row, cells_dict, type_=types[0]), axis=1)
    if pathogen_loc is None:
        if pathogens != None:
            df['pathogen'] = 'none'
    else:
        pathogens_dict = dict(zip(pathogens, pathogen_loc))
        df['pathogen'] = df.apply(lambda row: map_values(row, pathogens_dict, type_=types[1]), axis=1)
    if treatment_loc is None:
        df['treatment'] = 'cm'
    else:
        treatments_dict = dict(zip(treatments, treatment_loc))
        df['treatment'] = df.apply(lambda row: map_values(row, treatments_dict, type_=types[2]), axis=1)
    if pathogens != None:
        df['condition'] = df['pathogen']+'_'+df['treatment']
    else:
        df['condition'] = df['treatment']
    
    return df

def calculate_percentiles(x):
    return pd.qcut(x, q=np.linspace(0, 1, 11), labels=['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])

def calculate_slope(df, channel, object_type):
    cols = [col for col in df.columns if f'{object_type}_rad_dist_channel_{channel}_bin_' in col]
    x = np.arange(len(cols))
    slopes = df[cols].apply(lambda row: np.polyfit(x, row, 1)[0], axis=1)
    return slopes

def calculate_recruitment(df, channel):

    df['pathogen_cell_mean_mean'] = df[f'pathogen_channel_{channel}_mean_intensity']/df[f'cell_channel_{channel}_mean_intensity']
    df['pathogen_cytoplasm_mean_mean'] = df[f'pathogen_channel_{channel}_mean_intensity']/df[f'cytoplasm_channel_{channel}_mean_intensity']
    df['pathogen_nucleus_mean_mean'] = df[f'pathogen_channel_{channel}_mean_intensity']/df[f'nucleus_channel_{channel}_mean_intensity']
    
    df['pathogen_cell_q75_mean'] = df[f'pathogen_channel_{channel}_percentile_75']/df[f'cell_channel_{channel}_mean_intensity']
    df['pathogen_cytoplasm_q75_mean'] = df[f'pathogen_channel_{channel}_percentile_75']/df[f'cytoplasm_channel_{channel}_mean_intensity']
    df['pathogen_nucleus_q75_mean'] = df[f'pathogen_channel_{channel}_percentile_75']/df[f'nucleus_channel_{channel}_mean_intensity']
    
    df['pathogen_outside_cell_mean_mean'] = df[f'pathogen_channel_{channel}_outside_mean']/df[f'cell_channel_{channel}_mean_intensity']
    df['pathogen_outside_cytoplasm_mean_mean'] = df[f'pathogen_channel_{channel}_outside_mean']/df[f'cytoplasm_channel_{channel}_mean_intensity']
    df['pathogen_outside_nucleus_mean_mean'] = df[f'pathogen_channel_{channel}_outside_mean']/df[f'nucleus_channel_{channel}_mean_intensity']
    
    df['pathogen_outside_cell_q75_mean'] = df[f'pathogen_channel_{channel}_outside_75_percentile']/df[f'cell_channel_{channel}_mean_intensity']
    df['pathogen_outside_cytoplasm_q75_mean'] = df[f'pathogen_channel_{channel}_outside_75_percentile']/df[f'cytoplasm_channel_{channel}_mean_intensity']
    df['pathogen_outside_nucleus_q75_mean'] = df[f'pathogen_channel_{channel}_outside_75_percentile']/df[f'nucleus_channel_{channel}_mean_intensity']

    df['pathogen_periphery_cell_mean_mean'] = df[f'pathogen_channel_{channel}_periphery_mean']/df[f'cell_channel_{channel}_mean_intensity']
    df['pathogen_periphery_cytoplasm_mean_mean'] = df[f'pathogen_channel_{channel}_periphery_mean']/df[f'cytoplasm_channel_{channel}_mean_intensity']
    df['pathogen_periphery_nucleus_mean_mean'] = df[f'pathogen_channel_{channel}_periphery_mean']/df[f'nucleus_channel_{channel}_mean_intensity']
    
    channels = [0,1,2,3]
    object_type = 'pathogen'
    for chan in channels:
        df[f'{object_type}_slope_channel_{chan}'] = 1
    
    object_type = 'nucleus'
    for chan in channels:
        df[f'{object_type}_slope_channel_{chan}'] = 1
    
    for chan in channels:
        df[f'nucleus_coordinates_{chan}'] = df[[f'nucleus_channel_{chan}_centroid_weighted_local-0', f'nucleus_channel_{chan}_centroid_weighted_local-1']].values.tolist()
        df[f'pathogen_coordinates_{chan}'] = df[[f'pathogen_channel_{chan}_centroid_weighted_local-0', f'pathogen_channel_{chan}_centroid_weighted_local-1']].values.tolist()
        df[f'cell_coordinates_{chan}'] = df[[f'cell_channel_{chan}_centroid_weighted_local-0', f'cell_channel_{chan}_centroid_weighted_local-1']].values.tolist()
        df[f'cytoplasm_coordinates_{chan}'] = df[[f'cytoplasm_channel_{chan}_centroid_weighted_local-0', f'cytoplasm_channel_{chan}_centroid_weighted_local-1']].values.tolist()

        df[f'pathogen_cell_distance_channel_{chan}'] = df.apply(lambda row: np.sqrt((row[f'pathogen_coordinates_{chan}'][0] - row[f'cell_coordinates_{chan}'][0])**2 + 
                                                      (row[f'pathogen_coordinates_{chan}'][1] - row[f'cell_coordinates_{chan}'][1])**2), axis=1)
        df[f'nucleus_cell_distance_channel_{chan}'] = df.apply(lambda row: np.sqrt((row[f'nucleus_coordinates_{chan}'][0] - row[f'cell_coordinates_{chan}'][0])**2 + 
                                                      (row[f'nucleus_coordinates_{chan}'][1] - row[f'cell_coordinates_{chan}'][1])**2), axis=1)
    return df

def plot_controls(df, mask_chans, channel_of_interest, figuresize=5):
    mask_chans.append(channel_of_interest)
    if len(mask_chans) == 4:
        mask_chans = [0,1,2,3]
    if len(mask_chans) == 3:
        mask_chans = [0,1,2]
    if len(mask_chans) == 2:
        mask_chans = [0,1]
    if len(mask_chans) == 1:
        mask_chans = [0]
    controls_cols = []
    for chan in mask_chans:
	
        controls_cols_c = []
        controls_cols_c.append(f'cell_channel_{chan}_mean_intensity')
        controls_cols_c.append(f'nucleus_channel_{chan}_mean_intensity')
        controls_cols_c.append(f'pathogen_channel_{chan}_mean_intensity')
        controls_cols_c.append(f'cytoplasm_channel_{chan}_mean_intensity')
        controls_cols.append(controls_cols_c)
    
    unique_conditions = df['condition'].unique().tolist()

    if len(unique_conditions) ==1:
        unique_conditions=unique_conditions+unique_conditions
    
    fig, axes = plt.subplots(len(unique_conditions), len(mask_chans)+1, figsize=(figuresize*len(mask_chans), figuresize*len(unique_conditions)))

    # Define RGB color tuples (scaled to 0-1 range)
    color_list = [(55/255, 155/255, 155/255), 
                  (155/255, 55/255, 155/255), 
                  (55/255, 155/255, 255/255), 
                  (255/255, 55/255, 155/255)]

    for idx_condition, condition in enumerate(unique_conditions):
        df_temp = df[df['condition'] == condition]
        for idx_channel, control_cols_c in enumerate(controls_cols):
            data = []
            std_dev = []
            for control_col in control_cols_c:
                if control_col in df_temp.columns:
                    mean_intensity = df_temp[control_col].mean()
                    mean_intensity = 0 if np.isnan(mean_intensity) else mean_intensity
                    data.append(mean_intensity)
                    std_dev.append(df_temp[control_col].std())

            current_axis = axes[idx_condition][idx_channel]
            current_axis.bar(["cell", "nucleus", "pathogen", "cytoplasm"], data, yerr=std_dev, 
                             capsize=4, color=color_list)
            current_axis.set_xlabel('Component')
            current_axis.set_ylabel('Mean Intensity')
            current_axis.set_title(f'Condition: {condition} - Channel {idx_channel}')
    plt.tight_layout()
    plt.show()
        
def plot_recruitment(df, df_type, channel_of_interest, target, columns=[], figuresize=50):
    
    color_list = [(55/255, 155/255, 155/255), 
                  (155/255, 55/255, 155/255), 
                  (55/255, 155/255, 255/255), 
                  (255/255, 55/255, 155/255)]
    
    sns.set_palette(sns.color_palette(color_list))
    font = figuresize/2
    width=figuresize
    height=figuresize/4
    
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(width, height))
    sns.barplot(ax=axes[0], data=df, x='condition', y=f'cell_channel_{channel_of_interest}_mean_intensity', hue='pathogen', capsize=.1, ci='sd', dodge=False)
    axes[0].set_xlabel(f'pathogen {df_type}', fontsize=font)
    axes[0].set_ylabel(f'cell_channel_{channel_of_interest}_mean_intensity', fontsize=font)
    
    sns.barplot(ax=axes[1], data=df, x='condition', y=f'nucleus_channel_{channel_of_interest}_mean_intensity', hue='pathogen', capsize=.1, ci='sd', dodge=False)
    axes[1].set_xlabel(f'pathogen {df_type}', fontsize=font)
    axes[1].set_ylabel(f'nucleus_channel_{channel_of_interest}_mean_intensity', fontsize=font)
    
    sns.barplot(ax=axes[2], data=df, x='condition', y=f'cytoplasm_channel_{channel_of_interest}_mean_intensity', hue='pathogen', capsize=.1, ci='sd', dodge=False)
    axes[2].set_xlabel(f'pathogen {df_type}', fontsize=font)
    axes[2].set_ylabel(f'cytoplasm_channel_{channel_of_interest}_mean_intensity', fontsize=font)
    
    sns.barplot(ax=axes[3], data=df, x='condition', y=f'pathogen_channel_{channel_of_interest}_mean_intensity', hue='pathogen', capsize=.1, ci='sd', dodge=False)
    axes[3].set_xlabel(f'pathogen {df_type}', fontsize=font)
    axes[3].set_ylabel(f'pathogen_channel_{channel_of_interest}_mean_intensity', fontsize=font)
    
    axes[0].legend_.remove()
    axes[1].legend_.remove()
    axes[2].legend_.remove()
    axes[3].legend_.remove()
    
    handles, labels = axes[3].get_legend_handles_labels()
    axes[3].legend(handles, labels, bbox_to_anchor=(1.05, 0.5), loc='center left')
    for i in [0,1,2,3]:
        axes[i].tick_params(axis='both', which='major', labelsize=font)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
	    
    plt.tight_layout()
    plt.show()
    
    columns = columns + ['pathogen_cytoplasm_mean_mean', 'pathogen_cytoplasm_q75_mean', 'pathogen_periphery_cytoplasm_mean_mean', 'pathogen_outside_cytoplasm_mean_mean', 'pathogen_outside_cytoplasm_q75_mean']
    columns = columns + [f'pathogen_slope_channel_{channel_of_interest}', f'pathogen_cell_distance_channel_{channel_of_interest}', f'nucleus_cell_distance_channel_{channel_of_interest}']

    width = figuresize*2
    columns_per_row = math.ceil(len(columns) / 2)
    height = (figuresize*2)/columns_per_row

    fig, axes = plt.subplots(nrows=2, ncols=columns_per_row, figsize=(width, height * 2))
    axes = axes.flatten()

    print(f'{columns}')

    for i, col in enumerate(columns):

        ax = axes[i]
        sns.barplot(ax=ax, data=df, x='condition', y=f'{col}', hue='pathogen', capsize=.1, ci='sd', dodge=False)
        ax.set_xlabel(f'pathogen {df_type}', fontsize=font)
        ax.set_ylabel(f'{col}', fontsize=int(font*2))
        ax.legend_.remove()
        ax.tick_params(axis='both', which='major', labelsize=font)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        if i <= 5:
            ax.set_ylim(1, None)

    for i in range(len(columns), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def plot_data(df, csv_loc, category_order, figuresize=50, y_min=1):

    #df = pd.read_csv(csv_loc)

    color_list = [(55/255, 155/255, 155/255), 
              (155/255, 55/255, 155/255), 
              (55/255, 155/255, 255/255), 
              (255/255, 55/255, 155/255)]
    
    columns = ['pathogen_cytoplasm_mean_mean', 'pathogen_cytoplasm_q75_mean', 'pathogen_outside_cytoplasm_mean_mean', 'pathogen_outside_cytoplasm_q75_mean','pathogen_periphery_cytoplasm_mean_mean']

    width = figuresize*2
    columns_per_row = math.ceil(len(columns) / 2)
    height = (figuresize*2)/columns_per_row
    font = figuresize/2
    fig, axes = plt.subplots(nrows=2, ncols=columns_per_row, figsize=(width, height * 2))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        sns.barplot(ax=ax, data=df, x='condition', y=f'{col}', hue='pathogen', capsize=.1, ci='sd', dodge=False, order=category_order, palette=sns.color_palette(color_list))
        ax.set_ylabel(f'{col}', fontsize=int(font*2))
        ax.legend_.remove()
        ax.tick_params(axis='both', which='major', labelsize=font)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        if i <= 5:
            ax.set_ylim(y_min, None)

    for i in range(len(columns), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    results_dir = os.path.dirname(csv_loc)
    csv_name = os.path.basename(csv_loc)
    csv_name, extension = os.path.splitext(csv_name)
    pdf_name_bar = csv_name + '_bar.pdf'
    pdf_name_jitter = csv_name + '_jitter.pdf'

    fig.savefig(os.path.join(results_dir, pdf_name_bar))

    fig, axes = plt.subplots(nrows=2, ncols=columns_per_row, figsize=(width, height * 2))
    axes = axes.flatten()
    for i, col in enumerate(columns):

        ax = axes[i]
        sns.stripplot(ax=ax, data=df, x='condition', y=f'{col}', hue='pathogen', dodge=False, jitter=True, order=category_order, palette=sns.color_palette(color_list))
        ax.set_ylabel(f'{col}', fontsize=int(font*2))
        ax.legend_.remove()
        ax.tick_params(axis='both', which='major', labelsize=font)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        if i <= 5:
            ax.set_ylim(y_min, None)

    for i in range(len(columns), len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(results_dir, pdf_name_jitter))

def analyze_recruitment(src, metadata_settings, advanced_settings):
    
    def object_filter(df, object_type, size_range, intensity_range, mask_chans, mask_chan):
        if not size_range is None:
            if isinstance(size_range, list):
                if isinstance(size_range[0], int): 
                    df = df[df[f'{object_type}_area'] > size_range[0]]
                    print(f'After {object_type} minimum area filter: {len(df)}')
                if isinstance(size_range[1], int):
                    df = df[df[f'{object_type}_area'] < size_range[1]]
                    print(f'After {object_type} maximum area filter: {len(df)}')
        if not intensity_range is None:
            if isinstance(intensity_range, list):
                if isinstance(intensity_range[0], int):
                    df = df[df[f'{object_type}_channel_{mask_chans[mask_chan]}_mean_intensity'] > intensity_range[0]]
                    print(f'After {object_type} minimum mean intensity filter: {len(df)}')
                if isinstance(intensity_range[1], int):
                    df = df[df[f'{object_type}_channel_{mask_chans[mask_chan]}_mean_intensity'] < intensity_range[1]]
                    print(f'After {object_type} maximum mean intensity filter: {len(df)}')
        return df
    
    def results_to_csv(src, df, df_well):
        cells = df
        wells = df_well
        results_loc = src+'/results'
        wells_loc = results_loc+'/wells.csv'
        cells_loc = results_loc+'/cells.csv'
        os.makedirs(results_loc, exist_ok=True)
        wells.to_csv(wells_loc, index=True, header=True)
        cells.to_csv(cells_loc, index=True, header=True)
        return cells, wells
    
    settings_dict = {**metadata_settings, **advanced_settings}
    settings_df = pd.DataFrame(list(settings_dict.items()), columns=['Key', 'Value'])
    settings_csv = os.path.join(src,'settings','analyze_settings.csv')
    os.makedirs(os.path.join(src,'settings'), exist_ok=True)
    settings_df.to_csv(settings_csv, index=False)

    # metadata settings
    target = metadata_settings['target']
    cell_types = metadata_settings['cell_types']
    cell_plate_metadata = metadata_settings['cell_plate_metadata']
    pathogen_types = metadata_settings['pathogen_types']
    pathogen_plate_metadata = metadata_settings['pathogen_plate_metadata']
    treatments = metadata_settings['treatments']
    treatment_plate_metadata = metadata_settings['treatment_plate_metadata']
    metadata_types = metadata_settings['metadata_types']
    channel_dims = metadata_settings['channel_dims']
    cell_chann_dim = metadata_settings['cell_chann_dim']
    cell_mask_dim = metadata_settings['cell_mask_dim']
    nucleus_chann_dim = metadata_settings['nucleus_chann_dim']
    nucleus_mask_dim = metadata_settings['nucleus_mask_dim']
    pathogen_chann_dim = metadata_settings['pathogen_chann_dim']
    pathogen_mask_dim = metadata_settings['pathogen_mask_dim']
    channel_of_interest = metadata_settings['channel_of_interest']
    
    # Advanced settings
    plot = advanced_settings['plot']
    plot_nr = advanced_settings['plot_nr']
    plot_control = advanced_settings['plot_control']
    figuresize = advanced_settings['figuresize']
    remove_background = advanced_settings['remove_background']
    backgrounds = advanced_settings['backgrounds']
    include_noninfected = advanced_settings['include_noninfected']
    include_multiinfected = advanced_settings['include_multiinfected']
    include_multinucleated = advanced_settings['include_multinucleated']
    cells_per_well = advanced_settings['cells_per_well']
    pathogen_size_range = advanced_settings['pathogen_size_range']
    nucleus_size_range = advanced_settings['nucleus_size_range']
    cell_size_range = advanced_settings['cell_size_range']
    pathogen_intensity_range = advanced_settings['pathogen_intensity_range']
    nucleus_intensity_range = advanced_settings['nucleus_intensity_range']
    cell_intensity_range = advanced_settings['cell_intensity_range']
    target_intensity_min = advanced_settings['target_intensity_min']
    
    print(f'Cell(s): {cell_types}, in {cell_plate_metadata}')
    print(f'Pathogen(s): {pathogen_types}, in {pathogen_plate_metadata}')
    print(f'Treatment(s): {treatments}, in {treatment_plate_metadata}')
    
    mask_dims=[cell_mask_dim,nucleus_mask_dim,pathogen_mask_dim]
    mask_chans=[nucleus_chann_dim, pathogen_chann_dim, cell_chann_dim]

    if isinstance(metadata_types, str):
        metadata_types = [metadata_types, metadata_types, metadata_types]
    if isinstance(metadata_types, list):
        if len(metadata_types) < 3:
            metadata_types = [metadata_types[0], metadata_types[0], metadata_types[0]]
            print(f'WARNING: setting metadata types to first element times 3: {metadata_types}. To avoid this behaviour, set metadata_types to a list with 3 elements. Elements should be col row or plate.')
        else:
            metadata_types = metadata_types
    
    if isinstance(backgrounds, (int,float)):
        backgrounds = [backgrounds, backgrounds, backgrounds, backgrounds]

    sns.color_palette("mako", as_cmap=True)
    print(f'channel:{channel_of_interest} = {target}')
    overlay_channels = [0, 1, 2, 3]
    overlay_channels.remove(channel_of_interest)
    overlay_channels.reverse()
    
    db_loc = [src+'/measurements/measurements.db']
    tables = ['cell', 'nucleus', 'pathogen','cytoplasm']
    df, object_dfs = read_and_merge_data(db_loc, 
                                         tables, 
                                         verbose=True, 
                                         include_multinucleated=include_multinucleated, 
                                         include_multiinfected=include_multiinfected, 
                                         include_noninfected=include_noninfected)
    
    df = annotate_conditions(df, 
                             cells=cell_types, 
                             cell_loc=cell_plate_metadata, 
                             pathogens=pathogen_types,
                             pathogen_loc=pathogen_plate_metadata,
                             treatments=treatments, 
                             treatment_loc=treatment_plate_metadata,
                             types=metadata_types)
    
    df = df.dropna(subset=['condition'])
    print(f'After dropping non-annotated wells: {len(df)} rows')
    files = df['file_name'].tolist()
    files = [item + '.npy' for item in files]
    random.shuffle(files)
    max_ = 100**10

    if plot:
        plot_settings = {'include_noninfected':include_noninfected, 
                         'include_multiinfected':include_multiinfected,
                         'include_multinucleated':include_multinucleated,
                         'remove_background':remove_background,
                         'filter_min_max':[[cell_size_min,max_],[nucleus_size_min,max_],[pathogen_size_min,max_],[0,max_]],
                         'channel_dims':channel_dims,
                         'backgrounds':backgrounds,
                         'cell_mask_dim':mask_dims[0],
                         'nucleus_mask_dim':mask_dims[1],
                         'pathogen_mask_dim':mask_dims[2],
                         'overlay_chans':[0,2,3],
                         'outline_thickness':3,
                         'outline_color':'gbr',
                         'overlay_chans':overlay_channels,
                         'overlay':True,
                         'normalization_percentiles':[1,99],
                         'normalize':True,
                         'print_object_number':True,
                         'nr':plot_nr,
                         'figuresize':20,
                         'cmap':'inferno',
                         'verbose':True}
        
    if os.path.exists(os.path.join(src,'merged')):
        plot_merged(src=os.path.join(src,'merged'), settings=plot_settings)
    
    if not cell_chann_dim is None:
        df = object_filter(df, object_type='cell', size_range=cell_size_range, intensity_range=cell_intensity_range, mask_chans=mask_chans, mask_chan=0)
        if not target_intensity_min is None:
            df = df[df[f'cell_channel_{channel_of_interest}_percentile_95'] > target_intensity_min]
            print(f'After channel {channel_of_interest} filtration', len(df))
    if not nucleus_chann_dim is None:
        df = object_filter(df, object_type='nucleus', size_range=nucleus_size_range, intensity_range=nucleus_intensity_range, mask_chans=mask_chans, mask_chan=1)
    if not pathogen_chann_dim is None:
        df = object_filter(df, object_type='pathogen', size_range=pathogen_size_range, intensity_range=pathogen_intensity_range, mask_chans=mask_chans, mask_chan=2)
       
    df['recruitment'] = df[f'pathogen_channel_{channel_of_interest}_mean_intensity']/df[f'cytoplasm_channel_{channel_of_interest}_mean_intensity']
    for chan in channel_dims:
        df = calculate_recruitment(df, channel=chan)
    print(f'calculated recruitment for: {len(df)} rows')
    df_well = group_by_well(df)
    print(f'found: {len(df_well)} wells')
    
    df_well = df_well[df_well['cells_per_well'] >= cells_per_well]
    prc_list = df_well['prc'].unique().tolist()
    df = df[df['prc'].isin(prc_list)]
    print(f'After cells per well filter: {len(df)} cells in {len(df_well)} wells left wth threshold {cells_per_well}')
    
    if plot_control:
        plot_controls(df, mask_chans, channel_of_interest, figuresize=5)

    print(f'PV level: {len(df)} rows')
    plot_recruitment(df=df, df_type='by PV', channel_of_interest=channel_of_interest, target=target, figuresize=figuresize)
    print(f'well level: {len(df_well)} rows')
    plot_recruitment(df=df_well, df_type='by well', channel_of_interest=channel_of_interest, target=target, figuresize=figuresize)
    cells,wells = results_to_csv(src, df, df_well)
    return [cells,wells]

def filter_pathogen_nuclei(df, upper_quantile=0.95, plot=True):
    upper_int = df['mean_intensity'].quantile(0.3)
    upper_ecc = df['eccentricity'].quantile(upper_quantile)
    df_filtered = df[(df['eccentricity'] < upper_ecc) & (df['mean_intensity'] < upper_int)]
    print(f'Before filtration: {len(df)}, After filtration: {len(df_filtered)}')
    print(f'Eccentricity 0.95 quantile: {upper_ecc}, Mean intensity 0.95 quantile: {upper_int}')
    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(10,10))

        sns.histplot(data=df, x='eccentricity', ax=axs[0,0])
        axs[0,0].set_title('Eccentricity before filtration')

        sns.histplot(data=df, x='mean_intensity', ax=axs[1,0])
        axs[1,0].set_title('Mean Intensity before filtration')

        sns.histplot(data=df_filtered, x='eccentricity', ax=axs[0,1])
        axs[0,1].set_title('Eccentricity after filtration')

        sns.histplot(data=df_filtered, x='mean_intensity', ax=axs[1,1])
        axs[1,1].set_title('Mean Intensity after filtration')
        plt.show()
    return df_filtered

def analyze_pathogen_nuclei(db_loc,  cells='Hela', cell_loc=None, upper_quantile=0.95, col_names=['col','col','col'], treatment_loc = [['c2','c3','c4','c5'], ['c7','c8','c9','c10'], ['c14','c15','c16','c17'], ['c19','c20','c21','c22']], pathogens = ['rh', 'dgra14', 'ku80','gra8'], treatments = ['C+_P+', 'C+_P-', 'C-_P+', 'C-_P-'] , pathogen_loc = [['c2','c7','c14','c19'],['c3','c8','c15','c20'],['c4','c9','c16','c21'],['c5','c10','c17','c22']]):
    df = load_and_extract_data(db_loc)
    df = filter_pathogen_nuclei(df, upper_quantile=upper_quantile)
    df = pd.DataFrame(df.groupby(['plate', 'row', 'col', 'field', 'obj'])['label'].nunique())
    df = df.reset_index()
    df = pd.DataFrame(df.groupby(['plate', 'row', 'col'])['label'].mean())
    df = df.reset_index()
    df = df.dropna()
    
    df = annotate_conditions(df, 
                    cells='Hela', 
                    cell_loc=None, 
                    pathogens=pathogens,
                    pathogen_loc=pathogen_loc,
                    treatments=treatments, 
                    treatment_loc=treatment_loc,
                    types=col_names)

    plt.figure(figsize=(50,10))
    sns.stripplot(x="condition", y="label", data=df, jitter=True, size=5)
    plt.xlabel('Condition')
    plt.ylabel('Count')
    plt.show()
    return df

class ImageApp:
    def __init__(self, root, db_path, image_type=None, channels=None, grid_rows=None, grid_cols=None, image_size=(200, 200), annotation_column='annotate'):
        self.root = root
        self.db_path = db_path
        self.index = 0
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.image_size = image_size
        self.annotation_column = annotation_column
        self.image_type = image_type
        self.channels = channels
        self.images = {}
        self.pending_updates = {}
        self.labels = []
        #self.updating_db = True
        self.terminate = False
        self.update_queue = Queue()
        self.status_label = Label(self.root, text="", font=("Arial", 12))
        self.status_label.grid(row=self.grid_rows + 1, column=0, columnspan=self.grid_cols)

        self.db_update_thread = threading.Thread(target=self.update_database_worker)
        self.db_update_thread.start()
        
        for i in range(grid_rows * grid_cols):
            label = tk.Label(root)
            label.grid(row=i // grid_cols, column=i % grid_cols)
            self.labels.append(label)
        
    @staticmethod
    def normalize_image(img):
        img_array = np.array(img)
        img_array = ((img_array - img_array.min()) * (1/(img_array.max() - img_array.min()) * 255)).astype('uint8')
        return Image.fromarray(img_array)

    def add_colored_border(self, img, border_width, border_color):
        top_border = Image.new('RGB', (img.width, border_width), color=border_color)
        bottom_border = Image.new('RGB', (img.width, border_width), color=border_color)
        left_border = Image.new('RGB', (border_width, img.height), color=border_color)
        right_border = Image.new('RGB', (border_width, img.height), color=border_color)

        bordered_img = Image.new('RGB', (img.width + 2 * border_width, img.height + 2 * border_width), color='white')
        bordered_img.paste(top_border, (border_width, 0))
        bordered_img.paste(bottom_border, (border_width, img.height + border_width))
        bordered_img.paste(left_border, (0, border_width))
        bordered_img.paste(right_border, (img.width + border_width, border_width))
        bordered_img.paste(img, (border_width, border_width))

        return bordered_img
    
    def filter_channels(self, img):
        r, g, b = img.split()
        if self.channels:
            if 'r' not in self.channels:
                r = r.point(lambda _: 0)
            if 'g' not in self.channels:
                g = g.point(lambda _: 0)
            if 'b' not in self.channels:
                b = b.point(lambda _: 0)

            if len(self.channels) == 1:
                channel_img = r if 'r' in self.channels else (g if 'g' in self.channels else b)
                return ImageOps.grayscale(channel_img)

        return Image.merge("RGB", (r, g, b))

    def load_images(self):
        for label in self.labels:
            label.config(image='')

        self.images = {}

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        if self.image_type:
            c.execute(f"SELECT png_path, {self.annotation_column} FROM png_list WHERE png_path LIKE ? LIMIT ?, ?", (f"%{self.image_type}%", self.index, self.grid_rows * self.grid_cols))
        else:
            c.execute(f"SELECT png_path, {self.annotation_column} FROM png_list LIMIT ?, ?", (self.index, self.grid_rows * self.grid_cols))
        
        paths = c.fetchall()
        conn.close()

        with ThreadPoolExecutor() as executor:
            loaded_images = list(executor.map(self.load_single_image, paths))

        for i, (img, annotation) in enumerate(loaded_images):
            if annotation:
                border_color = 'teal' if annotation == 1 else 'red'
                img = self.add_colored_border(img, border_width=5, border_color=border_color)
            
            photo = ImageTk.PhotoImage(img)
            label = self.labels[i]
            self.images[label] = photo
            label.config(image=photo)
            
            path = paths[i][0]
            label.bind('<Button-1>', self.get_on_image_click(path, label, img))
            label.bind('<Button-3>', self.get_on_image_click(path, label, img))

        self.root.update()

    def load_single_image(self, path_annotation_tuple):
        path, annotation = path_annotation_tuple
        img = Image.open(path)
        if img.mode == "I":
            img = self.normalize_image(img)
        img = img.convert('RGB')
        img = self.filter_channels(img)
        img = img.resize(self.image_size)
        return img, annotation
        
    def get_on_image_click(self, path, label, img):
        def on_image_click(event):
            
            new_annotation = 1 if event.num == 1 else (2 if event.num == 3 else None)
            
            if path in self.pending_updates and self.pending_updates[path] == new_annotation:
                self.pending_updates[path] = None
                new_annotation = None
            else:
                self.pending_updates[path] = new_annotation
            
            print(f"Image {os.path.split(path)[1]} annotated: {new_annotation}")
            
            img_ = img.crop((5, 5, img.width-5, img.height-5))
            border_fill = 'teal' if new_annotation == 1 else ('red' if new_annotation == 2 else None)
            img_ = ImageOps.expand(img_, border=5, fill=border_fill) if border_fill else img_

            photo = ImageTk.PhotoImage(img_)
            self.images[label] = photo
            label.config(image=photo)
            self.root.update()

        return on_image_click
     
    @staticmethod
    def update_html(text):
        display(HTML(f"""
        <script>
        document.getElementById('unique_id').innerHTML = '{text}';
        </script>
        """))

    def update_database_worker(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        display(HTML("<div id='unique_id'>Initial Text</div>"))

        while True:
            if self.terminate:
                conn.close()
                break

            if not self.update_queue.empty():
                ImageApp.update_html("Do not exit, Updating database...")
                self.status_label.config(text='Do not exit, Updating database...')

                pending_updates = self.update_queue.get()
                for path, new_annotation in pending_updates.items():
                    if new_annotation is None:
                        c.execute(f'UPDATE png_list SET {self.annotation_column} = NULL WHERE png_path = ?', (path,))
                    else:
                        c.execute(f'UPDATE png_list SET {self.annotation_column} = ? WHERE png_path = ?', (new_annotation, path))
                conn.commit()

                # Reset the text
                ImageApp.update_html('')
                self.status_label.config(text='')
                self.root.update()
            time.sleep(0.1)

    def update_gui_text(self, text):
        self.status_label.config(text=text)
        self.root.update()

    def next_page(self):
        if self.pending_updates:  # Check if the dictionary is not empty
            self.update_queue.put(self.pending_updates.copy())
        self.pending_updates.clear()
        self.index += self.grid_rows * self.grid_cols
        self.load_images()

    def previous_page(self):
        if self.pending_updates:  # Check if the dictionary is not empty
            self.update_queue.put(self.pending_updates.copy())
        self.pending_updates.clear()
        self.index -= self.grid_rows * self.grid_cols
        if self.index < 0:
            self.index = 0
        self.load_images()

    def shutdown(self):
        self.terminate = True  # Set terminate first
        self.update_queue.put(self.pending_updates.copy())
        self.pending_updates.clear()
        self.db_update_thread.join()  # Join the thread to make sure database is updated
        self.root.quit()
        self.root.destroy()
        print(f'Quit application')

def annotate(db, image_type=None, channels=None, geom="1000x1100", img_size=(200, 200), rows=5, columns=5, annotation_column='annotate'):
    #display(HTML("<div id='unique_id'>Initial Text</div>"))
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('PRAGMA table_info(png_list)')
    cols = c.fetchall()
    if annotation_column not in [col[1] for col in cols]:
        c.execute(f'ALTER TABLE png_list ADD COLUMN {annotation_column} integer')
    conn.commit()
    conn.close()

    root = tk.Tk()
    root.geometry(geom)
    app = ImageApp(root, db, image_type=image_type, channels=channels, image_size=img_size, grid_rows=rows, grid_cols=columns, annotation_column=annotation_column)
    
    next_button = tk.Button(root, text="Next", command=app.next_page)
    next_button.grid(row=app.grid_rows, column=app.grid_cols - 1)
    back_button = tk.Button(root, text="Back", command=app.previous_page)
    back_button.grid(row=app.grid_rows, column=app.grid_cols - 2)
    exit_button = tk.Button(root, text="Exit", command=app.shutdown)
    exit_button.grid(row=app.grid_rows, column=app.grid_cols - 3)
    
    app.load_images()
    root.mainloop()
    
def check_for_duplicates(db):
    db_path = db
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT file_name, COUNT(file_name) FROM png_list GROUP BY file_name HAVING COUNT(file_name) > 1')
    duplicates = c.fetchall()
    for duplicate in duplicates:
        file_name = duplicate[0]
        count = duplicate[1]
        #print(f"Found {count} duplicates for file_name {file_name}. Deleting {count-1} of them.")
        c.execute('SELECT rowid FROM png_list WHERE file_name = ?', (file_name,))
        rowids = c.fetchall()
        for rowid in rowids[:-1]:
            c.execute('DELETE FROM png_list WHERE rowid = ?', (rowid[0],))
    conn.commit()
    conn.close()
    
def annotate_database(db, geom="3050x1650", img_size=(200, 200), rows=8, columns=15, annotation_column='annotate'):
    check_for_duplicates(db)
    annotate(db, geom, img_size, rows, columns, annotation_column)
    
def remove_highly_correlated_columns(df, threshold):
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    return df.drop(to_drop, axis=1)

def train_apply_xgb_model(df, column='treatment', classes=['infected', 'noninfected'], channel=None, compartment=None, remove_highly_correlated=True, class_equality=False, test_size=0.2, verbose=False):
    df.loc[df[column] == classes[0], 'target'] = int(1)
    df.loc[df[column] == classes[1], 'target'] = int(0)
    if class_equality:
        count_1 = len(df[df['target'] == 1])
        count_0 = len(df[df['target'] == 0])
        min_count = min(count_1, count_0)
        if verbose:
            print(f'class {classes[0]}: {count_0}, class {classes[1]}: {count_1}')
        selected_rows_0 = df[df['target'] == 0].sample(n=min_count, random_state=42)
        selected_rows_1 = df[df['target'] == 1].sample(n=min_count, random_state=42)
        df = pd.concat([selected_rows_0, selected_rows_1])

    numeric_df = df.select_dtypes(include=['int', 'float'])
    y = numeric_df['target']
    
    if channel != None:
        X = numeric_df[[col for col in numeric_df.columns if f'channel_{str(channel)}' in col]]
        if compartment != None:
            X = X[[col for col in X.columns if f'{str(compartment)}' in col]]
    
    if channel == None:
        X = numeric_df.drop('target', axis=1)
        if compartment != None:
            X = X[[col for col in X.columns if f'{str(compartment)}' in col]]

    if remove_highly_correlated:
        X = remove_highly_correlated_columns(X, threshold=0.95)
    
    if verbose:
        print('columns included in Training set')    
        for col in X.columns:
            print(col)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    if verbose:
        print(f'Train set: {len(X_train)}, Test set: {len(X_test)}')
        print(f'== Training XGBoost model ==')
    xgb_model = xgb.XGBClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    
    if verbose:
        print(f'== Testing XGBoost model ===')
    y_pred = xgb_model.predict(X_test)
    if verbose:
        print(f'Classification Report:')
        print(classification_report(y_test, y_pred))
        print("Accuracy Score:", accuracy_score(y_test, y_pred))
    
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': xgb_model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    if verbose:
        display(feature_importance)
    
    return df, feature_importance


def analyze_measurement_data(src, tables = ['cell', 'nucleus', 'cytoplasm'], verbose=True, include_multinucleated=True, include_multiinfected=True, include_noninfected=True, cells=['Hela'], cell_loc=None, pathogens=['wt'], pathogen_loc=None, treatments=None, treatment_loc=None,types=['col','col','col'], column='treatment', classes=['infected', 'noninfected'], channel=None, compartment=None, remove_highly_correlated=True, class_equality=False,test_size=0.2):
    db_loc = [src+'/measurements/measurements.db']
    
    df, object_dfs = read_and_merge_data(db_loc, tables, verbose=True, include_multinucleated=True, include_multiinfected=True, include_noninfected=True)
    
    df = annotate_conditions(df, 
                        cells=cells, 
                        cell_loc=cell_loc, 
                        pathogens=pathogens,
                        pathogen_loc=pathogen_loc,
                        treatments=treatments, 
                        treatment_loc=treatment_loc,
                        types=types)
    
    rf_df, feature_importance = train_apply_xgb_model(df=df,
                                                     column=column,
                                                     classes=classes,
                                                     channel=channel,
                                                     compartment=compartment,
                                                     remove_highly_correlated=remove_highly_correlated,
                                                     class_equality=class_equality,
                                                     test_size=test_size,
                                                     verbose=verbose)
    return rf_df, feature_importance

def generate_train_test(locs, tables=['png_list'], train=10, test=5, equal_samples_per_plate=False, verbose=True):      
    
    def copy_files(df, dst_pc, dst_nc):
        """Helper function to copy files."""
        paths = df.png_path.tolist()
        annotations = df.annotation.tolist()
        for i, path in enumerate(paths):
            annot = annotations[i]
            file = os.path.split(path)[1]
            if annot == 1:
                move = os.path.join(dst_pc, file)
            else:
                move = os.path.join(dst_nc, file)
            shutil.copy(path, move)
    
    def sample_from_groups(df, n):
        """Helper function to sample equally from each group in a grouped dataframe."""
        return df.groupby('plate').apply(lambda x: x.sample(min(n, len(x)), random_state=42)).reset_index(drop=True)
    
    all_dfs = [read_db(loc + '/measurements/measurements.db', tables=tables) for loc in locs]
    
    dfs = pd.concat(all_dfs, axis=0)
    dfs.replace([np.inf, -np.inf], np.nan, inplace=True)
    dfs.dropna(subset=['annotation'], inplace=True)
    dfs['annotation'] = dfs['annotation'].astype(int)
    
    class_0 = dfs[dfs['annotation'] == 0]
    class_1 = dfs[dfs['annotation'] == 1]
    
    if verbose:
        print(f'class 0: {len(class_0)}, class 1: {len(class_1)}')

    min_count = min(len(class_0), len(class_1))

    if equal_samples_per_plate:
        selected_rows_0 = sample_from_groups(class_0, min_count // class_0['plate'].nunique())
        selected_rows_1 = sample_from_groups(class_1, min_count // class_1['plate'].nunique())
    else:
        selected_rows_0 = class_0.sample(n=min_count, random_state=42)
        selected_rows_1 = class_1.sample(n=min_count, random_state=42)

    train_0 = selected_rows_0.sample(n=train, random_state=42)
    train_1 = selected_rows_1.sample(n=train, random_state=42)

    selected_rows_0 = selected_rows_0.drop(train_0.index)
    selected_rows_1 = selected_rows_1.drop(train_1.index)

    test_0 = selected_rows_0.sample(n=test, random_state=42)
    test_1 = selected_rows_1.sample(n=test, random_state=42)

    dst = os.path.join(os.path.dirname(locs[0]), 'resnet')
    dirs = [os.path.join(dst, subfolder, folder) for subfolder in ['train', 'test'] for folder in ['pc', 'nc']]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

    print(f'Generating Train set')
    copy_files(pd.concat([train_0, train_1]), dirs[0], dirs[1])
    print(f'Train pc:{len(os.listdir(dirs[0]))}')
    print(f'Train nc:{len(os.listdir(dirs[1]))}')

    print(f'Generating Test set')
    copy_files(pd.concat([test_0, test_1]), dirs[2], dirs[3])
    print(f'Test pc:{len(os.listdir(dirs[2]))}')
    print(f'test nc:{len(os.listdir(dirs[3]))}')

    return print(f'========== complete ==========')

# Process a single folder and append data to SQLite
def process_folder_and_append_to_db(folder, db_conn, counter):
    with os.scandir(folder) as entries:
        png_paths = [entry.path for entry in entries if entry.is_file()]

    png_df = pd.DataFrame({'png_path': png_paths})
    png_df['file_name'] = png_df['png_path'].apply(lambda x: os.path.basename(x))
    png_df[['plate', 'row', 'col', 'field', 'cell_id', 'prcfo']] = png_df['file_name'].apply(lambda x: pd.Series(map_wells_png(x)))

    try:
        png_df.to_sql('png_list', db_conn, if_exists='append', index=False)
        for i in range(len(png_df)):
            counter += 1
            if counter % 1000 == 0:  # Print every 1000 files
                print(f'Processed {counter} files...', flush=True)
    except sqlite3.OperationalError as e:
        print("SQLite error:", e, flush=True)

# Your function to find png folders
def find_png_folders(directory, png_folders):
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_dir():
                if entry.name.endswith('_png'):
                    print(f'found: {entry.path}')
                    png_folders.append(entry.path)
                else:
                    find_png_folders(entry.path, png_folders)
    return png_folders
        
# A generator function to yield file paths in chunks
def batch_scandir(folder, chunk_size=1000):
    png_paths = []
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_file():
                png_paths.append(entry.path)
                if len(png_paths) >= chunk_size:
                    yield png_paths
                    png_paths = []
    if png_paths:
        yield png_paths

def process_folder_and_append_to_db(folder, db_conn, counter):
    print("Processing folder:", folder, flush=True)  # Debug print
    with os.scandir(folder) as entries:
        png_paths = [entry.path for entry in entries if entry.is_file()]

    png_df = pd.DataFrame({'png_path': png_paths})
    png_df['file_name'] = png_df['png_path'].apply(lambda x: os.path.basename(x))
    #png_df = png_df[~png_df['file_name'].str.contains('[\[\]]')]

    try:
        print("Attempting to write to DB...", flush=True)  # Debug print
        png_df.to_sql('png_list', db_conn, if_exists='append', index=False)
        for i in range(len(png_paths)):
            counter += 1
            if counter % 1000 == 0:  # Print every 1000 files
                print(f'Processed {counter} files...', flush=True)
    except sqlite3.OperationalError as e:
        print("SQLite error:", e, flush=True)
    
    return counter
