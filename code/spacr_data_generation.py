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

dependencies = ["pandas", "ipykernel", "mahotas","scikit-learn", "scikit-image", "seaborn", "matplotlib", "xgboost", "moviepy", "ipywidgets", "ffmpeg"]

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

import os, gc, re, cv2, csv, math, time, torch, json, traceback, glob

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
from IPython.display import Image as ipyimage

import btrack

from btrack import datasets as btrack_datasets

# Data visualization
#%matplotlib inline
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.animation import FuncAnimation

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
import trackpy as tp
import matplotlib.colors as mcolors

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
from scipy.ndimage import gaussian_filter

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
import logging

# Set the logging level to ERROR to suppress informational and warning messages
logging.getLogger('btrack').setLevel(logging.ERROR)

def preprocess_img_data(src, metadata_type='cellvoyager', custom_regex=None, img_format='.tif', bitdepth='uint16', cmap='inferno', figuresize=15, normalize=False, nr=1, plot=False, mask_channels=[0,1,2], batch_size=[100,100,100], timelapse=False, remove_background=False, backgrounds=100, lower_quantile=0.01, save_dtype=np.float32, correct_illumination=False, randomize=True, all_to_mip=False, pick_slice=False, skip_mode='01',settings={}):
    
    def __convert_cq1_well_id(well_id):
        well_id = int(well_id)
        # ASCII code for 'A'
        ascii_A = ord('A')
        # Calculate row and column
        row, col = divmod(well_id - 1, 24)
        # Convert row to letter (A-P) and adjust col to start from 1
        row_letter = chr(ascii_A + row)
        # Format column as two digits
        well_format = f"{row_letter}{col + 1:02d}" 
        return well_format
        
    def ___safe_int_convert(value, default=0):
        try:
            return int(value)
        except ValueError:
            print(f'Could not convert {value} to int using {default}')
            return default

    def _z_to_mip(src, regex, batch_size=100, pick_slice=False, skip_mode='01', metadata_type=''):
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

                            if well[0].isdigit():
                                well = str(___safe_int_convert(well))
                            if field[0].isdigit():
                                field = str(___safe_int_convert(field))
                            if channel[0].isdigit():
                                channel = str(___safe_int_convert(channel))

                            if metadata_type =='cq1':
                                orig_wellID = wellID
                                wellID = __convert_cq1_well_id(wellID)
                                print(f'Converted Well ID: {orig_wellID} to {wellID}', end='\r', flush=True)

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

    def _move_to_chan_folder(src, regex, timelapse=False, metadata_type=''):
        src_path = src
        src = Path(src)
        valid_exts = ['.tif', '.png']

        if not (src / 'stack').exists():
            for file in src.iterdir():
                if file.is_file():
                    name, ext = file.stem, file.suffix
                    if ext in valid_exts:
                        metadata = re.match(regex, file.name) 
                        try:                  
                            try:
                                plateID = metadata.group('plateID')
                            except:
                                plateID = src.name

                            wellID = metadata.group('wellID')
                            fieldID = metadata.group('fieldID')
                            chanID = metadata.group('chanID')
                            timeID = metadata.group('timeID')

                            if wellID[0].isdigit():
                                wellID = str(___safe_int_convert(wellID))
                            if fieldID[0].isdigit():
                                fieldID = str(___safe_int_convert(fieldID))
                            if chanID[0].isdigit():
                                chanID = str(___safe_int_convert(chanID))
                            if timeID[0].isdigit():
                                timeID = str(___safe_int_convert(timeID))

                            if metadata_type =='cq1':
                                orig_wellID = wellID
                                wellID = __convert_cq1_well_id(wellID)
                                print(f'Converted Well ID: {orig_wellID} to {wellID}')

                            newname = f"{plateID}_{wellID}_{fieldID}_{timeID if timelapse else ''}{ext}"
                            newpath = src / chanID
                            move = newpath / newname
                            if move.exists():
                                print(f'WARNING: A file with the same name already exists at location {move}')
                            else:
                                newpath.mkdir(exist_ok=True)
                                shutil.copy(file, move)
                        except:
                            print(f"Could not extract information from filename {name}{ext} with {regex}")

            # Move original images to a new directory
            valid_exts = ['.tif', '.png']
            newpath = os.path.join(src_path, 'orig')
            os.makedirs(newpath, exist_ok=True)
            for filename in os.listdir(src_path):
                if os.path.splitext(filename)[1] in valid_exts:
                    move = os.path.join(newpath, filename)
                    if os.path.exists(move):
                        print(f'WARNING: A file with the same name already exists at location {move}')
                    else:
                        shutil.move(os.path.join(src, filename), move)
        return

    def _merge_channels(src, plot=False):
        src = Path(src)
        stack_dir = src / 'stack'
        chan_dirs = [d for d in src.iterdir() if d.is_dir() and d.name in ['01', '02', '03', '04', '00', '1', '2', '3', '4','0']]

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

    def _mip_all(src, include_first_chan=True):
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

    def _concatenate_channel(src, channels, randomize=True, timelapse=False, batch_size=100):
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
                    np.savez(save_loc, data=stack, filenames=filenames_region)
                    print(save_loc)
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

    def _get_lists_for_normalization(settings):

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

    def _normalize_stack(src, backgrounds=[100,100,100], remove_background=False, lower_quantile=0.01, save_dtype=np.float32, signal_to_noise=[5,5,5], signal_thresholds=[1000,1000,1000], correct_illumination=False):

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

    def _normalize_timelapse(src, lower_quantile=0.01, save_dtype=np.float32):
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

                for array_index in range(single_channel.shape[0]):
                    arr_2d = single_channel[array_index]
                    # Calculate the 1% and 98% percentiles for this specific image
                    q_low = np.percentile(arr_2d[arr_2d != 0], 2)
                    q_high = np.percentile(arr_2d[arr_2d != 0], 98)

                    # Rescale intensity based on the calculated percentiles to fill the dtype range
                    arr_2d_rescaled = exposure.rescale_intensity(arr_2d, in_range=(q_low, q_high), out_range='dtype')
                    normalized_stack[array_index, :, :, chan_index] = arr_2d_rescaled

                    print(f'Progress: files {file_index+1}/{len(paths)}, channels:{chan_index+1}/{stack.shape[-1]}, arrays:{array_index+1}/{single_channel.shape[0]}', end='\r')

            save_loc = os.path.join(output_fldr, f'{name}_norm_timelapse.npz')
            np.savez(save_loc, data=normalized_stack, filenames=filenames)

            del normalized_stack, stack, filenames
            gc.collect()

        print(f'\nSaved normalized stacks: {output_fldr}')

    def _plot_4D_arrays(src, figuresize=10, cmap='inferno', nr_npz=1, nr=1):
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
        print(f'========== creating single channel folders ==========')
        if timelapse:
            _move_to_chan_folder(src, regex, timelapse, metadata_type)
        else:
            _z_to_mip(src, regex, batch_size, pick_slice, skip_mode, metadata_type)

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
        _merge_channels(src, plot=False)
        if timelapse:
            _create_movies_from_npy_per_channel(src+'/stack', fps=2)

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
    _concatenate_channel(src+'/stack', 
                        channels=mask_channels, 
                        randomize=randomize, 
                        timelapse=timelapse, 
                        batch_size=batch_size)
        
    if plot:
        print(f'plotting {nr} images from {src}/channel_stack')
        _plot_4D_arrays(src+'/channel_stack', figuresize, cmap, nr_npz=1, nr=nr)
    nr_of_chan_stacks = len(src+'/channel_stack')
    
    print(f'========== normalizing concatinated npz ==========: {batch_size} stacks per npz in {nr_of_chan_stacks}')
    
    backgrounds, signal_to_noise, signal_thresholds = _get_lists_for_normalization(settings=settings)
    
    if not timelapse:
        _normalize_stack(src+'/channel_stack',
                    backgrounds=backgrounds,
                    lower_quantile=lower_quantile,
                    save_dtype=save_dtype,
                    signal_thresholds=signal_thresholds,
                    correct_illumination=correct_illumination,
                    signal_to_noise=signal_to_noise, 
                    remove_background=remove_background)
    else:
        _normalize_timelapse(src+'/channel_stack', lower_quantile=lower_quantile, save_dtype=np.float32)
        
    if plot:
        _plot_4D_arrays(src+'/norm_channel_stack', nr_npz=1, nr=nr)

    return print(f'========== complete ==========')


def generate_mask_random_cmap(mask):  
    unique_labels = np.unique(mask)
    num_objects = len(unique_labels[unique_labels != 0])
    random_colors = np.random.rand(num_objects+1, 4)
    random_colors[:, 3] = 1
    random_colors[0, :] = [0, 0, 0, 1]
    random_cmap = mpl.colors.ListedColormap(random_colors)
    return random_cmap
    
def random_cmap(num_objects=100):
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

def plot_merged(src, settings):

    def __remove_noninfected(stack, cell_dim, nucleus_dim, pathogen_dim):
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

    def __remove_outside_objects(stack, cell_dim, nucleus_dim, pathogen_dim):
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

    def __remove_multiobject_cells(stack, mask_dim, cell_dim, nucleus_dim, pathogen_dim, object_dim):
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
    
    def __generate_mask_random_cmap(mask):  
        unique_labels = np.unique(mask)
        num_objects = len(unique_labels[unique_labels != 0])
        random_colors = np.random.rand(num_objects+1, 4)
        random_colors[:, 3] = 1
        random_colors[0, :] = [0, 0, 0, 1]
        random_cmap = mpl.colors.ListedColormap(random_colors)
        return random_cmap
    
    def __get_colours_merged(outline_color):
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
    
    def __filter_objects_in_plot(stack, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim, mask_dims, filter_min_max, include_multinucleated, include_multiinfected):

        stack = __remove_outside_objects(stack, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim)
        
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
                    stack = __remove_multiobject_cells(stack, mask_dim, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim, object_dim=pathogen_mask_dim)
                if include_multiinfected is False and cell_mask_dim is not None and pathogen_mask_dim is not None:
                    stack = __remove_multiobject_cells(stack, mask_dim, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim, object_dim=nucleus_mask_dim)
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
        
    def __normalize_and_outline(image, remove_background, backgrounds, normalize, normalization_percentiles, overlay, overlay_chans, mask_dims, outline_colors, outline_thickness):
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
        
    def __plot_merged_plot(overlay, image, stack, mask_dims, figuresize, overlayed_image, outlines, cmap, outline_colors, print_object_number):
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
            random_cmap = __generate_mask_random_cmap(mask)
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
    outline_colors = __get_colours_merged(settings['outline_color'])
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
                stack = __remove_noninfected(stack, settings['cell_mask_dim'], settings['nucleus_mask_dim'], settings['pathogen_mask_dim'])

        if settings['include_multiinfected'] is not True or settings['include_multinucleated'] is not True or settings['filter_min_max'] is not None:
            stack = __filter_objects_in_plot(stack, settings['cell_mask_dim'], settings['nucleus_mask_dim'], settings['pathogen_mask_dim'], mask_dims, settings['filter_min_max'], settings['include_multinucleated'], settings['include_multiinfected'])

        image = np.take(stack, settings['channel_dims'], axis=2)

        overlayed_image, image, outlines = __normalize_and_outline(image, settings['remove_background'], settings['backgrounds'], settings['normalize'], settings['normalization_percentiles'], settings['overlay'], settings['overlay_chans'], mask_dims, outline_colors, settings['outline_thickness'])
        
        if index < settings['nr']:
            index += 1
            fig = __plot_merged_plot(settings['overlay'], image, stack, mask_dims, settings['figuresize'], overlayed_image, outlines, settings['cmap'], outline_colors, settings['print_object_number'])
        else:
            return

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

def generate_time_lists(file_list):
    file_dict = defaultdict(list)
    for filename in file_list:
        if filename.endswith('.npy'):
            parts = filename.split('_')
            if len(parts) >= 4:
                plate, well, field = parts[:3]
                try:
                    timepoint = int(parts[3].split('.')[0])
                except ValueError:
                    continue  # Skip file on conversion error
                key = (plate, well, field)
                file_dict[key].append((timepoint, filename))
            else:
                continue  # Skip file if not correctly formatted

    # Sort each list by timepoint, but keep them grouped
    sorted_grouped_filenames = [sorted(files, key=lambda x: x[0]) for files in file_dict.values()]
    # Extract just the filenames from each group
    sorted_file_lists = [[filename for _, filename in group] for group in sorted_grouped_filenames]

    return sorted_file_lists

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



def create_database(db_path):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()

def __morphological_measurements(cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask, settings, zernike=True, degree=8):

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

    def _calculate_zernike(mask, df, degree=8):
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
        
    morphological_props = ['label', 'area', 'area_filled', 'area_bbox', 'convex_area', 'major_axis_length', 'minor_axis_length', 
                           'eccentricity', 'solidity', 'extent', 'perimeter', 'euler_number', 'equivalent_diameter_area', 'feret_diameter_max']
    
    prop_ls = []
    ls = []
    
    # Create mappings from each cell to its nuclei, pathogens, and cytoplasms
    if settings['cell_mask_dim'] is not None:
        cell_to_nucleus, cell_to_pathogen = get_components(cell_mask, nuclei_mask, pathogen_mask)
        cell_props = pd.DataFrame(regionprops_table(cell_mask, properties=morphological_props))
        cell_props = _calculate_zernike(cell_mask, cell_props, degree=degree)
        prop_ls = prop_ls + [cell_props]
        ls = ls + ['cell']
    else:
        prop_ls = prop_ls + [pd.DataFrame()]
        ls = ls + ['cell']

    if settings['nuclei_mask_dim'] is not None:
        nucleus_props = pd.DataFrame(regionprops_table(nuclei_mask, properties=morphological_props))
        nucleus_props = _calculate_zernike(nuclei_mask, nucleus_props, degree=degree)
        if settings['cell_mask_dim'] is not None:
            nucleus_props = pd.merge(nucleus_props, cell_to_nucleus, left_on='label', right_on='nucleus', how='left')
        prop_ls = prop_ls + [nucleus_props]
        ls = ls + ['nucleus']
    else:
        prop_ls = prop_ls + [pd.DataFrame()]
        ls = ls + ['nucleus']
    
    if settings['pathogen_mask_dim'] is not None:
        pathogen_props = pd.DataFrame(regionprops_table(pathogen_mask, properties=morphological_props))
        pathogen_props = _calculate_zernike(pathogen_mask, pathogen_props, degree=degree)
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

def __intensity_measurements(cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask, channel_arrays, settings, sizes=[3, 6, 12, 24], periphery=True, outside=True):
    
    def _create_dataframe(radial_distributions, object_type):
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
    
    def _extended_regionprops_table(labels, image, intensity_props):
        regions = regionprops(labels, image)
        props = regionprops_table(labels, image, properties=intensity_props)
        percentiles = [5, 10, 25, 50, 75, 85, 95]
        for p in percentiles:
            props[f'percentile_{p}'] = [
                np.percentile(region.intensity_image.flatten()[~np.isnan(region.intensity_image.flatten())], p)
                for region in regions]
        return pd.DataFrame(props)

    def _calculate_homogeneity(label, channel, distances=[2,4,8,16,32,64]):
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
        columns = [f'homogeneity_distance_{d}' for d in distances]
        homogeneity_df = pd.DataFrame(homogeneity_values, columns=columns)

        return homogeneity_df

    def _periphery_intensity(label_mask, image):
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

    def _outside_intensity(label_mask, image, distance=5):
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
    
    def _calculate_radial_distribution(cell_mask, object_mask, channel_arrays, num_bins=6):
        
        def __calculate_average_intensity(distance_map, single_channel_image, num_bins):
            radial_distribution = np.zeros(num_bins)
            for i in range(num_bins):
                min_distance = i * (distance_map.max() / num_bins)
                max_distance = (i + 1) * (distance_map.max() / num_bins)
                bin_mask = (distance_map >= min_distance) & (distance_map < max_distance)
                radial_distribution[i] = single_channel_image[bin_mask].mean()
            return radial_distribution

        
        object_radial_distributions = {}

        # get unique cell labels
        cell_labels = np.unique(cell_mask)
        cell_labels = cell_labels[cell_labels != 0]

        for cell_label in cell_labels:
            cell_region = cell_mask == cell_label

            object_labels = np.unique(object_mask[cell_region])
            object_labels = object_labels[object_labels != 0]

            for object_label in object_labels:
                objecyt_region = object_mask == object_label
                object_boundary = find_boundaries(objecyt_region, mode='outer')
                distance_map = distance_transform_edt(~object_boundary) * cell_region
                for channel_index in range(channel_arrays.shape[2]):
                    radial_distribution = __calculate_average_intensity(distance_map, channel_arrays[:, :, channel_index], num_bins)
                    object_radial_distributions[(cell_label, object_label, channel_index)] = radial_distribution

        return object_radial_distributions
    
    def _calculate_correlation_object_level(channel_image1, channel_image2, mask, settings):
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
    
    def _estimate_blur(image):
        # Compute the Laplacian of the image
        lap = cv2.Laplacian(image, cv2.CV_64F)
        # Compute and return the variance of the Laplacian
        return lap.var()

    radial_dist = settings['radial_dist']
    calculate_correlation = settings['calculate_correlation']
    homogeneity = settings['homogeneity']
    distances = settings['homogeneity_distances']
    
    intensity_props = ["label", "centroid_weighted", "centroid_weighted_local", "max_intensity", "mean_intensity", "min_intensity"]
    col_lables = ['region_label', 'mean', '5_percentile', '10_percentile', '25_percentile', '50_percentile', '75_percentile', '85_percentile', '95_percentile']
    cell_dfs, nucleus_dfs, pathogen_dfs, cytoplasm_dfs = [], [], [], []
    ls = ['cell','nucleus','pathogen','cytoplasm']
    labels = [cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask]
    dfs = [cell_dfs, nucleus_dfs, pathogen_dfs, cytoplasm_dfs]
    
    for i in range(0,channel_arrays.shape[-1]):
        channel = channel_arrays[:, :, i]
        for j, (label, df) in enumerate(zip(labels, dfs)):
            
            if np.max(label) == 0:
                empty_df = pd.DataFrame()
                df.append(empty_df)
                continue
                
            mask_intensity_df = _extended_regionprops_table(label, channel, intensity_props) 
            mask_intensity_df['shannon_entropy'] = shannon_entropy(channel, base=2)

            if homogeneity:
                homogeneity_df = _calculate_homogeneity(label, channel, distances)
                mask_intensity_df = pd.concat([mask_intensity_df.reset_index(drop=True), homogeneity_df], axis=1)

            if periphery:
                if ls[j] == 'nucleus' or ls[j] == 'pathogen':
                    periphery_intensity_stats = _periphery_intensity(label, channel)
                    mask_intensity_df = pd.concat([mask_intensity_df, pd.DataFrame(periphery_intensity_stats, columns=[f'periphery_{stat}' for stat in col_lables])],axis=1)

            if outside:
                if ls[j] == 'nucleus' or ls[j] == 'pathogen':
                    outside_intensity_stats = _outside_intensity(label, channel)
                    mask_intensity_df = pd.concat([mask_intensity_df, pd.DataFrame(outside_intensity_stats, columns=[f'outside_{stat}' for stat in col_lables])], axis=1)

            blur_col = [_estimate_blur(channel[label == region_label]) for region_label in mask_intensity_df['label']]
            mask_intensity_df[f'{ls[j]}_channel_{i}_blur'] = blur_col

            mask_intensity_df.columns = [f'{ls[j]}_channel_{i}_{col}' if col != 'label' else col for col in mask_intensity_df.columns]
            df.append(mask_intensity_df)
    
    if radial_dist:
        if np.max(nuclei_mask) != 0:
            nucleus_radial_distributions = _calculate_radial_distribution(cell_mask, nuclei_mask, channel_arrays, num_bins=6)
            nucleus_df = _create_dataframe(nucleus_radial_distributions, 'nucleus')
            dfs[1].append(nucleus_df)
            
        if np.max(nuclei_mask) != 0:
            pathogen_radial_distributions = _calculate_radial_distribution(cell_mask, pathogen_mask, channel_arrays, num_bins=6)
            pathogen_df = _create_dataframe(pathogen_radial_distributions, 'pathogen')
            dfs[2].append(pathogen_df)
        
    if calculate_correlation:
        if channel_arrays.shape[-1] >= 2:
            for i in range(channel_arrays.shape[-1]):
                for j in range(i+1, channel_arrays.shape[-1]):
                    chan_i = channel_arrays[:, :, i]
                    chan_j = channel_arrays[:, :, j]
                    for m, mask in enumerate(labels):
                        coloc_df = _calculate_correlation_object_level(chan_i, chan_j, mask, settings)
                        coloc_df.columns = [f'{ls[m]}_channel_{i}_channel_{j}_{col}' for col in coloc_df.columns]
                        dfs[m].append(coloc_df)
    
    return pd.concat(cell_dfs, axis=1), pd.concat(nucleus_dfs, axis=1), pd.concat(pathogen_dfs, axis=1), pd.concat(cytoplasm_dfs, axis=1)

def _measure_crop_core(index, time_ls, file, settings):

    def ___get_percentiles(array, q1=2, q2=98):
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

    def ___crop_center(img, cell_mask, new_width, new_height, normalize=(2,98)):
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

    def ___plot_cropped_arrays(stack, figuresize=20,cmap='inferno'):
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

    def ___check_integrity(df):
        df.columns = [col + f'_{i}' if df.columns.tolist().count(col) > 1 and i != 0 else col for i, col in enumerate(df.columns)]
        label_cols = [col for col in df.columns if 'label' in col]
        df['label_list'] = df[label_cols].values.tolist()
        df['object_label'] = df['label_list'].apply(lambda x: x[0])
        df = df.drop(columns=label_cols)
        df['label_list'] = df['label_list'].astype(str)
        return df

    def ___filter_object(mask, min_value):
        count = np.bincount(mask.ravel())
        to_remove = np.where(count < min_value)
        mask[np.isin(mask, to_remove)] = 0
        return mask

    def ___safe_int_convert(value, default=0):
        try:
            return int(value)
        except ValueError:
            print(f'Could not convert {value} to int using {default}')
            return default

    def __map_wells(file_name, timelapse=False):
        try:
            parts = file_name.split('_')
            plate = 'p' + parts[0]
            well = parts[1]
            field = 'f' + str(___safe_int_convert(parts[2]))
            if timelapse:
                timeid = 't' + str(___safe_int_convert(parts[3]))
            if well[0].isalpha():
                row = 'r' + str(string.ascii_uppercase.index(well[0]) + 1)
                column = 'c' + str(int(well[1:]))
            else:
                row, column = well, well
            if timelapse:    
                prcf = '_'.join([plate, row, column, field, timeid])
            else:
                prcf = '_'.join([plate, row, column, field])
        except Exception as e:
            print(f"Error processing filename: {file_name}")
            print(f"Error: {e}")
            plate, row, column, field, timeid, prcf = 'error','error','error','error','error', 'error'
        if timelapse:
            return plate, row, column, field, timeid, prcf
        else:
            return plate, row, column, field, prcf

    def __map_wells_png(file_name, timelapse=False):
        try:
            root, ext = os.path.splitext(file_name)
            parts = root.split('_')
            plate = 'p' + parts[0]
            well = parts[1]
            field = 'f' + str(___safe_int_convert(parts[2]))
            if timelapse:
                timeid = 't' + str(___safe_int_convert(parts[3]))
            object_id = 'o' + str(___safe_int_convert(parts[-1], default='none'))
            if well[0].isalpha():
                row = 'r' + str(string.ascii_uppercase.index(well[0]) + 1)
                column = 'c' + str(___safe_int_convert(well[1:]))
            else:
                row, column = well, well
            if timelapse:
                prcfo = '_'.join([plate, row, column, field, timeid, object_id])
            else:
                prcfo = '_'.join([plate, row, column, field, object_id])
        except Exception as e:
            print(f"Error processing filename: {file_name}")
            print(f"Error: {e}")
            plate, row, column, field, object_id, prcfo = 'error', 'error', 'error', 'error', 'error', 'error'
        if timelapse:
            return plate, row, column, field, timeid, prcfo, object_id,
        else:
            return plate, row, column, field, prcfo, object_id

    def __merge_and_save_to_database(morph_df, intensity_df, table_type, source_folder, file_name, experiment, timelapse=False):
        morph_df = ___check_integrity(morph_df)
        intensity_df = ___check_integrity(intensity_df)
        if len(morph_df) > 0 and len(intensity_df) > 0:
            merged_df = pd.merge(morph_df, intensity_df, on='object_label', how='outer')
            merged_df = merged_df.rename(columns={"label_list_x": "label_list_morphology", "label_list_y": "label_list_intensity"})
            merged_df['file_name'] = file_name
            merged_df['path_name'] = os.path.join(source_folder, file_name + '.npy')
            if timelapse:
                merged_df[['plate', 'row', 'col', 'field', 'timeid', 'prcf']] = merged_df['file_name'].apply(lambda x: pd.Series(__map_wells(x, timelapse)))
            else:
                merged_df[['plate', 'row', 'col', 'field', 'prcf']] = merged_df['file_name'].apply(lambda x: pd.Series(__map_wells(x, timelapse)))
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

    def __exclude_objects(cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask, include_uninfected=True):
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

    def __merge_overlapping_objects(mask1, mask2):
        labeled_1 = label(mask1)
        num_1 = np.max(labeled_1)
        for m1_id in range(1, num_1 + 1):
            current_1_mask = labeled_1 == m1_id
            overlapping_2_labels = np.unique(mask2[current_1_mask])
            overlapping_2_labels = overlapping_2_labels[overlapping_2_labels != 0]
            if len(overlapping_2_labels) > 1:
                overlap_percentages = [np.sum(current_1_mask & (mask2 == m2_label)) / np.sum(current_1_mask) * 100 for m2_label in overlapping_2_labels]
                max_overlap_label = overlapping_2_labels[np.argmax(overlap_percentages)]
                max_overlap_percentage = max(overlap_percentages)
                if max_overlap_percentage >= 90:
                    for m2_label in overlapping_2_labels:
                        if m2_label != max_overlap_label:
                            mask1[(current_1_mask) & (mask2 == m2_label)] = 0
                else:
                    for m2_label in overlapping_2_labels[1:]:
                        mask2[mask2 == m2_label] = overlapping_2_labels[0]
        return mask1, mask2

    def __generate_names(file_name, cell_id, cell_nuclei_ids, cell_pathogen_ids, source_folder, crop_mode='cell'):
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
        parts = file_name.split('_')
        plate = parts[0]
        well = parts[1] 
        metadata = f'{plate}_{well}'
        fldr = os.path.join(fldr,metadata)
        table_name = fldr.replace("/", "_")
        return img_name, fldr, table_name

    def __find_bounding_box(crop_mask, _id, buffer=10):
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

    def _relabel_parent_with_child_labels(parent_mask, child_mask):
        # Label parent mask to identify unique objects
        parent_labels = label(parent_mask, background=0)
        # Use the original child mask labels directly, without relabeling
        child_labels = child_mask

        # Create a new parent mask for updated labels
        parent_mask_new = np.zeros_like(parent_mask)

        # Directly relabel parent cells based on overlapping child labels
        unique_child_labels = np.unique(child_labels)[1:]  # Skip background
        for child_label in unique_child_labels:
            child_area_mask = (child_labels == child_label)
            overlapping_parent_label = np.unique(parent_labels[child_area_mask])

            # Since each parent is assumed to overlap with exactly one nucleus,
            # directly set the parent label to the child label where overlap occurs
            for parent_label in overlapping_parent_label:
                if parent_label != 0:  # Skip background
                    parent_mask_new[parent_labels == parent_label] = child_label

        # For cells containing multiple nuclei, standardize all nuclei to the first label
        # This will be done only if needed, as per your condition
        for parent_label in np.unique(parent_mask_new)[1:]:  # Skip background
            parent_area_mask = (parent_mask_new == parent_label)
            child_labels_in_parent = np.unique(child_mask[parent_area_mask])
            child_labels_in_parent = child_labels_in_parent[child_labels_in_parent != 0]  # Exclude background

            if len(child_labels_in_parent) > 1:
                # Standardize to the first child label within this parent
                first_child_label = child_labels_in_parent[0]
                for child_label in child_labels_in_parent:
                    child_mask[child_mask == child_label] = first_child_label

        return parent_mask_new, child_mask

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
            plot_cropped_arrays(data)
        
        channel_arrays = data[:, :, settings['channels']].astype(data_type)        
        if settings['cell_mask_dim'] is not None:
            cell_mask = data[:, :, settings['cell_mask_dim']].astype(data_type)
            
            if settings['cell_min_size'] is not None and settings['cell_min_size'] != 0:
                cell_mask = ___filter_object(cell_mask, settings['cell_min_size'])
        else:
            cell_mask = np.zeros_like(data[:, :, 0])
            settings['cytoplasm'] = False
            settings['include_uninfected'] = True

        if settings['nuclei_mask_dim'] is not None:
            nuclei_mask = data[:, :, settings['nuclei_mask_dim']].astype(data_type)
            if settings['cell_mask_dim'] is not None:
                nuclei_mask, cell_mask = __merge_overlapping_objects(mask1=nuclei_mask, mask2=cell_mask)
            if settings['nucleus_min_size'] is not None and settings['nucleus_min_size'] != 0:
                nuclei_mask = ___filter_object(nuclei_mask, settings['nucleus_min_size'])
            if settings['timelapse_objects'] == 'nuclei':
                if settings['cell_mask_dim'] is not None:
                    cell_mask, nucleus_mask = _relabel_parent_with_child_labels(cell_mask, nuclei_mask)
                    data[:, :, settings['cell_mask_dim']] = cell_mask
                    data[:, :, settings['nuclei_mask_dim']] = nucleus_mask
                    save_folder = settings['input_folder']
                    np.save(os.path.join(save_folder, file), data)
                
        else:
            nuclei_mask = np.zeros_like(data[:, :, 0])

        if settings['pathogen_mask_dim'] is not None:
            pathogen_mask = data[:, :, settings['pathogen_mask_dim']].astype(data_type)
            if settings['merge_edge_pathogen_cells']:
                if settings['cell_mask_dim'] is not None:
                    pathogen_mask, cell_mask = __merge_overlapping_objects(mask1=pathogen_mask, mask2=cell_mask)
            if settings['pathogen_min_size'] is not None and settings['pathogen_min_size'] != 0:
                pathogen_mask = ___filter_object(pathogen_mask, settings['pathogen_min_size'])
        else:
            pathogen_mask = np.zeros_like(data[:, :, 0])

        # Create cytoplasm mask
        if settings['cytoplasm']:
            if settings['cell_mask_dim'] is not None:
                if settings['nuclei_mask_dim'] is not None and settings['pathogen_mask_dim'] is not None:
                    cytoplasm_mask = np.where(np.logical_or(nuclei_mask != 0, pathogen_mask != 0), 0, cell_mask)
                elif settings['nuclei_mask_dim'] is not None:
                    cytoplasm_mask = np.where(nuclei_mask != 0, 0, cell_mask)
                elif settings['pathogen_mask_dim'] is not None:
                    cytoplasm_mask = np.where(pathogen_mask != 0, 0, cell_mask)
                else:
                    cytoplasm_mask = np.zeros_like(cell_mask)
        else:
            cytoplasm_mask = np.zeros_like(cell_mask)

        if settings['cell_min_size'] is not None and settings['cell_min_size'] != 0:
            cell_mask = ___filter_object(cell_mask, settings['cell_min_size'])
        if settings['nucleus_min_size'] is not None and settings['nucleus_min_size'] != 0:
            nuclei_mask = ___filter_object(nuclei_mask, settings['nucleus_min_size'])
        if settings['pathogen_min_size'] is not None and settings['pathogen_min_size'] != 0:
            pathogen_mask = ___filter_object(pathogen_mask, settings['pathogen_min_size'])
        if settings['cytoplasm_min_size'] is not None and settings['cytoplasm_min_size'] != 0:
            cytoplasm_mask = ___filter_object(cytoplasm_mask, settings['cytoplasm_min_size'])

        if settings['cell_mask_dim'] is not None and settings['pathogen_mask_dim'] is not None:
            if settings['include_uninfected'] == False:
                cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask = __exclude_objects(cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask, include_uninfected=False)

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
            plot_cropped_arrays(data)

        if settings['save_measurements']:

            cell_df, nucleus_df, pathogen_df, cytoplasm_df = __morphological_measurements(cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask, settings)

            cell_intensity_df, nucleus_intensity_df, pathogen_intensity_df, cytoplasm_intensity_df = __intensity_measurements(cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask, channel_arrays, settings, sizes=[1, 2, 3, 4, 5], periphery=True, outside=True)
            if settings['cell_mask_dim'] is not None:
                cell_merged_df = __merge_and_save_to_database(cell_df, cell_intensity_df, 'cell', source_folder, file_name, settings['experiment'], settings['timelapse'])

            if settings['nuclei_mask_dim'] is not None:
                nucleus_merged_df = __merge_and_save_to_database(nucleus_df, nucleus_intensity_df, 'nucleus', source_folder, file_name, settings['experiment'], settings['timelapse'])

            if settings['pathogen_mask_dim'] is not None:
                pathogen_merged_df = __merge_and_save_to_database(pathogen_df, pathogen_intensity_df, 'pathogen', source_folder, file_name, settings['experiment'], settings['timelapse'])

            if settings['cytoplasm']:
                cytoplasm_merged_df = __merge_and_save_to_database(cytoplasm_df, cytoplasm_intensity_df, 'cytoplasm', source_folder, file_name, settings['experiment'], settings['timelapse'])

        
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
                        
                        region = (crop_mask == _id)  # This creates a boolean mask for the region of interest

                        # Use the boolean mask to filter the cell_mask and then find unique IDs
                        region_cell_ids = np.atleast_1d(np.unique(cell_mask[region]))
                        region_nuclei_ids = np.atleast_1d(np.unique(nuclei_mask[region]))
                        region_pathogen_ids = np.atleast_1d(np.unique(pathogen_mask[region]))

                        if settings['use_bounding_box']:
                            region = __find_bounding_box(crop_mask, _id, buffer=10)

                        img_name, fldr, table_name = __generate_names(file_name=file_name, cell_id = region_cell_ids, cell_nuclei_ids=region_nuclei_ids, cell_pathogen_ids=region_pathogen_ids, source_folder=source_folder, crop_mode=crop_mode)

                        if dialate_png:
                            region_area = np.sum(region)
                            approximate_diameter = np.sqrt(region_area)
                            dialate_png_px = int(approximate_diameter * dialate_png_ratio) 
                            struct = generate_binary_structure(2, 2)
                            region = binary_dilation(region, structure=struct, iterations=dialate_png_px)

                        if settings['save_png']:
                            fldr_type = f"{crop_mode}_png/"
                            png_folder = os.path.join(fldr,fldr_type)

                            img_path = os.path.join(png_folder, img_name)
                            
                            png_channels = data[:, :, settings['png_dims']].astype(data_type)

                            if settings['normalize_by'] == 'fov':
                                percentiles_list = get_percentiles(png_channels, settings['normalize_percentiles'][0],q2=settings['normalize_percentiles'][1])

                            png_channels = ___crop_center(png_channels, region, new_width=width, new_height=height)

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

                            img_paths.append(img_path)

                            if len(img_paths) == len(objects_in_image):

                                png_df = pd.DataFrame(img_paths, columns=['png_path'])

                                png_df['file_name'] = png_df['png_path'].apply(lambda x: os.path.basename(x))

                                parts = png_df['file_name'].apply(lambda x: pd.Series(__map_wells_png(x, timelapse=settings['timelapse'])))

                                columns = ['plate', 'row', 'col', 'field']

                                if settings['timelapse']:
                                    columns = columns + ['time_id']

                                columns = columns + ['prcfo']

                                if crop_mode == 'cell':
                                    columns = columns + ['cell_id']

                                if crop_mode == 'nucleus':
                                    columns = columns + ['nucleus_id']

                                if crop_mode == 'pathogen':
                                    columns = columns + ['pathogen_id']

                                if crop_mode == 'cytoplasm':
                                    columns = columns + ['cytoplasm_id']

                                png_df[columns] = parts

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
    
def measure_crop(settings):

    def _timelapse_masks_to_gif(folder_path, mask_channels, object_types):

        def __sort_key(file_path):
            match = re.search(r'(\d+)_([A-Z]\d+)_(\d+)_(\d+).npy', os.path.basename(file_path))
            if match:
                plate, well, field, time = match.groups()
                # Assuming plate, well, and field are to be returned as is and time converted to int for sorting
                return (plate, well, field, int(time))
            else:
                # Return a tuple that sorts this file as "earliest" or "lowest"
                return ('', '', '', 0)

        def __save_mask_timelapse_as_gif(masks, path, cmap, norm, filenames):

            def ___update(frame):
                nonlocal filename_text_obj
                if filename_text_obj is not None:
                    filename_text_obj.remove()
                ax.clear()
                ax.axis('off')
                current_mask = masks[frame]
                ax.imshow(current_mask, cmap=cmap, norm=norm)
                ax.set_title(f'Frame: {frame}', fontsize=24, color='white')
                filename_text = filenames[frame]
                filename_text_obj = fig.text(0.5, 0.01, filename_text, ha='center', va='center', fontsize=20, color='white')
                for label_value in np.unique(current_mask):
                    if label_value == 0: continue  # Skip background
                    y, x = np.mean(np.where(current_mask == label_value), axis=1)
                    ax.text(x, y, str(label_value), color='white', fontsize=24, ha='center', va='center')

            fig, ax = plt.subplots(figsize=(50, 50), facecolor='black')
            ax.set_facecolor('black')
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

            filename_text_obj = None
            anim = FuncAnimation(fig, ___update, frames=len(masks), blit=False)
            anim.save(path, writer='pillow', fps=2, dpi=80)  # Adjust DPI for size/quality
            plt.close(fig)
            print(f'Saved timelapse to {path}')

        def __masks_to_gif(masks, gif_folder, name, filenames, object_type):

            def __display_gif(path):
                with open(path, 'rb') as file:
                    display(ipyimage(file.read()))

            highest_label = max(np.max(mask) for mask in masks)
            random_colors = np.random.rand(highest_label + 1, 4)
            random_colors[:, 3] = 1  # Full opacity
            random_colors[0] = [0, 0, 0, 1]  # Background color
            cmap = plt.cm.colors.ListedColormap(random_colors)
            norm = plt.cm.colors.Normalize(vmin=0, vmax=highest_label)

            save_path_gif = os.path.join(gif_folder, f'timelapse_masks_{object_type}_{name}.gif')
            __save_mask_timelapse_as_gif(masks, save_path_gif, cmap, norm, filenames)
            #__display_gif(save_path_gif)

        master_folder = os.path.dirname(folder_path)
        gif_folder = os.path.join(master_folder, 'movies', 'gif')
        os.makedirs(gif_folder, exist_ok=True)

        paths = glob.glob(os.path.join(folder_path, '*.npy'))
        paths.sort(key=__sort_key)

        organized_files = {}
        for file in paths:
            match = re.search(r'(\d+)_([A-Z]\d+)_(\d+)_\d+.npy', os.path.basename(file))
            if match:
                plate, well, field = match.groups()
                key = (plate, well, field)
                if key not in organized_files:
                    organized_files[key] = []
                organized_files[key].append(file)

        for key, file_list in organized_files.items():
            # Generate the name for the GIF based on plate, well, field
            name = f'{key[0]}_{key[1]}_{key[2]}'
            save_path_gif = os.path.join(gif_folder, f'timelapse_masks_{name}.gif')

            for i, mask_channel in enumerate(mask_channels):
                object_type = object_types[i]
                # Initialize an empty list to store masks for the current object type
                mask_arrays = []

                for file in file_list:
                    # Load only the current time series array
                    array = np.load(file)
                    # Append the specific channel mask to the mask_arrays list
                    mask_arrays.append(array[:, :, mask_channel])

                # Convert mask_arrays list to a numpy array for processing
                mask_arrays_np = np.array(mask_arrays)
                # Generate filenames for each frame in the time series
                filenames = [os.path.basename(f) for f in file_list]
                # Create the GIF for the current time series and object type
                __masks_to_gif(mask_arrays_np, gif_folder, name, filenames, object_type)

    def _list_endpoint_subdirectories(base_dir):
        endpoint_subdirectories = []
        for root, dirs, _ in os.walk(base_dir):
            if not dirs:
                endpoint_subdirectories.append(root)
        return endpoint_subdirectories

    def _scmovie(folder_paths):
        folder_paths = list(set(folder_paths))
        for folder_path in folder_paths:
            movie_path = os.path.join(folder_path, 'movies')
            os.makedirs(movie_path, exist_ok=True)
            # Regular expression to parse the filename
            filename_regex = re.compile(r'(\w+)_(\w+)_(\w+)_(\d+)_(\d+).png')
            # Dictionary to hold lists of images by plate, well, field, and object number
            grouped_images = defaultdict(list)
            # Iterate over all PNG files in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith('.png'):
                    match = filename_regex.match(filename)
                    if match:
                        plate, well, field, time, object_number = match.groups()
                        key = (plate, well, field, object_number)
                        grouped_images[key].append((int(time), os.path.join(folder_path, filename)))
            for key, images in grouped_images.items():
                # Sort images by time using sorted and lambda function for custom sort key
                images = sorted(images, key=lambda x: x[0])
                _, image_paths = zip(*images)
                # Determine the size to which all images should be padded
                max_height = max_width = 0
                for image_path in image_paths:
                    image = cv2.imread(image_path)
                    h, w, _ = image.shape
                    max_height, max_width = max(max_height, h), max(max_width, w)
                # Initialize VideoWriter
                plate, well, field, object_number = key
                output_filename = f"{plate}_{well}_{field}_{object_number}.mp4"
                output_path = os.path.join(movie_path, output_filename)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(output_path, fourcc, 10, (max_width, max_height))
                # Process each image
                for image_path in image_paths:
                    image = cv2.imread(image_path)
                    h, w, _ = image.shape
                    padded_image = np.zeros((max_height, max_width, 3), dtype=np.uint8)
                    padded_image[:h, :w, :] = image
                    video.write(padded_image)
                video.release()

    def _save_settings_to_db(settings):
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

    if settings['timelapse_objects'] == 'nuclei':
        if not settings['cell_mask_dim'] is None:
            tlo = settings['timelapse_objects']
            print(f'timelapse object:{tlo}, cells will be relabeled to nucleus labels to track cells.')

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
    
    if settings['pathogen_mask_dim'] is None:
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
    
    _save_settings_to_db(settings)

    files = [f for f in os.listdir(settings['input_folder']) if f.endswith('.npy')]
    max_workers = settings['max_workers'] or mp.cpu_count()-4
    print(f'using {max_workers} cpu cores')

    with mp.Manager() as manager:
        time_ls = manager.list()
        with mp.Pool(max_workers) as pool:
            result = pool.starmap_async(_measure_crop_core, [(index, time_ls, file, settings) for index, file in enumerate(files)])

            # Track progress in the main process
            while not result.ready():  # Run the loop until all tasks have finished
                time.sleep(1)  # Wait for a short amount of time to avoid excessive printing
                files_processed = len(time_ls)
                files_to_process = len(files)
                average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
                time_left = (((files_to_process-files_processed)*average_time)/max_workers)/60
                print(f'Progress: {files_processed}/{files_to_process} Time/img {average_time:.3f}sec, Time Remaining {time_left:.3f} min.', end='\r', flush=True)
            result.get()
            
    if settings['timelapse']:
        if settings['timelapse_objects'] == 'nuclei':
            folder_path = settings['input_folder']
            mask_channels = [settings['nuclei_mask_dim'], settings['pathogen_mask_dim'],settings['cell_mask_dim']]
            object_types = ['nuclei','pathogen','cell']
            _timelapse_masks_to_gif(folder_path, mask_channels, object_types)

        if settings['save_png']:
            img_fldr = os.path.join(os.path.dirname(settings['input_folder']), 'data')  
            sc_img_fldrs = _list_endpoint_subdirectories(img_fldr)
            _scmovie(sc_img_fldrs)
            
def _npz_to_movie(arrays, filenames, save_path, fps=10):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if save_path.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize VideoWriter with the size of the first image
    height, width = arrays[0].shape[:2]
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for i, frame in enumerate(arrays):
        # Handle float32 images by scaling or normalizing
        if frame.dtype == np.float32:
            frame = np.clip(frame, 0, 1)
            frame = (frame * 255).astype(np.uint8)

        # Convert 16-bit image to 8-bit
        elif frame.dtype == np.uint16:
            frame = cv2.convertScaleAbs(frame, alpha=(255.0/65535.0))

        # Handling 1-channel (grayscale) or 2-channel images
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] in [1, 2]):
            if frame.ndim == 2 or frame.shape[2] == 1:
                # Convert grayscale to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 2:
                # Create an RGB image with the first channel as red, second as green, blue set to zero
                rgb_frame = np.zeros((height, width, 3), dtype=np.uint8)
                rgb_frame[..., 0] = frame[..., 0]  # Red channel
                rgb_frame[..., 1] = frame[..., 1]  # Green channel
                frame = rgb_frame

        # For 3-channel images, ensure it's in BGR format for OpenCV
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Add filenames as text on frames
        cv2.putText(frame, filenames[i], (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"Movie saved to {save_path}")
    
def _create_movies_from_npy_per_channel(src, fps=10):
    master_path = os.path.dirname(src)
    save_path = os.path.join(master_path,'movies')
    os.makedirs(save_path, exist_ok=True)
    # Organize files by plate, well, field
    files = [f for f in os.listdir(src) if f.endswith('.npy')]
    organized_files = {}
    for f in files:
        match = re.match(r'(\w+)_(\w+)_(\w+)_(\d+)\.npy', f)
        if match:
            plate, well, field, time = match.groups()
            key = (plate, well, field)
            if key not in organized_files:
                organized_files[key] = []
            organized_files[key].append((int(time), os.path.join(src, f)))
    for key, file_list in organized_files.items():
        plate, well, field = key
        file_list.sort(key=lambda x: x[0])
        arrays = []
        filenames = []
        for f in file_list:
            array = np.load(f[1])
            #if array.dtype != np.uint8:
            #    array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
            arrays.append(array)
            filenames.append(os.path.basename(f[1]))
        arrays = np.stack(arrays, axis=0)
    for channel in range(arrays.shape[-1]):
        # Extract the current channel for all time points
        channel_arrays = arrays[..., channel]
        # Flatten the channel data to compute global percentiles
        channel_data_flat = channel_arrays.reshape(-1)
        p1, p99 = np.percentile(channel_data_flat, [1, 99])
        # Normalize and rescale each array in the channel
        normalized_channel_arrays = [(np.clip((arr - p1) / (p99 - p1), 0, 1) * 255).astype(np.uint8) for arr in channel_arrays]
        # Convert the list of 2D arrays into a list of 3D arrays with a single channel
        normalized_channel_arrays_3d = [arr[..., np.newaxis] for arr in normalized_channel_arrays]
        # Save as movie for the current channel
        channel_save_path = os.path.join(save_path, f'{plate}_{well}_{field}_channel_{channel}.mp4')
        _npz_to_movie(normalized_channel_arrays_3d, filenames, channel_save_path, fps)
    
def identify_masks(src, object_type, model_name, batch_size, channels, diameter, minimum_size, maximum_size, flow_threshold=30, cellprob_threshold=1, figuresize=25, cmap='inferno', refine_masks=True, filter_size=True, filter_dimm=True, remove_border_objects=False, verbose=False, plot=False, merge=False, save=True, start_at=0, file_type='.npz', net_avg=True, resample=True, timelapse=False, timelapse_displacement=None, timelapse_frame_limits=None, timelapse_memory=3, timelapse_remove_transient=False, timelapse_mode='btrack', timelapse_objects='cell'):

    def __filter_cp_masks(masks, flows, refine_masks, filter_size, minimum_size, maximum_size, remove_border_objects, merge, filter_dimm, batch, moving_avg_q1, moving_avg_q3, moving_count, plot, figuresize, split_objects=False):
        mask_stack = []
        for idx, (mask, flow, image) in enumerate(zip(masks, flows[0], batch)):
            if plot:
                num_objects = mask_object_count(mask)
                print(f'Number of objects before filtration: {num_objects}')
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

    def __save_object_counts_to_database(arrays, object_type, file_names, db_path, added_string):

        def ___count_objects(mask):
                """Count unique objects in a mask, assuming 0 is the background."""
                unique, counts = np.unique(mask, return_counts=True)
                # Assuming 0 is the background label, remove it from the count
                if unique[0] == 0:
                    return len(unique) - 1
                return len(unique)

        records = []
        for mask, file_name in zip(arrays, file_names):
            object_count = ___count_objects(mask)
            count_type = f"{object_type}{added_string}"

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

    def __masks_to_masks_stack(masks):
        mask_stack = []
        for idx, mask in enumerate(masks):
            mask_stack.append(mask)
        return mask_stack

    def __display_gif(path):
        with open(path, 'rb') as file:
            display(ipyimage(file.read()))

    def __save_mask_timelapse_as_gif(masks, tracks_df, path, cmap, norm, filenames):

        # Set the face color for the figure to black
        fig, ax = plt.subplots(figsize=(50, 50), facecolor='black')
        ax.set_facecolor('black')  # Set the axes background color to black
        ax.axis('off')  # Turn off the axis
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)  # Adjust the subplot edges

        filename_text_obj = None  # Initialize a variable to keep track of the text object

        def ___update(frame):
            nonlocal filename_text_obj  # Reference the nonlocal variable to update it
            if filename_text_obj is not None:
                filename_text_obj.remove()  # Remove the previous text object if it exists

            ax.clear()  # Clear the axis to draw the new frame
            ax.axis('off')  # Ensure axis is still off after clearing
            current_mask = masks[frame]
            ax.imshow(current_mask, cmap=cmap, norm=norm)
            ax.set_title(f'Frame: {frame}', fontsize=24, color='white')

            # Add the filename as text on the figure
            filename_text = filenames[frame]  # Get the filename corresponding to the current frame
            filename_text_obj = fig.text(0.5, 0.01, filename_text, ha='center', va='center', fontsize=20, color='white')  # Adjust text position, size, and color as needed

            # Annotate each object with its label number from the mask
            for label_value in np.unique(current_mask):
                if label_value == 0: continue  # Skip background
                y, x = np.mean(np.where(current_mask == label_value), axis=1)
                ax.text(x, y, str(label_value), color='white', fontsize=24, ha='center', va='center')

            # Overlay tracks
            for track in tracks_df['track_id'].unique():
                _track = tracks_df[tracks_df['track_id'] == track]
                ax.plot(_track['x'], _track['y'], '-w', linewidth=1)

        anim = FuncAnimation(fig, ___update, frames=len(masks), blit=False)
        anim.save(path, writer='pillow', fps=2, dpi=80)  # Adjust DPI for size/quality
        plt.close(fig)
        print(f'Saved timelapse to {path}')

    def __visualize_and_save_timelapse_stack_with_tracks(masks, tracks_df, save, src, name, plot, filenames, object_type, mode='btrack',interactive=False):
        highest_label = max(np.max(mask) for mask in masks)
        # Generate random colors for each label, including the background
        random_colors = np.random.rand(highest_label + 1, 4)
        random_colors[:, 3] = 1  # Full opacity
        random_colors[0] = [0, 0, 0, 1]  # Background color
        cmap = plt.cm.colors.ListedColormap(random_colors)
        # Ensure the normalization range covers all labels
        norm = plt.cm.colors.Normalize(vmin=0, vmax=highest_label)
        # Function to plot a frame and overlay tracks
        def ___view_frame_with_tracks(frame=0):
            fig, ax = plt.subplots(figsize=(50, 50))
            current_mask = masks[frame]
            ax.imshow(current_mask, cmap=cmap, norm=norm)  # Apply both colormap and normalization
            ax.set_title(f'Frame: {frame}')

            # Directly annotate each object with its label number from the mask
            for label_value in np.unique(current_mask):
                if label_value == 0: continue  # Skip background
                y, x = np.mean(np.where(current_mask == label_value), axis=1)
                ax.text(x, y, str(label_value), color='white', fontsize=24, ha='center', va='center')

            # Overlay tracks
            for track in tracks_df['track_id'].unique():
                _track = tracks_df[tracks_df['track_id'] == track]
                ax.plot(_track['x'], _track['y'], '-k', linewidth=1)

            ax.axis('off')
            plt.show()

        if plot:
            if interactive:
                interact(___view_frame_with_tracks, frame=IntSlider(min=0, max=len(masks)-1, step=1, value=0))

        if save:
            #save as gif
            gif_path = os.path.join(os.path.dirname(src), 'movies', 'gif')
            os.makedirs(gif_path, exist_ok=True)
            save_path_gif = os.path.join(gif_path, f'timelapse_masks_{object_type}_{name}.gif')
            __save_mask_timelapse_as_gif(masks, tracks_df, save_path_gif, cmap, norm, filenames)
            if plot:
                if not interactive:
                    __display_gif(save_path_gif)

    def __relabel_masks_based_on_tracks(masks, tracks, mode='btrack'):
        # Initialize an array to hold the relabeled masks with the same shape and dtype as the input masks
        relabeled_masks = np.zeros(masks.shape, dtype=masks.dtype)

        # Iterate through each frame
        for frame_number in range(masks.shape[0]):
            # Extract the mapping for the current frame from the tracks DataFrame
            frame_tracks = tracks[tracks['frame'] == frame_number]
            mapping = dict(zip(frame_tracks['original_label'], frame_tracks['track_id']))
            current_mask = masks[frame_number, :, :]

            # Apply the mapping to the current mask
            for original_label, new_label in mapping.items():
                # Where the current mask equals the original label, set it to the new label value
                relabeled_masks[frame_number][current_mask == original_label] = new_label

        return relabeled_masks

    def __prepare_for_tracking(mask_array):
        frames = []
        for t, frame in enumerate(mask_array):
            props = regionprops(frame)
            for obj in props:
                # Include 'label' in the dictionary to capture the original label of the object
                frames.append({
                    'frame': t, 
                    'y': obj.centroid[0], 
                    'x': obj.centroid[1], 
                    'mass': obj.area,
                    'original_label': obj.label  # Capture the original label
                })
        return pd.DataFrame(frames)

    #def __remove_even_labels_first_frame(array, by=2):
    #    # Check if the input array has the correct shape (3 dimensions)
    #    if array.ndim != 3:
    #        raise ValueError("Input array must be 3D.")
    #    # Select the first frame
    #    first_frame = array[0]
    #    # Find even labels in the first frame
    #    even_labels_mask = first_frame % by == 0
    #    # Set those labels to 0
    #    array[0][even_labels_mask] = 0
    #    return array

    def __find_optimal_search_range(features, initial_search_range=500, increment=10, max_attempts=49, memory=3):
        optimal_search_range = initial_search_range
        for attempt in range(max_attempts):
            try:
                # Attempt to link features with the current search range
                tracks_df = tp.link(features, search_range=optimal_search_range, memory=memory)
                print(f"Success with search_range={optimal_search_range}")
                return optimal_search_range
            except Exception as e:
                #print(f"SubnetOversizeException with search_range={optimal_search_range}: {e}")
                optimal_search_range -= increment
                print(f'Retrying with displacement value: {optimal_search_range}', end='\r', flush=True)
        min_range = initial_search_range-(max_attempts*increment)
        if optimal_search_range <= min_range:
            print(f'timelapse_displacement={optimal_search_range} is to high. Lower timelapse_displacement or set to None for automatic thresholding.')
        return optimal_search_range
        
    def __remove_objects_from_first_frame(masks, percentage=10):
        first_frame = masks[0]
        unique_labels = np.unique(first_frame[first_frame != 0])
        num_labels_to_remove = max(1, int(len(unique_labels) * (percentage / 100)))
        labels_to_remove = random.sample(list(unique_labels), num_labels_to_remove)

        for label in labels_to_remove:
            masks[0][first_frame == label] = 0
        return masks

    def __facilitate_trackin_with_adaptive_removal(masks, search_range=500, max_attempts=100, memory=3):
        attempts = 0
        first_frame = masks[0]
        starting_objects = np.unique(first_frame[first_frame != 0])
        while attempts < max_attempts:
            try:
                masks = __remove_objects_from_first_frame(masks, 10)
                first_frame = masks[0]
                objects = np.unique(first_frame[first_frame != 0])
                print(len(objects))
                features = __prepare_for_tracking(masks)
                tracks_df = tp.link(features, search_range=search_range, memory=memory)
                print(f"Success with {len(objects)} objects, started with {len(starting_objects)} objects")
                return masks, features, tracks_df
            except Exception as e:  # Consider catching a more specific exception if possible
                print(f"Retrying with fewer objects. Exception: {e}", flush=True)
            finally:
                attempts += 1
        print(f"Failed to track objects after {max_attempts} attempts. Consider adjusting parameters.")
        return None, None, None

    def __trackpy_track_cells(src, name, batch_filenames, object_type, masks, timelapse_displacement, timelapse_memory, timelapse_remove_transient, plot, save, mode):

        if timelapse_displacement is None:
            features = __prepare_for_tracking(masks)
            timelapse_displacement = __find_optimal_search_range(features, initial_search_range=500, increment=10, max_attempts=49, memory=3)
            if timelapse_displacement is None:
                timelapse_displacement = 50

        masks, features, tracks_df = __facilitate_trackin_with_adaptive_removal(masks, search_range=timelapse_displacement, max_attempts=100, memory=timelapse_memory)
            
        tracks_df['particle'] += 1

        if timelapse_remove_transient:
            tracks_df_filter = tp.filter_stubs(tracks_df, len(masks))
        else:
            tracks_df_filter = tracks_df.copy()

        tracks_df_filter = tracks_df_filter.rename(columns={'particle': 'track_id'})
        print(f'Removed {len(tracks_df)-len(tracks_df_filter)} objects that were not present in all frames')
        masks = __relabel_masks_based_on_tracks(masks, tracks_df_filter)
        tracks_path = os.path.join(os.path.dirname(src), 'tracks')
        os.makedirs(tracks_path, exist_ok=True)
        tracks_df_filter.to_csv(os.path.join(tracks_path, f'trackpy_tracks_{object_type}_{name}.csv'), index=False)
        if plot or save:
            __visualize_and_save_timelapse_stack_with_tracks(masks, tracks_df_filter, save, src, name, plot, batch_filenames, object_type, mode)

        mask_stack = __masks_to_masks_stack(masks)
        return mask_stack

    def __filter_short_tracks(df, min_length=5):
        """Filter out tracks that are shorter than min_length."""
        track_lengths = df.groupby('track_id').size()
        long_tracks = track_lengths[track_lengths >= min_length].index
        return df[df['track_id'].isin(long_tracks)]

    def __btrack_track_cells(src, name, batch_filenames, object_type, plot, save, masks_3D, mode, timelapse_remove_transient, radius=100, workers=10):

        CONFIG_FILE = btrack_datasets.cell_config()
        frame, width, height = masks_3D.shape

        FEATURES = ["area", "major_axis_length", "minor_axis_length", "orientation", "solidity"]
        objects = btrack.utils.segmentation_to_objects(masks_3D, properties=tuple(FEATURES), num_workers=workers)

        # initialise a tracker session using a context manager
        with btrack.BayesianTracker() as tracker:
            tracker.configure(CONFIG_FILE) # configure the tracker using a config file
            tracker.max_search_radius = radius
            tracker.tracking_updates = ["MOTION", "VISUAL"]
            #tracker.tracking_updates = ["MOTION"]
            tracker.features = FEATURES
            tracker.append(objects) # append the objects to be tracked
            tracker.volume=((0, height), (0, width)) # set the tracking volume
            tracker.track(step_size=100) # track them (in interactive mode)
            tracker.optimize() # generate hypotheses and run the global optimizer
            #data, properties, graph = tracker.to_napari() # get the tracks in a format for napari visualization
            tracks = tracker.tracks # store the tracks
            #cfg = tracker.configuration # store the configuration

        # Process the track data to create a DataFrame
        track_data = []
        for track in tracks:
            for t, x, y, z in zip(track.t, track.x, track.y, track.z):
                track_data.append({
                    'track_id': track.ID,
                    'frame': t,
                    'x': x,
                    'y': y,
                    'z': z
                })
        # Convert track data to a DataFrame
        tracks_df = pd.DataFrame(track_data)
        if timelapse_remove_transient:
            tracks_df = __filter_short_tracks(tracks_df, min_length=len(masks_3D))

        objects_df = __prepare_for_tracking(masks_3D)

        # Optional: If necessary, round 'x' and 'y' to ensure matching precision
        tracks_df['x'] = tracks_df['x'].round(decimals=2)
        tracks_df['y'] = tracks_df['y'].round(decimals=2)
        objects_df['x'] = objects_df['x'].round(decimals=2)
        objects_df['y'] = objects_df['y'].round(decimals=2)

        # Merge the DataFrames on 'frame', 'x', and 'y'
        merged_df = pd.merge(tracks_df, objects_df, on=['frame', 'x', 'y'], how='inner')
        final_df = merged_df[['track_id', 'frame', 'x', 'y', 'original_label']]

        masks = __relabel_masks_based_on_tracks(masks_3D, final_df)
        tracks_path = os.path.join(os.path.dirname(src), 'tracks')
        os.makedirs(tracks_path, exist_ok=True)
        final_df.to_csv(os.path.join(tracks_path, f'btrack_tracks_{object_type}_{name}.csv'), index=False)
        if plot or save:
            __visualize_and_save_timelapse_stack_with_tracks(masks, final_df, save, src, name, plot, batch_filenames, object_type, mode)

        mask_stack = __masks_to_masks_stack(masks)
        return mask_stack
    
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

        name = os.path.basename(path)
        name, ext = os.path.splitext(name)
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
                    if isinstance(timelapse_frame_limits, list):
                        if len(timelapse_frame_limits) >= 2:
                            stack = stack[timelapse_frame_limits[0]: timelapse_frame_limits[1], :, :, :].astype(stack.dtype)
                            filenames = filenames[timelapse_frame_limits[0]: timelapse_frame_limits[1]]
                            batch_size = len(stack)
                            print(f'Cut batch an indecies: {timelapse_frame_limits}, New batch_size: {batch_size} ')

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
                    
                if timelapse:
                    stitch_threshold=100.0
                    movie_path = os.path.join(os.path.dirname(src), 'movies')
                    os.makedirs(movie_path, exist_ok=True)
                    save_path = os.path.join(movie_path, f'timelapse_{object_type}_{name}.mp4')
                    _npz_to_movie(batch, batch_filenames, save_path, fps=2)
                else:
                    stitch_threshold=0.0
                   
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
                                stitch_threshold=stitch_threshold,
                                progress=None)
                                
                if timelapse:
                    __save_object_counts_to_database(masks, object_type, batch_filenames, count_loc, added_string='_timelapse')
                    if object_type in timelapse_objects:
                        if timelapse_mode == 'btrack':
                            if not timelapse_displacement is None:
                                radius = timelapse_displacement
                            else:
                                radius = 100

                            workers = os.cpu_count()-2
                            if workers < 1:
                                workers = 1

                            mask_stack = __btrack_track_cells(src, name, batch_filenames, object_type, plot, save, masks_3D=masks, mode=timelapse_mode, timelapse_remove_transient=timelapse_remove_transient, radius=radius, workers=workers)
                        if timelapse_mode == 'trackpy':
                            mask_stack = __trackpy_track_cells(src, name, batch_filenames, object_type, masks, timelapse_displacement, timelapse_memory, timelapse_remove_transient, plot, save, timelapse_mode)
                            
                    else:
                        mask_stack = __masks_to_masks_stack(masks)

                else:
                    __save_object_counts_to_database(masks, object_type, batch_filenames, count_loc, added_string='_before_filtration')
                    mask_stack = __filter_cp_masks(masks, flows, refine_masks, filter_size, minimum_size, maximum_size, remove_border_objects, merge, filter_dimm, batch, moving_avg_q1, moving_avg_q3, moving_count, plot, figuresize)
                    __save_object_counts_to_database(mask_stack, object_type, batch_filenames, count_loc, added_string='_after_filtration')

                if not np.any(mask_stack):
                    average_obj_size = 0
                else:
                    average_obj_size = get_avg_object_size(mask_stack)

                average_sizes.append(average_obj_size) 
                overall_average_size = np.mean(average_sizes) if len(average_sizes) > 0 else 0
            
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
        if save:
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

def _generate_masks(src, object_type, mag, batch_size, channels, cellprob_threshold, plot, save, verbose, nr=1, start_at=0, merge=False, file_type='.npz', timelapse=False, timelapse_displacement=None, timelapse_memory=3, timelapse_frame_limits=None, timelapse_remove_transient=False, timelapse_mode='btrack', timelapse_objects = ['cell'], settings={}):

    def _get_diam(mag, obj):
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
    
    if object_type == 'cell':
        refine_masks = False
        filter_size = False
        filter_dimm = False
        remove_border_objects = False
        if settings['nucleus_channel'] is None:
            model_name = 'cyto'
        else:
            model_name = 'cyto2'
        diameter = _get_diam(mag, obj='cell')
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
        diameter = _get_diam(mag, obj='nuclei')
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
        diameter = _get_diam(mag, obj='pathogen')
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
        diameter = _get_diam(mag, obj='pathogen_nuclei')
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
                   timelapse=timelapse,
                   timelapse_displacement=timelapse_displacement,
                   timelapse_frame_limits=timelapse_frame_limits,
                   timelapse_memory=timelapse_memory,
                   timelapse_remove_transient=timelapse_remove_transient,
                   timelapse_mode=timelapse_mode,
                   timelapse_objects=timelapse_objects)
    
    return print('========== complete ==========')

def preprocess_generate_masks(src, settings={},advanced_settings={}):

    def _pivot_counts_table(db_path):

        def __read_table_to_dataframe(db_path, table_name='object_counts'):
            # Connect to the SQLite database
            conn = sqlite3.connect(db_path)
            # Read the entire table into a pandas DataFrame
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            # Close the connection
            conn.close()
            return df

        def __pivot_dataframe(df):
            # Pivot the DataFrame
            pivoted_df = df.pivot(index='file_name', columns='count_type', values='object_count').reset_index()
            # Because the pivot operation can introduce NaN values for missing data,
            # you might want to fill those NaNs with a default value, like 0
            pivoted_df = pivoted_df.fillna(0)
            return pivoted_df

        # Read the original 'object_counts' table
        df = __read_table_to_dataframe(db_path, 'object_counts')
        # Pivot the DataFrame to have one row per filename and a column for each object type
        pivoted_df = __pivot_dataframe(df)
        # Reconnect to the SQLite database to overwrite the 'object_counts' table with the pivoted DataFrame
        conn = sqlite3.connect(db_path)
        # When overwriting, ensure that you drop the existing table or use if_exists='replace' to overwrite it
        pivoted_df.to_sql('object_counts', conn, if_exists='replace', index=False)
        conn.close()

    def _load_and_concatenate_arrays(src, channels, cell_chann_dim, nucleus_chann_dim, pathogen_chann_dim):
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

                if stack.shape[-1] > concatenated_array.shape[-1]:
                    output_path = os.path.join(output_folder, filename)
                    np.save(output_path, stack)
            print(f'Files merged: {count}/{all_imgs}', end='\r', flush=True)
        return
        
    def _get_cellpose_channels(mask_channels, nucleus_chann_dim, pathogen_chann_dim, cell_chann_dim):
        cellpose_channels = {}
        if nucleus_chann_dim in mask_channels:
            cellpose_channels['nucleus'] = [0, mask_channels.index(nucleus_chann_dim)]
        if pathogen_chann_dim in mask_channels:
            cellpose_channels['pathogen'] = [0, mask_channels.index(pathogen_chann_dim)]
        if cell_chann_dim in mask_channels:
            cellpose_channels['cell'] = [0, mask_channels.index(cell_chann_dim)]
        return cellpose_channels

    settings = {**settings, **advanced_settings}
    settings_df = pd.DataFrame(list(settings.items()), columns=['Key', 'Value'])
    settings_csv = os.path.join(src,'settings','preprocess_generate_masks_settings.csv')
    os.makedirs(os.path.join(src,'settings'), exist_ok=True)
    settings_df.to_csv(settings_csv, index=False)
    
    if settings['timelapse']:
        settings['randomize'] = False
    
    mask_channels = [settings['nucleus_channel'], settings['cell_channel'], settings['pathogen_channel']]
    mask_channels = [item for item in mask_channels if item is not None]
    
    if settings['preprocess']:
        if not settings['masks']:
            print(f'WARNING: channels for mask generation are defined when preprocess = True')
    
    if isinstance(settings['merge'], bool):
        settings['merge'] = [settings['merge']]*3
    if isinstance(settings['save'], bool):
        settings['save'] = [settings['save']]*3

    if settings['preprocess']: 
        preprocess_img_data(src,
                            metadata_type=settings['metadata_type'],
                            custom_regex=settings['custom_regex'],
                            plot=settings['plot'],
                            normalize=settings['normalize_plots'],
                            mask_channels=mask_channels,
                            batch_size=settings['batch_size'],
                            timelapse=settings['timelapse'],
                            remove_background=settings['remove_background'],
                            lower_quantile=settings['lower_quantile'],
                            save_dtype=np.float32,
                            correct_illumination=False,
                            randomize=settings['randomize'],
                            nr=settings['examples_to_plot'],
                            all_to_mip=settings['all_to_mip'],
                            pick_slice=settings['pick_slice'],
                            skip_mode=settings['skip_mode'],
                            settings = settings)
    if settings['masks']:

        cellpose_channels = _get_cellpose_channels(mask_channels, settings['nucleus_channel'], settings['pathogen_channel'], settings['cell_channel'])

        if settings['cell_channel'] != None:
            cell_channels = cellpose_channels['cell']
            _generate_masks(src,
                           object_type='cell',
                           mag=settings['magnefication'],
                           batch_size=settings['batch_size'],
                           channels=cell_channels,
                           cellprob_threshold=settings['cell_CP_prob'],
                           plot=settings['plot'],
                           nr=settings['examples_to_plot'],
                           save=settings['save'][0],
                           merge=settings['merge'][0],
                           verbose=settings['verbose'],
                           timelapse=settings['timelapse'],
                           file_type='.npz',
                           timelapse_displacement=settings['timelapse_displacement'], 
                           timelapse_memory=settings['timelapse_memory'],
                           timelapse_frame_limits=settings['timelapse_frame_limits'],
                           timelapse_remove_transient=settings['timelapse_remove_transient'],
                           timelapse_mode=settings['timelapse_mode'],
                           timelapse_objects=settings['timelapse_objects'],
                           settings=settings)
            torch.cuda.empty_cache()
        if settings['nucleus_channel'] != None:
            nucleus_channels = cellpose_channels['nucleus']
            _generate_masks(src,
                           object_type='nuclei',
                           mag=settings['magnefication'],
                           batch_size=settings['batch_size'],
                           channels=nucleus_channels,
                           cellprob_threshold=settings['nucleus_CP_prob'],
                           plot=settings['plot'],
                           nr=settings['examples_to_plot'],
                           save=settings['save'][1],
                           merge=settings['merge'][1],
                           verbose=settings['verbose'],
                           timelapse=settings['timelapse'],
                           file_type='.npz',
                           timelapse_displacement=settings['timelapse_displacement'], 
                           timelapse_memory=settings['timelapse_memory'],
                           timelapse_frame_limits=settings['timelapse_frame_limits'],
                           timelapse_remove_transient=settings['timelapse_remove_transient'],
                           timelapse_mode=settings['timelapse_mode'],
                           timelapse_objects=settings['timelapse_objects'],
                           settings=settings)
            torch.cuda.empty_cache()
        if settings['pathogen_channel'] != None:
            pathogen_channels = cellpose_channels['pathogen']
            _generate_masks(src,
                           object_type='pathogen',
                           mag=settings['magnefication'],
                           batch_size=settings['batch_size'],
                           channels=pathogen_channels,
                           cellprob_threshold=settings['pathogen_CP_prob'],
                           plot=settings['plot'],
                           nr=settings['examples_to_plot'],
                           save=settings['save'][2],
                           merge=settings['merge'][2],
                           verbose=settings['verbose'],
                           timelapse=settings['timelapse'],
                           file_type='.npz',
                           timelapse_displacement=settings['timelapse_displacement'], 
                           timelapse_memory=settings['timelapse_memory'],
                           timelapse_frame_limits=settings['timelapse_frame_limits'],
                           timelapse_remove_transient=settings['timelapse_remove_transient'],
                           timelapse_mode=settings['timelapse_mode'],
                           timelapse_objects=settings['timelapse_objects'],
                           settings=settings)
            torch.cuda.empty_cache()
        if os.path.exists(os.path.join(src,'measurements')):
            _pivot_counts_table(db_path=os.path.join(src,'measurements', 'measurements.db'))

        #Concatinate stack with masks
        _load_and_concatenate_arrays(src, settings['channels'], settings['cell_channel'], settings['nucleus_channel'], settings['pathogen_channel'])
        
        if settings['plot']:
            if not settings['timelapse']:
                plot_dims = len(settings['channels'])
                overlay_channels = [2,1,0]
                cell_mask_dim = nucleus_mask_dim = pathogen_mask_dim = None
                plot_counter = plot_dims

                if settings['cell_channel'] is not None:
                    cell_mask_dim = plot_counter
                    plot_counter += 1

                if settings['nucleus_channel'] is not None:
                    nucleus_mask_dim = plot_counter
                    plot_counter += 1

                if settings['pathogen_channel'] is not None:
                    pathogen_mask_dim = plot_counter

                plot_settings = {'include_noninfected':True, 
                                 'include_multiinfect':True,
                                 'include_multinucleated':True,
                                 'remove_background':False,
                                 'filter_min_max':None,
                                 'channel_dims':settings['channels'],
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
                                 'nr':settings['examples_to_plot'],
                                 'figuresize':20,
                                 'cmap':'inferno',
                                 'verbose':True}
                plot_merged(src=os.path.join(src,'merged'), settings=plot_settings)
            else:
                plot_arrays(src=os.path.join(src,'merged'), figuresize=50, cmap='inferno', nr=1, normalize=True, q1=1, q2=99)
            
    torch.cuda.empty_cache()
    gc.collect()
    return

def annotate_conditions(df, cells=['HeLa'], cell_loc=None, pathogens=['rh'], pathogen_loc=None, treatments=['cm'], treatment_loc=None, types = ['col','col','col']):

    # Function to apply to each row
    def _map_values(row, dict_, type_='col'):
        for values, cols in dict_.items():
            if row[type_] in cols:
                return values
        return None

    if cell_loc is None:
        df['host_cells'] = cells[0]
    else:
        cells_dict = dict(zip(cells, cell_loc))
        df['host_cells'] = df.apply(lambda row: _map_values(row, cells_dict, type_=types[0]), axis=1)
    if pathogen_loc is None:
        if pathogens != None:
            df['pathogen'] = 'none'
    else:
        pathogens_dict = dict(zip(pathogens, pathogen_loc))
        df['pathogen'] = df.apply(lambda row: _map_values(row, pathogens_dict, type_=types[1]), axis=1)
    if treatment_loc is None:
        df['treatment'] = 'cm'
    else:
        treatments_dict = dict(zip(treatments, treatment_loc))
        df['treatment'] = df.apply(lambda row: _map_values(row, treatments_dict, type_=types[2]), axis=1)
    if pathogens != None:
        df['condition'] = df['pathogen']+'_'+df['treatment']
    else:
        df['condition'] = df['treatment']
    return df
    
def read_and_merge_data(locs, tables, verbose=False, include_multinucleated=False, include_multiinfected=False, include_noninfected=False):

    def read_db(db_loc, tables):
        conn = sqlite3.connect(db_loc)
        dfs = []
        for table in tables:
            query = f'SELECT * FROM {table}'
            df = pd.read_sql_query(query, conn)
            dfs.append(df)
        conn.close()
        return dfs

    def _split_data(df, group_by, object_type):

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
        cells_g_df, metadata = _split_data(cells, 'prcfo', 'object_label')
        print(f'cells: {len(cells)}')
        print(f'cells grouped: {len(cells_g_df)}')
    if 'cytoplasm' in tables:
        cytoplasms = cytoplasms.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        cytoplasms = cytoplasms.assign(prcfo = lambda x: x['prcf'] + '_' + x['object_label'])
        cytoplasms_g_df, _ = _split_data(cytoplasms, 'prcfo', 'object_label')
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
        nuclei_g_df, _ = _split_data(nuclei, 'prcfo', 'cell_id')
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
        pathogens_g_df, _ = _split_data(pathogens, 'prcfo', 'cell_id')
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

def analyze_recruitment(src, metadata_settings, advanced_settings):

    def _group_by_well(df):
        numeric_cols = df._get_numeric_data().columns
        non_numeric_cols = df.select_dtypes(include=['object']).columns

        # Apply mean function to numeric columns and first to non-numeric
        df_grouped = df.groupby(['plate', 'row', 'col']).agg({**{col: np.mean for col in numeric_cols}, **{col: 'first' for col in non_numeric_cols}})
        return df_grouped

    def _plot_controls(df, mask_chans, channel_of_interest, figuresize=5):
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

    def _plot_recruitment(df, df_type, channel_of_interest, target, columns=[], figuresize=50):

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

    def _calculate_recruitment(df, channel):

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
    
    def _object_filter(df, object_type, size_range, intensity_range, mask_chans, mask_chan):
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
    
    def _results_to_csv(src, df, df_well):
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
        df = _object_filter(df, object_type='cell', size_range=cell_size_range, intensity_range=cell_intensity_range, mask_chans=mask_chans, mask_chan=0)
        if not target_intensity_min is None:
            df = df[df[f'cell_channel_{channel_of_interest}_percentile_95'] > target_intensity_min]
            print(f'After channel {channel_of_interest} filtration', len(df))
    if not nucleus_chann_dim is None:
        df = _object_filter(df, object_type='nucleus', size_range=nucleus_size_range, intensity_range=nucleus_intensity_range, mask_chans=mask_chans, mask_chan=1)
    if not pathogen_chann_dim is None:
        df = _object_filter(df, object_type='pathogen', size_range=pathogen_size_range, intensity_range=pathogen_intensity_range, mask_chans=mask_chans, mask_chan=2)
       
    df['recruitment'] = df[f'pathogen_channel_{channel_of_interest}_mean_intensity']/df[f'cytoplasm_channel_{channel_of_interest}_mean_intensity']
    for chan in channel_dims:
        df = _calculate_recruitment(df, channel=chan)
    print(f'calculated recruitment for: {len(df)} rows')
    df_well = _group_by_well(df)
    print(f'found: {len(df_well)} wells')
    
    df_well = df_well[df_well['cells_per_well'] >= cells_per_well]
    prc_list = df_well['prc'].unique().tolist()
    df = df[df['prc'].isin(prc_list)]
    print(f'After cells per well filter: {len(df)} cells in {len(df_well)} wells left wth threshold {cells_per_well}')
    
    if plot_control:
        _plot_controls(df, mask_chans, channel_of_interest, figuresize=5)

    print(f'PV level: {len(df)} rows')
    _plot_recruitment(df=df, df_type='by PV', channel_of_interest=channel_of_interest, target=target, figuresize=figuresize)
    print(f'well level: {len(df_well)} rows')
    _plot_recruitment(df=df_well, df_type='by well', channel_of_interest=channel_of_interest, target=target, figuresize=figuresize)
    cells,wells = _results_to_csv(src, df, df_well)
    return [cells,wells]

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
