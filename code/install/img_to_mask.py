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

def preprocess_img_data(src, metadata_type='cellvoyager', custom_regex=None, img_format='.tif', bitdepth='uint16', cmap='inferno', figuresize=15, normalize=False, nr=1, plot=False, mask_channels=[0,1,2], batch_size=[100,100,100], timelapse=False, remove_background=False, backgrounds=100, lower_quantile=0.01, save_dtype=np.float32, correct_illumination=False, randomize=True, generate_movies=False, all_to_mip=False, fps=2, pick_slice=False, skip_mode='01',settings={}):
    
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




def identify_masks(src, object_type, model_name, batch_size, channels, diameter, minimum_size, maximum_size, flow_threshold=30, cellprob_threshold=1, figuresize=25, cmap='inferno', refine_masks=True, filter_size=True, filter_dimm=True, remove_border_objects=False, verbose=False, plot=False, merge=False, save=True, start_at=0, file_type='.npz', net_avg=True, resample=True, timelapse=False, fps=2, timelapse_displacement=None, timelapse_frame_limits=None, timelapse_memory=3, timelapse_remove_transient=False, timelapse_mode='btrack', generate_movies=False):

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

    def __npz_to_movie(arrays, filenames, save_path, fps=10):
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
                frame = np.clip(frame, 0, 1)  # Ensure values are within [0, 1]
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

    def __display_gif(path):
        with open(path, 'rb') as file:
            display(ipyimage(file.read()))

    def __save_mask_timelapse_as_gif(masks, tracks_df, path, fps, cmap, norm, filenames):

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
        anim.save(path, writer='pillow', fps=fps, dpi=80)  # Adjust DPI for size/quality
        plt.close(fig)
        print(f'Saved timelapse to {path}')

    def __visualize_and_save_timelapse_stack_with_tracks(masks, tracks_df, save, src, name, fps, plot, filenames, object_type, mode='btrack',interactive=False):
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
            __save_mask_timelapse_as_gif(masks, tracks_df, save_path_gif, fps, cmap, norm, filenames)
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
        return optimal_search_range

    def __trackpy_track_cells(src, name, batch_filenames, object_type, fps, masks, timelapse_displacement, timelapse_memory, timelapse_remove_transient, plot, save, mode):

        features = __prepare_for_tracking(masks)
        if timelapse_displacement is None:
            timelapse_displacement = __find_optimal_search_range(features, initial_search_range=500, increment=10, max_attempts=49, memory=timelapse_memory)
        try:
            tracks_df = tp.link(features, search_range=timelapse_displacement, memory=timelapse_memory)
        except Exception as e:
            print(f'timelapse_displacement={timelapse_displacement} is to high. Lower timelapse_displacement or set to None for automatic thresholding.')

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
            __visualize_and_save_timelapse_stack_with_tracks(masks, tracks_df_filter, save, src, name, fps, plot, batch_filenames, object_type, mode)

        mask_stack = __masks_to_masks_stack(masks)
        return mask_stack

    def __filter_short_tracks(df, min_length=5):
        """Filter out tracks that are shorter than min_length."""
        track_lengths = df.groupby('track_id').size()
        long_tracks = track_lengths[track_lengths >= min_length].index
        return df[df['track_id'].isin(long_tracks)]

    def __btrack_track_cells(src, name, batch_filenames, object_type, fps, plot, save, masks_3D, mode, timelapse_remove_transient, radius=100, workers=10):

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
            __visualize_and_save_timelapse_stack_with_tracks(masks, final_df, save, src, name, fps, plot, batch_filenames, object_type, mode)

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
                    if generate_movies:
                        movie_path = os.path.join(os.path.dirname(src), 'movies')
                        os.makedirs(movie_path, exist_ok=True)
                        save_path = os.path.join(movie_path, f'timelapse_{object_type}_{name}.mp4')
                        __npz_to_movie(batch, batch_filenames, save_path, fps=fps)
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
                    if timelapse_mode == 'btrack':
                        if not timelapse_displacement is None:
                            radius = timelapse_displacement
                        else:
                            radius = 100
                        
                        workers = os.cpu_count()-2
                        if workers < 1:
                            workers = 1

                        mask_stack = __btrack_track_cells(src, name, batch_filenames, object_type, fps, plot, save, masks_3D=masks, mode=timelapse_mode, timelapse_remove_transient=timelapse_remove_transient, radius=radius, workers=workers)
                    if timelapse_mode == 'trackpy':
                        mask_stack = __trackpy_track_cells(src, name, batch_filenames, object_type, fps, masks, timelapse_displacement, timelapse_memory, timelapse_remove_transient, plot, save, timelapse_mode)

                    __save_object_counts_to_database(masks, object_type, batch_filenames, count_loc, added_string='_timelapse')
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


def _generate_masks(src, object_type, mag, batch_size, channels, cellprob_threshold, plot, save, verbose, nr=1, start_at=0, merge=False, file_type='.npz', fps=2, timelapse=False, timelapse_displacement=None, timelapse_memory=3, timelapse_frame_limits=None, timelapse_remove_transient=False, timelapse_mode='btrack', generate_movies=False, settings={}):

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
                   fps=fps,
                   timelapse_displacement=timelapse_displacement,
                   timelapse_frame_limits=timelapse_frame_limits,
                   timelapse_memory=timelapse_memory,
                   timelapse_remove_transient=timelapse_remove_transient,
                   timelapse_mode=timelapse_mode,
                   generate_movies=generate_movies)
    
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
                            generate_movies=settings['timelapse'],
                            nr=settings['examples_to_plot'],
                            all_to_mip=settings['all_to_mip'],
                            fps=settings['fps'],
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
                           fps=settings['fps'],
                           timelapse_displacement=settings['timelapse_displacement'], 
                           timelapse_memory=settings['timelapse_memory'],
                           timelapse_frame_limits=settings['timelapse_frame_limits'],
                           timelapse_remove_transient=settings['timelapse_remove_transient'],
                           timelapse_mode=settings['timelapse_mode'],
                           generate_movies=settings['generate_movies'],
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
                           fps=settings['fps'],
                           timelapse_displacement=settings['timelapse_displacement'], 
                           timelapse_memory=settings['timelapse_memory'],
                           timelapse_frame_limits=settings['timelapse_frame_limits'],
                           timelapse_remove_transient=settings['timelapse_remove_transient'],
                           timelapse_mode=settings['timelapse_mode'],
                           generate_movies=settings['generate_movies'],
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
                           fps=settings['fps'],
                           timelapse_displacement=settings['timelapse_displacement'], 
                           timelapse_memory=settings['timelapse_memory'],
                           timelapse_frame_limits=settings['timelapse_frame_limits'],
                           timelapse_remove_transient=settings['timelapse_remove_transient'],
                           timelapse_mode=settings['timelapse_mode'],
                           generate_movies=settings['generate_movies'],
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

