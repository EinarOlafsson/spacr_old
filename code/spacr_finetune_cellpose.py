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

    pip_packages = ["opencv-python", "numpy==1.24.0", "numba==0.58.0"]
    
    for package in pip_packages:
    	print(f"Installing {package}")
    	subprocess.run([pip_PATH, "install", package])

    print("Dependencies installation complete.")

env_name = "spacr_finetune_cellpose"

conda_PATH, python_PATH, pip_PATH, env_PATH = get_paths(env_name)

dependencies = ["pandas", "ipykernel", "scikit-learn", "scikit-image", "scikit-learn", "seaborn", "matplotlib", "ipywidgets"]

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

import os, gc, cv2, random, time, torch, re, warnings, imageio

print('Torch available:', torch.cuda.is_available())
print('CUDA version:',torch.version.cuda)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cellpose import models, io
from skimage.exposure import rescale_intensity
from collections import deque
from matplotlib.patches import Polygon
import matplotlib as mpl
from cellpose.io import imread
from glob import glob

from skimage.io import imshow
from skimage.io import imread as scikitimread
from skimage.transform import resize as resizescikit

import sqlite3

import imageio.v2 as imageio2
from skimage import img_as_uint
import numpy as np
import pandas as pd
import seaborn as sns

from skimage.morphology import binary_dilation, binary_erosion
from skimage.metrics import adapted_rand_error as rand_error
from skimage.segmentation import find_boundaries

from skimage.measure import label, regionprops
from sklearn.metrics import precision_recall_curve

import warnings

# Filter out the specific warning
warnings.filterwarnings('ignore', message='Downcasting int32 to uint16 without scaling because max value*')
warnings.filterwarnings('ignore', message="set_ticklabels() should only be used with a fixed number of ticks*")

def generate_cellpose_train_set(folders, dst, min_objects=5):
    os.makedirs(dst, exist_ok=True)
    os.makedirs(os.path.join(dst,'masks'), exist_ok=True)
    os.makedirs(os.path.join(dst,'imgs'), exist_ok=True)
    
    for folder in folders:
        mask_folder = os.path.join(folder, 'masks')
        experiment_id = os.path.basename(folder)
        for filename in os.listdir(mask_folder):  # List the contents of the directory
            path = os.path.join(mask_folder, filename)
            img_path = os.path.join(folder, filename)
            newname = experiment_id + '_' + filename
            new_mask = os.path.join(dst, 'masks', newname)
            new_img = os.path.join(dst, 'imgs', newname)

            mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                print(f"Error reading {path}, skipping.")
                continue

            nr_of_objects = len(np.unique(mask)) - 1  # Assuming 0 is background
            if nr_of_objects >= min_objects:  # Use >= to include min_objects
                try:
                    shutil.copy(path, new_mask)
                    shutil.copy(img_path, new_img)
                except Exception as e:
                    print(f"Error copying {path} to {new_mask}: {e}")

def normalize_to_dtype(array, q1=2, q2=98, percentiles=None):
    if len(array.shape) == 2:
        array = np.expand_dims(array, axis=-1)
    num_channels = array.shape[-1]
    new_stack = np.empty_like(array)
    for channel in range(num_channels):
        img = array[..., channel]
        non_zero_img = img[img > 0]
        if non_zero_img.size > 0:
            img_min = np.percentile(non_zero_img, q1)
            img_max = np.percentile(non_zero_img, q2)
        else:
            img_min, img_max = (percentiles[channel] if percentiles and channel < len(percentiles)
                                else (img.min(), img.max()))
        new_stack[..., channel] = rescale_intensity(img, in_range=(img_min, img_max), out_range='dtype')
    if new_stack.shape[-1] == 1:
        new_stack = np.squeeze(new_stack, axis=-1)
    return new_stack

def plot_arrays(src, figuresize=50, cmap='inferno', nr=1, normalize=True, q1=1, q2=99):
    paths = []
    for file in os.listdir(src):
        if file.endswith('.tif') or file.endswith('.tiff'):
            path = os.path.join(src, file)
            paths.append(path)
    paths = random.sample(paths, nr)
    for path in paths:
        print(f'Image path:{path}')
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if normalize:
            img = normalize_to_dtype(array=img, q1=q1, q2=q2)
        dim = img.shape
        if len(img.shape) > 2:
            array_nr = img.shape[2]
            fig, axs = plt.subplots(1, array_nr, figsize=(figuresize, figuresize))
            for channel in range(array_nr):
                i = np.take(img, [channel], axis=2)
                axs[channel].imshow(i, cmap=plt.get_cmap(cmap))
                axs[channel].set_title('Channel '+str(channel), size=24)
                axs[channel].axis('off')
        else:
            fig, ax = plt.subplots(1, 1, figsize=(figuresize, figuresize))
            ax.imshow(img, cmap=plt.get_cmap(cmap))
            ax.set_title('Channel 0', size=24)
            ax.axis('off')
        fig.tight_layout()
        plt.show()
    return

def print_mask_and_flows_v1(stack, mask, flows):
    # Create subplots: 1 for image, 1 for mask, rest for each flow
    fig, axs = plt.subplots(1,  3, figsize=(40, 5))
    # Plot the original image
    axs[0].imshow(stack[:, :, 0], cmap='gray')
    #axs[0].imshow(stack, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    # Plot the mask
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('Mask')
    axs[1].axis('off')
    # Plot flow
    axs[2].imshow(flows[0], cmap='jet')
    axs[2].set_title(f'Flows')
    axs[2].axis('off')
    fig.tight_layout()
    plt.show()
    
def print_mask_and_flows_v2(stack, mask, flows):
    # Create subplots: 1 for image, 1 for mask, rest for each flow
    fig, axs = plt.subplots(1, 3, figsize=(30, 30))
    
    # Check if stack is 2D (grayscale) or 3D (color) and plot accordingly
    if stack.ndim == 2:
        axs[0].imshow(stack, cmap='gray')  # Grayscale image
    elif stack.ndim == 3:
        axs[0].imshow(stack[:, :, 0], cmap='gray')  # First channel of color image
    else:
        raise ValueError("Unexpected stack dimensionality.")
    
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    # Plot the mask
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('Mask')
    axs[1].axis('off')
    
    # Plot flow - assuming flows is a list with flow images
    # Check if flows is not empty and has the expected structure
    if flows and isinstance(flows, list) and flows[0].ndim in [2, 3]:
        flow_image = flows[0]
        if flow_image.ndim == 3:
            flow_image = flow_image[:, :, 0]  # Show first channel if it's 3D
        axs[2].imshow(flow_image, cmap='jet')
        axs[2].set_title('Flows')
        axs[2].axis('off')
    else:
        raise ValueError("Unexpected flow dimensionality or structure.")
    
    fig.tight_layout()
    plt.show()
    
def generate_mask_random_cmap(mask):
    unique_labels = np.unique(mask)
    num_objects = len(unique_labels[unique_labels != 0])
    random_colors = np.random.rand(num_objects + 1, 4)  # RGBA colors
    random_colors[:, 3] = 1  # Set alpha to 1
    random_colors[0, :] = [0, 0, 0, 1]  # Background color
    random_cmap = mpl.colors.ListedColormap(random_colors)
    return random_cmap

def print_mask_and_flows(stack, mask, flows, overlay=False):
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))  # Adjust subplot layout
    
    if stack.shape[-1] == 1:
        stack = np.squeeze(stack)
    
    # Display original image or its first channel
    if stack.ndim == 2:
        axs[0].imshow(stack, cmap='gray')
    elif stack.ndim == 3:
        axs[0].imshow(stack)
    else:
        raise ValueError("Unexpected stack dimensionality.")

    axs[0].set_title('Original Image')
    axs[0].axis('off')
    

    # Overlay mask on original image if overlay is True
    if overlay:
        mask_cmap = generate_mask_random_cmap(mask)  # Generate random colormap for mask
        mask_overlay = np.ma.masked_where(mask == 0, mask)  # Mask background
        outlines = find_boundaries(mask, mode='thick')  # Find mask outlines

        if stack.ndim == 2 or stack.ndim == 3:
            axs[1].imshow(stack, cmap='gray' if stack.ndim == 2 else None)
            axs[1].imshow(mask_overlay, cmap=mask_cmap, alpha=0.5)  # Overlay mask
            axs[1].contour(outlines, colors='r', linewidths=2)  # Add red outlines with thickness 2
    else:
        axs[1].imshow(mask, cmap='gray')
    
    axs[1].set_title('Mask with Overlay' if overlay else 'Mask')
    axs[1].axis('off')

    # Display flow image or its first channel
    if flows and isinstance(flows, list) and flows[0].ndim in [2, 3]:
        flow_image = flows[0]
        if flow_image.ndim == 3:
            flow_image = flow_image[:, :, 0]  # Use first channel for 3D
        axs[2].imshow(flow_image, cmap='jet')
    else:
        raise ValueError("Unexpected flow dimensionality or structure.")
    
    axs[2].set_title('Flows')
    axs[2].axis('off')

    fig.tight_layout()
    plt.show()
    
def load_images_and_labels(image_files, label_files, circular=False, invert=False, image_extension="*.tif", label_extension="*.tif"):
    images = []
    labels = []
    
    if not image_files is None:
        image_names = sorted([os.path.basename(f) for f in image_files])
    else:
        image_names = []
        
    if not label_files is None:
        label_names = sorted([os.path.basename(f) for f in label_files])
    else:
        label_names = []

    if not image_files is None and not label_files is None: 
        for img_file, lbl_file in zip(image_files, label_files):
            image = imread(img_file)
            if invert:
                image = invert_image(image)
            if circular:
                image = apply_mask(image, output_value=0)
            label = imread(lbl_file)
            if image.max() > 1:
                image = image / image.max()
            images.append(image)
            labels.append(label)
    elif not image_files is None:
        for img_file in image_files:
            image = imread(img_file)
            if invert:
                image = invert_image(image)
            if circular:
                image = apply_mask(image, output_value=0)
            if image.max() > 1:
                image = image / image.max()
            images.append(image)
    elif not image_files is None:
            for lbl_file in label_files:
                label = imread(lbl_file)
                if circular:
                    label = apply_mask(label, output_value=0)
            labels.append(label)
            
    if not image_files is None:
        image_dir = os.path.dirname(image_files[0])
    else:
        image_dir = None
        
    if not label_files is None:
        label_dir = os.path.dirname(label_files[0])
    else:
        label_dir = None
    
    # Log the number of loaded images and labels
    print(f'Loaded {len(images)} images and {len(labels)} labels from {image_dir} and {label_dir}')
    print(f'image shape: {images[0].shape}, image type: images[0].shape mask shape: {labels[0].shape}, image type: labels[0].shape')
    return images, labels, image_names, label_names

def load_normalized_images_and_labels(image_files, label_files, signal_thresholds=[1000], channels=None, percentiles=None,  circular=False, invert=False, visualize=False):
    
    if isinstance(signal_thresholds, int):
        signal_thresholds = [signal_thresholds] * (len(channels) if channels is not None else 1)
    elif not isinstance(signal_thresholds, list):
        signal_thresholds = [signal_thresholds]

    images = []
    labels = []
    
    num_channels = 4
    percentiles_1 = [[] for _ in range(num_channels)]
    percentiles_99 = [[] for _ in range(num_channels)]

    image_names = [os.path.basename(f) for f in image_files]
    
    if label_files is not None:
        label_names = [os.path.basename(f) for f in label_files]

    # Load images and check percentiles
    for i,img_file in enumerate(image_files):
        image = imread(img_file)
        if invert:
            image = invert_image(image)
        if circular:
            image = apply_mask(image, output_value=0)
            
        # If specific channels are specified, select them
        if channels is not None and image.ndim == 3:
            image = image[..., channels]
        
        if image.ndim < 3:
            image = np.expand_dims(image, axis=-1)
        
        images.append(image)
        if percentiles is None:
            for c in range(image.shape[-1]):
                p1 = np.percentile(image[..., c], 1)
                percentiles_1[c].append(p1)
                for percentile in [99, 99.9, 99.99, 99.999]:
                    p = np.percentile(image[..., c], percentile)
                    if p > signal_thresholds[min(c, len(signal_thresholds)-1)]:
                        percentiles_99[c].append(p)
                        break
                    
    if not percentiles is None:
        normalized_images = []
        for image in images:
            normalized_image = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[-1]):
                high_p = np.percentile(image[..., c], percentiles[1])
                low_p = np.percentile(image[..., c], percentiles[0])
                normalized_image[..., c] = rescale_intensity(image[..., c], in_range=(low_p, high_p), out_range=(0, 1))
            normalized_images.append(normalized_image)
            if visualize:
                normalize_and_visualize(image, normalized_image, title=f"Channel {c+1} Normalized")
            
    if percentiles is None:
        # Calculate average percentiles for normalization
        avg_p1 = [np.mean(p) for p in percentiles_1]
        avg_p99 = [np.mean(p) if len(p) > 0 else np.mean(percentiles_1[i]) for i, p in enumerate(percentiles_99)]

        normalized_images = []
        for image in images:
            normalized_image = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[-1]):
            normalized_image[..., c] = rescale_intensity(image[..., c], in_range=(avg_p1[c], avg_p99[c]), out_range=(0, 1))
        normalized_images.append(normalized_image)
        if visualize:
            normalize_and_visualize(image, normalized_image, title=f"Channel {c+1} Normalized")
            
    if not image_files is None:
        image_dir = os.path.dirname(image_files[0])
    else:
        image_dir = None
            
    if label_files is not None:
        for lbl_file in label_files:
            labels.append(imread(lbl_file))
    else:
        label_names = []
        label_dir = None

    print(f'Loaded and normalized {len(normalized_images)} images and {len(labels)} labels from {image_dir} and {label_dir}')
    
    return normalized_images, labels, image_names, label_names 
    
def identify_masks(src, dst, model_name, channels, diameter, batch_size, flow_threshold=30, cellprob_threshold=1, figuresize=25, cmap='inferno', verbose=False, plot=False, save=False, custom_model=None, signal_thresholds=1000, normalize=True, resize=False, target_height=None, target_width=None, rescale=True, resample=True, net_avg=False, invert=False, circular=False, percentiles=None, overlay=True, grayscale=False):
    print('========== generating masks ==========')
    print('Torch available:', torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if custom_model == None:
        if model_name =='cyto':
            model = models.CellposeModel(gpu=True, model_type=model_name, net_avg=False, diam_mean=diameter, pretrained_model=None)
        else:
            model = models.CellposeModel(gpu=True, model_type=model_name)

    if custom_model != None:
        model = models.CellposeModel(gpu=torch.cuda.is_available(), model_type=None, pretrained_model=custom_model, diam_mean=diameter, device=device, net_avg=False)  #Assuming diameter is defined elsewhere 
        print(f'loaded custom model:{custom_model}')

    chans = [2, 1] if model_name == 'cyto2' else [0,0] if model_name == 'nuclei' else [1,0] if model_name == 'cyto' else [2, 0]
    
    if grayscale:
        chans=[0, 0]
    
    print(f'Using channels: {chans} for model of type {model_name}')
    
    if verbose == True:
        print(f'Cellpose settings: Model: {model_name}, channels: {channels}, cellpose_chans: {chans}, diameter:{diameter}, flow_threshold:{flow_threshold}, cellprob_threshold:{cellprob_threshold}')
        
    all_image_files = get_files_from_dir(src, file_extension="*.tif")
    random.shuffle(all_image_files)
    
    time_ls = []
    for i in range(0, len(all_image_files), batch_size):
        image_files = all_image_files[i:i+batch_size]
        if normalize:
            images, _, image_names, _ = load_normalized_images_and_labels(image_files=image_files, label_files=None, signal_thresholds=signal_thresholds, channels=channels, percentiles=percentiles,  circular=circular, invert=invert, visualize=verbose)
            images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in images]
            orig_dims = [(image.shape[1], image.shape[0]) for image in images]
        else:
            images, _, image_names, _ = load_images_and_labels(image_files=image_files, label_files=None, circular=circular, invert=invert) 
            images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in images]
            orig_dims = [(image.shape[1], image.shape[0]) for image in images]
        if resize:
            images, _ = resize_images_and_labels(images, None, target_height, target_width, True)

        for file_index, stack in enumerate(images):

            start = time.time()
            output = model.eval(x=stack,
                         normalize=False,
                         channels=chans,
                         channel_axis=3,
                         diameter=diameter,
                         flow_threshold=flow_threshold,
                         cellprob_threshold=cellprob_threshold,
                         rescale=rescale,
                         resample=resample,
                         net_avg=net_avg,
                         progress=False)

            if len(output) == 4:
                mask, flows, _, _ = output
            elif len(output) == 3:
                mask, flows, _ = output
            else:
                raise ValueError("Unexpected number of return values from model.eval()")

            if resize:
                dims = orig_dims[file_index]
                mask = resizescikit(mask, dims, order=0, preserve_range=True, anti_aliasing=False).astype(mask.dtype)

            stop = time.time()
            duration = (stop - start)
            time_ls.append(duration)
            average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
            print(f'Processing {file_index+1}/{len(images)} images : Time/image {average_time:.3f} sec', end='\r', flush=True)
            if plot:
                if resize:
                    stack = resizescikit(stack, dims, preserve_range=True, anti_aliasing=False).astype(stack.dtype)
                print_mask_and_flows(stack, mask, flows, overlay=overlay)
            if save:
                output_filename = os.path.join(dst, image_names[file_index])
                cv2.imwrite(output_filename, mask)
    return

def get_files_from_dir(dir_path, file_extension="*"):
    return glob(os.path.join(dir_path, file_extension))
    
def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask
    
def apply_mask(image, output_value=0):
    h, w = image.shape[:2]  # Assuming image is grayscale or RGB
    mask = create_circular_mask(h, w)
    
    # If the image has more than one channel, repeat the mask for each channel
    if len(image.shape) > 2:
        mask = np.repeat(mask[:, :, np.newaxis], image.shape[2], axis=2)
    
    # Apply the mask - set pixels outside of the mask to output_value
    masked_image = np.where(mask, image, output_value)
    return masked_image
    
def invert_image(image):
    # The maximum value depends on the image dtype (e.g., 255 for uint8)
    max_value = np.iinfo(image.dtype).max
    inverted_image = max_value - image
    return inverted_image  


def normalize_and_visualize(image, normalized_image, title=""):
    """Utility function for visualization"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    if image.ndim == 3:  # Multi-channel image
        ax[0].imshow(np.mean(image, axis=-1), cmap='gray')  # Display the average over channels for visualization
    else:  # Grayscale image
        ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original " + title)
    ax[0].axis('off')

    if normalized_image.ndim == 3:
        ax[1].imshow(np.mean(normalized_image, axis=-1), cmap='gray')  # Similarly, display the average over channels
    else:
        ax[1].imshow(normalized_image, cmap='gray')
    ax[1].set_title("Normalized " + title)
    ax[1].axis('off')
    
    plt.show()

def plot_resize(images, resized_images, labels, resized_labels):
    # Display an example image and label before and after resizing
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    
    # Check if the image is grayscale; if so, add a colormap and keep dimensions correct
    if images[0].ndim == 2:  # Grayscale image
        ax[0, 0].imshow(images[0], cmap='gray')
    else:  # RGB or RGBA image
        ax[0, 0].imshow(images[0])
    ax[0, 0].set_title('Original Image')

    if resized_images[0].ndim == 2:  # Grayscale image
        ax[0, 1].imshow(resized_images[0], cmap='gray')
    else:  # RGB or RGBA image
        ax[0, 1].imshow(resized_images[0])
    ax[0, 1].set_title('Resized Image')

    # Assuming labels are always grayscale (most common scenario)
    ax[1, 0].imshow(labels[0], cmap='gray')
    ax[1, 0].set_title('Original Label')
    ax[1, 1].imshow(resized_labels[0], cmap='gray')
    ax[1, 1].set_title('Resized Label')
    plt.show()

def resize_images_and_labels(images, labels, target_height, target_width, show_example=True):
    resized_images = []
    resized_labels = []
    if not images is None and not labels is None:
        for image, label in zip(images, labels):

            if image.ndim == 2:
                image_shape = (target_height, target_width)
            elif image.ndim == 3:
                image_shape = (target_height, target_width, image.shape[-1])
                
            resized_image = resizescikit(image, image_shape, preserve_range=True, anti_aliasing=True).astype(image.dtype)
            resized_label = resizescikit(label, (target_height, target_width), order=0, preserve_range=True, anti_aliasing=False).astype(label.dtype)
            
            if resized_image.shape[-1] == 1:
                resized_image = np.squeeze(resized_image)
            
            resized_images.append(resized_image)
            resized_labels.append(resized_label)
    
    elif not images is None:
        for image in images:
        
            if image.ndim == 2:
                image_shape = (target_height, target_width)
            elif image.ndim == 3:
                image_shape = (target_height, target_width, image.shape[-1])
                
            resized_image = resizescikit(image, image_shape, preserve_range=True, anti_aliasing=True).astype(image.dtype)
            
            if resized_image.shape[-1] == 1:
                resized_image = np.squeeze(resized_image)
            
            resized_images.append(resized_image)
            
    elif not labels is None:
        for label in labels:
            resized_label = resizescikit(label, (target_height, target_width), order=0, preserve_range=True, anti_aliasing=False).astype(label.dtype)
            resized_labels.append(resized_label)
        
    if show_example:     
        if not images is None and not labels is None:
            plot_resize(images, resized_images, labels, resized_labels)
        elif not images is None:
            plot_resize(images, resized_images, images, resized_images)
        elif not labels is None:
            plot_resize(labels, resized_labels, labels, resized_labels)
    
    return resized_images, resized_labels
    
def resize_labels_back_v1(labels, orig_dims):
    resized_labels = []

    for i, label in enumerate(labels):
        resized_label = resizescikit(label, (orig_dims), order=0, preserve_range=True, anti_aliasing=False).astype(label.dtype)
        resized_labels.append(resized_label)
    
    return resized_labels
    
def resize_labels_back(labels, orig_dims):
    resized_labels = []

    if len(labels) != len(orig_dims):
        raise ValueError("The length of labels and orig_dims must match.")

    for label, dims in zip(labels, orig_dims):
        # Ensure dims is a tuple of two integers (width, height)
        if not isinstance(dims, tuple) or len(dims) != 2:
            raise ValueError("Each element in orig_dims must be a tuple of two integers representing the original dimensions (width, height)")

        resized_label = resize(label, dims, order=0, preserve_range=True, anti_aliasing=False).astype(label.dtype)
        resized_labels.append(resized_label)

    return resized_labels
    

def train_cellpose(settings):

    img_src = settings['img_src'] 
    mask_src= settings['mask_src']
    secondary_image_dir = None
    model_name = settings['model_name']
    model_type = settings['model_type']
    learning_rate = settings['learning_rate']
    weight_decay = settings['weight_decay']
    batch_size = settings['batch_size']
    n_epochs = settings['n_epochs']
    verbose = settings['verbose']
    signal_thresholds = settings['signal_thresholds']
    channels = settings['channels']
    from_scratch = settings['from_scratch']
    diameter = settings['diameter']
    resize = settings['resize']
    rescale = settings['rescale']
    normalize = settings['normalize']
    target_height = settings['width_height'][1]
    target_width = settings['width_height'][0]
    circular = settings['circular']
    invert = settings['invert']
    percentiles = settings['percentiles']
    grayscale = settings['grayscale']
    
    print(settings)

    if from_scratch:
    	model_name=f'scratch_{model_name}_{model_type}_e{n_epochs}_X{target_width}_Y{target_height}.CP_model'
    else:
    	model_name=f'{model_name}_{model_type}_e{n_epochs}_X{target_width}_Y{target_height}.CP_model'
    	
    model_save_path = os.path.join(mask_src, 'models', 'cellpose_model')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    settings_df = pd.DataFrame(list(settings.items()), columns=['Key', 'Value'])
    settings_csv = os.path.join(model_save_path,f'{model_name}_settings.csv')
    settings_df.to_csv(settings_csv, index=False)
    
    if model_type =='cyto':
        if not from_scratch:
    	    model = models.CellposeModel(gpu=True, model_type=model_type)
        else:
    	    model = models.CellposeModel(gpu=True, model_type=model_type, net_avg=False, diam_mean=diameter, pretrained_model=None)
    if model_type !='cyto':
        model = models.CellposeModel(gpu=True, model_type=model_type)
        
    
    
    if normalize:    	
    	images, masks, image_names, mask_names = load_normalized_images_and_labels(image_dir=img_src, label_dir=mask_src, secondary_image_dir=secondary_image_dir, signal_thresholds=signal_thresholds, channels=channels, percentiles=percentiles,  circular=circular, invert=invert, visualize=verbose)
    	images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in images]
    else:
        images, masks, image_names, mask_names = load_images_and_labels(img_src, mask_src, circular, invert)
        images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in images]
    
    if resize:
        images, masks = resize_images_and_labels(images, masks, target_height, target_width, show_example=True)

    if model_type == 'cyto':
        cp_channels = [0,1]
    if model_type == 'cyto2':
        cp_channels = [0,2]
    if model_type == 'nucleus':
        cp_channels = [0,0]
    if grayscale:
        cp_channels = [0,0]
        images = [np.squeeze(img) if img.ndim == 3 and 1 in img.shape else img for img in images]
    
    masks = [np.squeeze(mask) if mask.ndim == 3 and 1 in mask.shape else mask for mask in masks]

    print(f'image shape: {images[0].shape}, image type: images[0].shape mask shape: {masks[0].shape}, image type: masks[0].shape')
    save_every = int(n_epochs/10)
    print('cellpose image input dtype', images[0].dtype)
    print('cellpose mask input dtype', masks[0].dtype)
    # Train the model
    model.train(train_data=images, #(list of arrays (2D or 3D)) – images for training
                train_labels=masks, #(list of arrays (2D or 3D)) – labels for train_data, where 0=no masks; 1,2,…=mask labels can include flows as additional images
                train_files=image_names, #(list of strings) – file names for images in train_data (to save flows for future runs)
                channels=cp_channels, #(list of ints (default, None)) – channels to use for training
                normalize=False, #(bool (default, True)) – normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel
                save_path=model_save_path, #(string (default, None)) – where to save trained model, if None it is not saved
                save_every=save_every, #(int (default, 100)) – save network every [save_every] epochs
                learning_rate=learning_rate, #(float or list/np.ndarray (default, 0.2)) – learning rate for training, if list, must be same length as n_epochs
                n_epochs=n_epochs, #(int (default, 500)) – how many times to go through whole training set during training
                weight_decay=weight_decay, #(float (default, 0.00001)) –
                SGD=True, #(bool (default, True)) – use SGD as optimization instead of RAdam
                batch_size=batch_size, #(int (optional, default 8)) – number of 224x224 patches to run simultaneously on the GPU (can make smaller or bigger depending on GPU memory usage)
                nimg_per_epoch=None, #(int (optional, default None)) – minimum number of images to train on per epoch, with a small training set (< 8 images) it may help to set to 8
                rescale=rescale, #(bool (default, True)) – whether or not to rescale images to diam_mean during training, if True it assumes you will fit a size model after training or resize your images accordingly, if False it will try to train the model to be scale-invariant (works worse)
                min_train_masks=1, #(int (default, 5)) – minimum number of masks an image must have to use in training set
                model_name=model_name) #(str (default, None)) – name of network, otherwise saved with name as params + training start time 

    return print(f"Model saved at: {model_save_path}/{model_name}")

def generate_cp_masks(settings):
    
    src = settings['src']
    model_name = settings['model_name']
    channels = settings['channels']
    diameter = settings['diameter']
    regex = '.tif'
    #flow_threshold = 30
    cellprob_threshold = settings['cellprob_threshold']
    figuresize = 25
    cmap = 'inferno'
    verbose = settings['verbose']
    plot = settings['plot']
    save = settings['save']
    custom_model = settings['custom_model']
    signal_thresholds = 1000
    normalize = settings['normalize']
    resize = settings['resize']
    target_height = settings['width_height'][1]
    target_width = settings['width_height'][0]
    rescale = settings['rescale']
    resample = settings['resample']
    net_avg = settings['net_avg']
    invert = settings['invert']
    circular = settings['circular']
    percentiles = settings['percentiles']
    overlay = settings['overlay']
    grayscale = settings['grayscale']
    flow_threshold = settings['flow_threshold']
    batch_size = settings['batch_size']
    
    dst = os.path.join(src,'masks')
    os.makedirs(dst, exist_ok=True)
		   
    identify_masks(src, dst, model_name, channels, diameter, batch_size, flow_threshold, cellprob_threshold, figuresize, cmap, verbose, plot, save, custom_model, signal_thresholds, normalize, resize, target_height, target_width, rescale, resample, net_avg, invert, circular, percentiles, overlay, grayscale)

def read_mask(mask_path):
    mask = imageio2.imread(mask_path)
    if mask.dtype != np.uint16:
        mask = img_as_uint(mask)
    return mask
    
def visualize_masks(mask1, mask2, mask3, title="Masks Comparison"):
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    for ax, mask, title in zip(axs, [mask1, mask2, mask3], ['Mask 1', 'Mask 2', 'Mask 3']):
        cmap = generate_mask_random_cmap(mask)
        # If the mask is binary, we can skip normalization
        if np.isin(mask, [0, 1]).all():
            ax.imshow(mask, cmap=cmap)
        else:
            # Normalize the image for displaying purposes
            norm = plt.Normalize(vmin=0, vmax=mask.max())
            ax.imshow(mask, cmap=cmap, norm=norm)
        ax.set_title(title)
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

def calculate_iou(mask1, mask2):
    mask1, mask2 = pad_to_same_shape(mask1, mask2)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0
    
def match_masks(true_masks, pred_masks, iou_threshold):
    matches = []
    matched_true_masks_indices = set()  # Use set to store indices of matched true masks

    for pred_mask in pred_masks:
        for true_mask_index, true_mask in enumerate(true_masks):
            if true_mask_index not in matched_true_masks_indices:
                iou = calculate_iou(true_mask, pred_mask)
                if iou >= iou_threshold:
                    matches.append((true_mask, pred_mask))
                    matched_true_masks_indices.add(true_mask_index)  # Store the index of the matched true mask
                    break  # Move on to the next predicted mask
    return matches
    
def compute_average_precision(matches, num_true_masks, num_pred_masks):
    TP = len(matches)
    FP = num_pred_masks - TP
    FN = num_true_masks - TP
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    return precision, recall

def pad_to_same_shape(mask1, mask2):
    # Find the shape differences
    shape_diff = np.array([max(mask1.shape[0], mask2.shape[0]) - mask1.shape[0], 
                           max(mask1.shape[1], mask2.shape[1]) - mask1.shape[1]])
    pad_mask1 = ((0, shape_diff[0]), (0, shape_diff[1]))
    shape_diff = np.array([max(mask1.shape[0], mask2.shape[0]) - mask2.shape[0], 
                           max(mask1.shape[1], mask2.shape[1]) - mask2.shape[1]])
    pad_mask2 = ((0, shape_diff[0]), (0, shape_diff[1]))
    
    padded_mask1 = np.pad(mask1, pad_mask1, mode='constant', constant_values=0)
    padded_mask2 = np.pad(mask2, pad_mask2, mode='constant', constant_values=0)
    
    return padded_mask1, padded_mask2
    
def compute_ap_over_iou_thresholds(true_masks, pred_masks, iou_thresholds):
    precision_recall_pairs = []
    for iou_threshold in iou_thresholds:
        matches = match_masks(true_masks, pred_masks, iou_threshold)
        precision, recall = compute_average_precision(matches, len(true_masks), len(pred_masks))
        # Check that precision and recall are within the range [0, 1]
        if not 0 <= precision <= 1 or not 0 <= recall <= 1:
            raise ValueError(f'Precision or recall out of bounds. Precision: {precision}, Recall: {recall}')
        precision_recall_pairs.append((precision, recall))

    # Sort by recall values
    precision_recall_pairs = sorted(precision_recall_pairs, key=lambda x: x[1])
    sorted_precisions = [p[0] for p in precision_recall_pairs]
    sorted_recalls = [p[1] for p in precision_recall_pairs]
    return np.trapz(sorted_precisions, x=sorted_recalls)
    
def compute_segmentation_ap(true_masks, pred_masks, iou_thresholds=np.linspace(0.5, 0.95, 10)):
    true_mask_labels = label(true_masks)
    pred_mask_labels = label(pred_masks)
    true_mask_regions = [region.image for region in regionprops(true_mask_labels)]
    pred_mask_regions = [region.image for region in regionprops(pred_mask_labels)]
    return compute_ap_over_iou_thresholds(true_mask_regions, pred_mask_regions, iou_thresholds)

def compute_ap(ious, threshold=0.5):
    tp = sum(iou >= threshold for iou in ious)
    fp = sum(iou < threshold for iou in ious)
    fn = len(true_masks) - len(set(matched_indices))  # Assuming true_masks is accessible
    if (tp + fp + fn) == 0:
        return 0
    else:
        return tp / (tp + fp + fn)

def jaccard_index(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)

def dice_coefficient(mask1, mask2):
    # Convert to binary masks
    mask1 = np.where(mask1 > 0, 1, 0)
    mask2 = np.where(mask2 > 0, 1, 0)

    # Calculate intersection and total
    intersection = np.sum(mask1 & mask2)
    total = np.sum(mask1) + np.sum(mask2)
    
    # Handle the case where both masks are empty
    if total == 0:
        return 1.0
    
    # Return the Dice coefficient
    return 2.0 * intersection / total

def extract_boundaries(mask, dilation_radius=1):
    binary_mask = (mask > 0).astype(np.uint8)
    struct_elem = np.ones((dilation_radius*2+1, dilation_radius*2+1))
    dilated = binary_dilation(binary_mask, footprint=struct_elem)
    eroded = binary_erosion(binary_mask, footprint=struct_elem)
    boundary = dilated ^ eroded
    return boundary

def boundary_f1_score(mask_true, mask_pred, dilation_radius=1):
    # Assume extract_boundaries is defined to extract object boundaries with given dilation_radius
    boundary_true = extract_boundaries(mask_true, dilation_radius)
    boundary_pred = extract_boundaries(mask_pred, dilation_radius)
    
    # Calculate intersection of boundaries
    intersection = np.logical_and(boundary_true, boundary_pred)
    
    # Calculate precision and recall for boundary detection
    precision = np.sum(intersection) / (np.sum(boundary_pred) + 1e-6)
    recall = np.sum(intersection) / (np.sum(boundary_true) + 1e-6)
    
    # Calculate F1 score as harmonic mean of precision and recall
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return f1

def plot_comparison_results(comparison_results):
    df = pd.DataFrame(comparison_results)
    df_melted = pd.melt(df, id_vars=['filename'], var_name='metric', value_name='value')
    df_jaccard = df_melted[df_melted['metric'].str.contains('jaccard')]
    df_dice = df_melted[df_melted['metric'].str.contains('dice')]
    df_boundary_f1 = df_melted[df_melted['metric'].str.contains('boundary_f1')]
    df_ap = df_melted[df_melted['metric'].str.contains('average_precision')]
    fig, axs = plt.subplots(1, 4, figsize=(40, 10))
    
    # Jaccard Index Plot
    sns.boxplot(data=df_jaccard, x='metric', y='value', ax=axs[0], color='lightgrey')
    sns.stripplot(data=df_jaccard, x='metric', y='value', ax=axs[0], jitter=True, alpha=0.6)
    axs[0].set_title('Jaccard Index by Comparison')
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, horizontalalignment='right')
    axs[0].set_xlabel('Comparison')
    axs[0].set_ylabel('Jaccard Index')
    # Dice Coefficient Plot
    sns.boxplot(data=df_dice, x='metric', y='value', ax=axs[1], color='lightgrey')
    sns.stripplot(data=df_dice, x='metric', y='value', ax=axs[1], jitter=True, alpha=0.6)
    axs[1].set_title('Dice Coefficient by Comparison')
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, horizontalalignment='right')
    axs[1].set_xlabel('Comparison')
    axs[1].set_ylabel('Dice Coefficient')
    # Border F1 scores
    sns.boxplot(data=df_boundary_f1, x='metric', y='value', ax=axs[2], color='lightgrey')
    sns.stripplot(data=df_boundary_f1, x='metric', y='value', ax=axs[2], jitter=True, alpha=0.6)
    axs[2].set_title('Boundary F1 Score by Comparison')
    axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=45, horizontalalignment='right')
    axs[2].set_xlabel('Comparison')
    axs[2].set_ylabel('Boundary F1 Score')
    # AP scores plot
    sns.boxplot(data=df_ap, x='metric', y='value', ax=axs[3], color='lightgrey')
    sns.stripplot(data=df_ap, x='metric', y='value', ax=axs[3], jitter=True, alpha=0.6)
    axs[3].set_title('Average Precision by Comparison')
    axs[3].set_xticklabels(axs[3].get_xticklabels(), rotation=45, horizontalalignment='right')
    axs[3].set_xlabel('Comparison')
    axs[3].set_ylabel('Average Precision')
    
    plt.tight_layout()
    plt.show()
    return fig
    
def compare_masks(dir1, dir2, dir3, verbose=False):
    filenames = os.listdir(dir1)
    results = []
    cond_1 = os.path.basename(dir1)
    cond_2 = os.path.basename(dir2)
    cond_3 = os.path.basename(dir3)
    for index, filename in enumerate(filenames):
        print(f'Processing image:{index+1}', end='\r', flush=True)
        path1, path2, path3 = os.path.join(dir1, filename), os.path.join(dir2, filename), os.path.join(dir3, filename)
        if os.path.exists(path2) and os.path.exists(path3):
            
            mask1, mask2, mask3 = read_mask(path1), read_mask(path2), read_mask(path3)
            boundary_true1, boundary_true2, boundary_true3 = extract_boundaries(mask1), extract_boundaries(mask2), extract_boundaries(mask3)
            
            
            true_masks, pred_masks = [mask1], [mask2, mask3]  # Assuming mask1 is the ground truth for simplicity
            true_labels, pred_labels_1, pred_labels_2 = label(mask1), label(mask2), label(mask3)
            average_precision_0, average_precision_1 = compute_segmentation_ap(mask1, mask2), compute_segmentation_ap(mask1, mask3)
            ap_scores = [average_precision_0, average_precision_1]

            if verbose:
                unique_values1, unique_values2, unique_values3 = np.unique(mask1),  np.unique(mask2), np.unique(mask3)
                print(f"Unique values in mask 1: {unique_values1}, mask 2: {unique_values2}, mask 3: {unique_values3}")
                visualize_masks(boundary_true1, boundary_true2, boundary_true3, title=f"Boundaries - {filename}")
            
            boundary_f1_12, boundary_f1_13, boundary_f1_23 = boundary_f1_score(mask1, mask2), boundary_f1_score(mask1, mask3), boundary_f1_score(mask2, mask3)

            if (np.unique(mask1).size == 1 and np.unique(mask1)[0] == 0) and \
               (np.unique(mask2).size == 1 and np.unique(mask2)[0] == 0) and \
               (np.unique(mask3).size == 1 and np.unique(mask3)[0] == 0):
                continue
            
            if verbose:
                unique_values4, unique_values5, unique_values6 = np.unique(boundary_f1_12), np.unique(boundary_f1_13), np.unique(boundary_f1_23)
                print(f"Unique values in boundary mask 1: {unique_values4}, mask 2: {unique_values5}, mask 3: {unique_values6}")
                visualize_masks(mask1, mask2, mask3, title=filename)
            
            jaccard12 = jaccard_index(mask1, mask2)
            dice12 = dice_coefficient(mask1, mask2)
            jaccard13 = jaccard_index(mask1, mask3)
            dice13 = dice_coefficient(mask1, mask3)
            jaccard23 = jaccard_index(mask2, mask3)
            dice23 = dice_coefficient(mask2, mask3)    

            results.append({
                f'filename': filename,
                f'jaccard_{cond_1}_{cond_2}': jaccard12,
                f'dice_{cond_1}_{cond_2}': dice12,
                f'jaccard_{cond_1}_{cond_3}': jaccard13,
                f'dice_{cond_1}_{cond_3}': dice13,
                f'jaccard_{cond_2}_{cond_3}': jaccard23,
                f'dice_{cond_2}_{cond_3}': dice23,
                f'boundary_f1_{cond_1}_{cond_2}': boundary_f1_12,
                f'boundary_f1_{cond_1}_{cond_3}': boundary_f1_13,
                f'boundary_f1_{cond_2}_{cond_3}': boundary_f1_23,
                f'average_precision_{cond_1}_{cond_2}': ap_scores[0],
                f'average_precision_{cond_1}_{cond_3}': ap_scores[1]
            })
        else:
            print(f'Cannot find {path1} or {path2} or {path3}')
    fig = plot_comparison_results(results)
    return results, fig
    
def analyze_plaques(folder):
    summary_data = []
    details_data = []
    
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            # Assuming each file is a NumPy array file (.npy) containing a 16-bit labeled image
            image = np.load(filepath)
            
            labeled_image = label(image)
            regions = regionprops(labeled_image)
            
            object_count = len(regions)
            sizes = [region.area for region in regions]
            average_size = np.mean(sizes) if sizes else 0
            
            summary_data.append({'file': filename, 'object_count': object_count, 'average_size': average_size})
            for size in sizes:
                details_data.append({'file': filename, 'plaque_size': size})
    
    # Convert lists to pandas DataFrames
    summary_df = pd.DataFrame(summary_data)
    details_df = pd.DataFrame(details_data)
    
    # Save DataFrames to a SQLite database
    db_name = 'plaques_analysis.db'
    conn = sqlite3.connect(db_name)
    
    summary_df.to_sql('summary', conn, if_exists='replace', index=False)
    details_df.to_sql('details', conn, if_exists='replace', index=False)
    
    conn.close()
    
    print(f"Analysis completed and saved to database '{db_name}'.")



    
##########################################################################################################################################################################################
