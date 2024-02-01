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

dependencies = ["pandas", "ipykernel", "scikit-learn", "scikit-image", "seaborn", "matplotlib", "ipywidgets"]

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

import imageio.v2 as imageio2
from skimage import img_as_uint
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation, binary_erosion
import warnings

# Filter out the specific warning
warnings.filterwarnings('ignore', message='Downcasting int32 to uint16 without scaling because max value*')
warnings.filterwarnings('ignore', message="set_ticklabels() should only be used with a fixed number of ticks*")

def generate_cellpose_dataset(src, dst, channel, number):
    if channel == 1:
        channel = '01'
    os.makedirs(dst, exist_ok=True)
    folder = os.path.join(src,channel)
    files = random.sample(os.listdir(folder), number)
    for file in files:
        path = os.path.join(folder, file)
        new_path = os.path.join(dst,file)
        shutil.copy(path,new_path)

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

def print_mask_and_flows(stack, mask, flows):
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
    
def identify_masks(paths, dst, model_name, channels, diameter, flow_threshold=30, cellprob_threshold=1, figuresize=25, cmap='inferno', verbose=False, plot=False, save=False, custom_model=None):
    print('========== generating masks ==========')
    print('Torch available:', torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if custom_model == None:
        model = models.Cellpose(gpu=True, model_type=model_name, net_avg=True, device=device)
    else:
        model_state = torch.load(custom_model, map_location=device)
        model = models.CellposeModel(gpu=True, model_type=model_name)
        model.net.load_state_dict(model_state)
        print(f'loaded custom model:{custom_model}')

    chans = [2, 1] if model_name == 'cyto2' else [0,0] if model_name == 'nuclei' else [1,0] if model_name == 'cyto' else [2, 0] 
    
    print(f'Using channels: {chans} for model of type {model_name}')
    
    if verbose == True:
        #print(f'Settings: minimum_size: {minimum_size}, maximum_size:{maximum_size}')
        print(f'Cellpose settings: Model: {model_name}, channels: {channels}, cellpose_chans: {chans}, diameter:{diameter}, flow_threshold:{flow_threshold}, cellprob_threshold:{cellprob_threshold}')
        
    time_ls = []
    
    for file_index, path in enumerate(paths):
        print(file_index, path)
        stack = cv2.imread(path).astype(np.float32)
        stack = stack[:, :, channels]
        filename = os.path.basename(path)
        start = time.time()
        stack = normalize_to_dtype(stack, q1=2,q2=98)
        
        if stack.max() > 1:
            stack = stack / stack.max()
                                        
        results = model.eval(x=stack,
                         normalize=False,
                         channels=chans,
                         channel_axis=3,
                         diameter=diameter,
                         flow_threshold=flow_threshold,
                         cellprob_threshold=cellprob_threshold,
                         rescale=None,
                         resample=True,
                         net_avg=True,
                         progress=None)

        print(len(results))

        # Unpack the results based on the number of returned values
        if len(results) == 4:
            mask, flows, _, _ = results
        elif len(results) == 3:
            mask, flows, _ = results
        else:
            raise ValueError("Unexpected number of return values from model.eval()")

        stop = time.time()
        duration = (stop - start)
        time_ls.append(duration)
        average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
        print(f'Processing {file_index+1}/{len(paths)} images : Time/image {average_time:.3f} sec', end='\r', flush=True)
        if plot:
            print_mask_and_flows(stack, mask, flows)
        if save:
            output_filename = os.path.join(dst, filename)
            cv2.imwrite(output_filename, mask)
    return
    
def train_cellpose(img_src, mask_src, model_name='toxopv', model_type='cyto', nchan=2, channels=[0, 0], learning_rate=0.2, weight_decay=1e-05, batch_size=8, n_epochs=500):

    print(f'Paramiters - model_type:{model_type} learning_rate:{learning_rate} weight_decay:{weight_decay} batch_size{batch_size} n_epochs{n_epochs}')
    
    model_name=f'{model_name}_epochs_{n_epochs}.CP_model'
    # Load training data
    train_images, train_masks, _, _, _, _ = io.load_train_test_data(img_src, mask_src, mask_filter='')

    # Create a CellposeModel instance
    model = models.CellposeModel(gpu=True, 
                                 model_type=model_type, 
                                 net_avg=True, 
                                 diam_mean=30.0, 
                                 residual_on=True, 
                                 style_on=True, 
                                 concatenation=False, 
                                 nchan=nchan)

    # Specify the save path for the model
    model_save_path = os.path.join(mask_src, 'models', 'cellpose_model')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Train the model
    model.train(train_data=train_images,
                train_labels=train_masks,
                channels=channels,  # Adjust based on your image channels
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                n_epochs=n_epochs,
                save_path=model_save_path,
                model_name=model_name)

    return print(f"Model saved at: {model_save_path}/{model_name}")

def generate_cp_masks(src, model_name, channels, diameter, regex='.tif', flow_threshold=30, cellprob_threshold=1, figuresize=25, cmap='inferno', verbose=False, plot=False, save=False, custom_model=None):
    dst = os.path.join(src,'masks')
    os.makedirs(dst, exist_ok=True)
    paths = []
    
    for filename in os.listdir(src):
        path = os.path.join(src, filename)
        
        if filename.endswith('.tif'):
            if re.search(regex, filename):
                paths.append(path)
    
    identify_masks(paths, dst, model_name, channels, diameter,  flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, figuresize=figuresize, cmap=cmap, verbose=verbose, plot=plot, save=save, custom_model=custom_model)

def read_mask(mask_path):
    mask = imageio2.imread(mask_path)
    if mask.dtype != np.uint16:
        mask = img_as_uint(mask)
    return mask

def visualize_masks(mask1, mask2, mask3, title="Masks Comparison"):
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    norm = plt.Normalize(vmin=0, vmax=np.max([mask1.max(), mask2.max(), mask3.max()]))
    axs[0].imshow(mask1, cmap='gray', norm=norm)
    axs[0].set_title('Mask 1')
    axs[0].axis('off')
    axs[1].imshow(mask2, cmap='gray', norm=norm)
    axs[1].set_title('Mask 2')
    axs[1].axis('off')
    axs[2].imshow(mask3, cmap='gray', norm=norm)
    axs[2].set_title('Mask 3')
    axs[2].axis('off')
    plt.suptitle(title)
    plt.show()
    
def visualize_masks(mask1, mask2, mask3, title="Masks Comparison"):
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    for ax, mask, title in zip(axs, [mask1, mask2, mask3], ['Mask 1', 'Mask 2', 'Mask 3']):
        # If the mask is binary, we can skip normalization
        if np.isin(mask, [0, 1]).all():
            ax.imshow(mask, cmap='gray')
        else:
            # Normalize the image for displaying purposes
            norm = plt.Normalize(vmin=0, vmax=mask.max())
            ax.imshow(mask, cmap='gray', norm=norm)
        ax.set_title(title)
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

def jaccard_index(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)

def dice_coefficient(mask1, mask2):
    intersection = np.sum(mask1[mask2 == 1]) * 2.0
    total = np.sum(mask1) + np.sum(mask2)
    return intersection / total

def extract_boundaries(mask, dilation_radius=1):
    binary_mask = (mask > 0).astype(np.uint8)
    struct_elem = np.ones((dilation_radius*2+1, dilation_radius*2+1))
    dilated = binary_dilation(binary_mask, footprint=struct_elem)
    eroded = binary_erosion(binary_mask, footprint=struct_elem)
    boundary = dilated ^ eroded
    return boundary

def boundary_f1_score(mask_true, mask_pred, dilation_radius=1):
    boundary_true = extract_boundaries(mask_true, dilation_radius)
    boundary_pred = extract_boundaries(mask_pred, dilation_radius)
    intersection = np.logical_and(boundary_true, boundary_pred)
    precision = np.sum(intersection) / (np.sum(boundary_pred) + 1e-6)
    recall = np.sum(intersection) / (np.sum(boundary_true) + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1

def compare_masks(dir1, dir2, dir3, verbose=False):
    filenames = os.listdir(dir1)
    results = []
    cond_1 = os.path.basename(dir1)
    cond_2 = os.path.basename(dir2)
    cond_3 = os.path.basename(dir3)
    for filename in filenames:
        path1 = os.path.join(dir1, filename)
        path2 = os.path.join(dir2, filename)
        path3 = os.path.join(dir3, filename)
        if os.path.exists(path2) and os.path.exists(path3):
            mask1 = read_mask(path1)
            mask2 = read_mask(path2)
            mask3 = read_mask(path3)
            boundary_true1 = extract_boundaries(mask1)
            boundary_true2 = extract_boundaries(mask2)
            boundary_true3 = extract_boundaries(mask3)
            if verbose:
                unique_values1 = np.unique(mask1)
                unique_values2 = np.unique(mask2)
                unique_values3 = np.unique(mask3)
                print(f"Unique values in mask 1: {unique_values1}")
                print(f"Unique values in mask 2: {unique_values2}")
                print(f"Unique values in mask 3: {unique_values3}")
                visualize_masks(boundary_true1, boundary_true2, boundary_true3, title=f"Boundaries - {filename}")
            boundary_f1_12 = boundary_f1_score(mask1, mask2)
            boundary_f1_13 = boundary_f1_score(mask1, mask3)
            boundary_f1_23 = boundary_f1_score(mask2, mask3)
            if (np.unique(mask1).size == 1 and np.unique(mask1)[0] == 0) and \
               (np.unique(mask2).size == 1 and np.unique(mask2)[0] == 0) and \
               (np.unique(mask3).size == 1 and np.unique(mask3)[0] == 0):
                continue
            if verbose:
                unique_values4 = np.unique(boundary_f1_12)
                unique_values5 = np.unique(boundary_f1_13)
                unique_values6 = np.unique(boundary_f1_23)
                print(f"Unique values in boundary mask 1: {unique_values4}")
                print(f"Unique values in boundary mask 2: {unique_values5}")
                print(f"Unique values in boundary mask 3: {unique_values6}")
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
                f'boundary_f1_{cond_2}_{cond_3}': boundary_f1_23
            })
        else:
            print(f'Cannot find {path1} or {path2} or {path3}')
    return results
    
def plot_comparison_results(comparison_results):
    df = pd.DataFrame(comparison_results)
    df_melted = pd.melt(df, id_vars=['filename'], var_name='metric', value_name='value')
    df_jaccard = df_melted[df_melted['metric'].str.contains('jaccard')]
    df_dice = df_melted[df_melted['metric'].str.contains('dice')]
    df_boundary_f1 = df_melted[df_melted['metric'].str.contains('boundary_f1')]
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
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
    plt.tight_layout()
    plt.show()
