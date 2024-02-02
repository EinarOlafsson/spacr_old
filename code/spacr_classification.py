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
    #subprocess.run([pip_PATH, "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])

    # Install remaining dependencies with conda
    for package in dependencies:
        print(f"Installing {package}")
        subprocess.run([conda_PATH, "install", "-n", env_name, package, "-y"])
        
    pip_packages = ["torchsummary", "opencv-python", "numpy==1.24.0", "numba==0.58.0"]
    
    for package in pip_packages:
    	print(f"Installing {package}")
    	subprocess.run([pip_PATH, "install", package])
    print("Dependencies installation complete.")

env_name = "spacr_classification"

conda_PATH, python_PATH, pip_PATH, env_PATH = get_paths(env_name)

dependencies = ["pandas", "ipykernel", "mahotas","scikit-learn", "scikit-image", "seaborn", "matplotlib", "xgboost", "moviepy", "ipywidgets", "adjustText"]

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

#system dependencies
import os, re, gc, cv2, sys, csv, time, math, string, shutil, random, sqlite3, datetime, torch, torchvision

print('Torch available:', torch.cuda.is_available())
print('CUDA version:',torch.version.cuda)

# image and array processing
import numpy as np
import pandas as pd
from PIL import Image, ImageTk, ImageOps
import tarfile

# statmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# other
import tkinter as tk
from io import BytesIO
from tkinter import filedialog
from itertools import combinations
from collections import OrderedDict, defaultdict
from functools import reduce
from adjustText import adjust_text
from IPython.display import display, clear_output

#paralell processing
import multiprocessing
from itertools import product
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count, Value, Lock

# torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, random_split, Subset, ConcatDataset
import torch.optim as optim
from torch.optim import Adagrad
from torch.optim import AdamW
from torch.autograd import grad, Variable
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, models
import torchvision.transforms as transforms
import torchvision.datasets.utils as dataset_utils

# Visualization dependencies
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Image processing dependencies
import scipy.stats as stats
import scipy.ndimage as ndi
from scipy.stats import zscore, fisher_exact
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_erosion, binary_dilation as binary_erosion, binary_dilation, distance_transform_edt, generate_binary_structure
from scipy.interpolate import UnivariateSpline

# scikit-image
from skimage import exposure, measure, morphology, filters
from skimage.segmentation import find_boundaries, clear_border, watershed
from skimage.morphology import opening, disk, closing, dilation, square
from skimage.exposure import rescale_intensity
from skimage.measure import label, regionprops_table, regionprops, shannon_entropy, find_contours
from skimage.feature import graycomatrix, graycoprops, peak_local_max
from mahotas.features import zernike_moments

# scikit-learn
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, roc_auc_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder

def imshow(img, labels, nrow=20, color='white', fontsize=12):
    n_images = len(labels)
    n_col = nrow
    n_row = int(np.ceil(n_images / n_col))
    img_height = img[0].shape[1]
    img_width = img[0].shape[2]
    canvas = np.zeros((img_height * n_row, img_width * n_col, 3))
    for i in range(n_row):
        for j in range(n_col):
            idx = i * n_col + j
            if idx < n_images:
                canvas[i * img_height:(i + 1) * img_height, j * img_width:(j + 1) * img_width] = np.transpose(img[idx], (1, 2, 0))        
    plt.figure(figsize=(50, 50))
    plt.imshow(canvas)
    plt.axis("off")
    for i, label in enumerate(labels):
        row = i // n_col
        col = i % n_col
        x = col * img_width + 2
        y = row * img_height + 15
        plt.text(x, y, label, color=color, fontsize=fontsize, fontweight='bold')
    plt.show()

#dataloader classes
class Cache:
    def __init__(self, max_size):
        self.cache = OrderedDict()
        self.max_size = max_size
    def get(self, key):
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None
    def put(self, key, value):
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = value

class MyDataset(Dataset):
    def __init__(self, data_dir, loader_classes, transform=None, shuffle=True, load_to_memory=False):
        self.data_dir = data_dir
        self.classes = loader_classes
        self.transform = transform
        self.shuffle = shuffle
        self.load_to_memory = load_to_memory
        self.filenames = []
        self.labels = []
        self.images = []
        self.image_cache = Cache(50)  
        for class_name in self.classes:
            class_path = os.path.join(data_dir, class_name)
            class_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            self.filenames.extend(class_files)
            self.labels.extend([self.classes.index(class_name)] * len(class_files))
        if self.shuffle:
            self.shuffle_dataset()
        if self.load_to_memory:
            self.images = [self.load_image(f) for f in self.filenames]
    def load_image(self, img_path):
        img = self.image_cache.get(img_path)
        if img is None:
            img = Image.open(img_path).convert('RGB')
            self.image_cache.put(img_path, img)
        return img
    def __len__(self):
        return len(self.filenames)
    def shuffle_dataset(self):
        combined = list(zip(self.filenames, self.labels))
        random.shuffle(combined)
        self.filenames, self.labels = zip(*combined)
    def __getitem__(self, index):
        label = self.labels[index]
        filename = self.filenames[index]
        if self.load_to_memory:
            img = self.images[index]
        else:
            img = self.load_image(filename)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = ToTensor()(img)
        return img, label, filename

class CombineLoaders:
    def __init__(self, train_loaders):
        self.train_loaders = train_loaders
        self.loader_iters = [iter(loader) for loader in train_loaders]
    def __iter__(self):
        return self
    def __next__(self):
        while self.loader_iters:
            random.shuffle(self.loader_iters)  # Shuffle the loader_iters list
            for i, loader_iter in enumerate(self.loader_iters):
                try:
                    batch = next(loader_iter)
                    return i, batch
                except StopIteration:
                    self.loader_iters.pop(i)
                    continue
            else:
                break
        raise StopIteration

class CombinedDataset(Dataset):
    def __init__(self, datasets, shuffle=True):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_length = sum(self.lengths)
        self.shuffle = shuffle
        if shuffle:
            self.indices = list(range(self.total_length))
            random.shuffle(self.indices)
        else:
            self.indices = None
    def __getitem__(self, index):
        if self.shuffle:
            index = self.indices[index]
        for dataset, length in zip(self.datasets, self.lengths):
            if index < length:
                return dataset[index]
            index -= length
    def __len__(self):
        return self.total_length
    
class NoClassDataset(Dataset):
    def __init__(self, data_dir, transform=None, shuffle=True, load_to_memory=False):
        self.data_dir = data_dir
        self.transform = transform
        self.shuffle = shuffle
        self.load_to_memory = load_to_memory
        self.filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        if self.shuffle:
            self.shuffle_dataset()
        if self.load_to_memory:
            self.images = [self.load_image(f) for f in self.filenames]
    #@lru_cache(maxsize=None)
    def load_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        return img
    def __len__(self):
        return len(self.filenames)
    def shuffle_dataset(self):
        if self.shuffle:
            random.shuffle(self.filenames)
    def __getitem__(self, index):
        if self.load_to_memory:
            img = self.images[index]
        else:
            img = self.load_image(self.filenames[index])
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = ToTensor()(img)
        # Return both the image and its filename
        return img, self.filenames[index]

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention_probs = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_probs, V)
        return output

class SelfAttention(nn.Module):
    def __init__(self, in_channels, d_k):
        super(SelfAttention, self).__init__()
        self.W_q = nn.Linear(in_channels, d_k)
        self.W_k = nn.Linear(in_channels, d_k)
        self.W_v = nn.Linear(in_channels, d_k)
        self.attention = ScaledDotProductAttention(d_k)
    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        output = self.attention(Q, K, V)
        return output

class NoClassDataset(Dataset):
    def __init__(self, data_dir, transform=None, shuffle=True, load_to_memory=False):
        self.data_dir = data_dir
        self.transform = transform
        self.shuffle = shuffle
        self.load_to_memory = load_to_memory
        self.filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        if self.shuffle:
            self.shuffle_dataset()
        if self.load_to_memory:
            self.images = [self.load_image(f) for f in self.filenames]
    #@lru_cache(maxsize=None)
    def load_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        return img
    def __len__(self):
        return len(self.filenames)
    def shuffle_dataset(self):
        if self.shuffle:
            random.shuffle(self.filenames)
    def __getitem__(self, index):
        if self.load_to_memory:
            img = self.images[index]
        else:
            img = self.load_image(self.filenames[index])
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = ToTensor()(img)
        # Return both the image and its filename
        return img, self.filenames[index]

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention_probs = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_probs, V)
        return output

class SelfAttention(nn.Module):
    def __init__(self, in_channels, d_k):
        super(SelfAttention, self).__init__()
        self.W_q = nn.Linear(in_channels, d_k)
        self.W_k = nn.Linear(in_channels, d_k)
        self.W_v = nn.Linear(in_channels, d_k)
        self.attention = ScaledDotProductAttention(d_k)
    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        output = self.attention(Q, K, V)
        return output

class MyDataset(Dataset):
    def __init__(self, data_dir, loader_classes, transform=None, shuffle=True, pin_memory=False, specific_files=None, specific_labels=None):
        self.data_dir = data_dir
        self.classes = loader_classes
        self.transform = transform
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.filenames = []
        self.labels = []
        
        if specific_files and specific_labels:
            self.filenames = specific_files
            self.labels = specific_labels
        else:
            for class_name in self.classes:
                class_path = os.path.join(data_dir, class_name)
                class_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
                self.filenames.extend(class_files)
                self.labels.extend([self.classes.index(class_name)] * len(class_files))
        
        if self.shuffle:
            self.shuffle_dataset()
            
        if self.pin_memory:
            self.images = [self.load_image(f) for f in self.filenames]
    
    def load_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        return img
    
    def __len__(self):
        return len(self.filenames)
    
    def shuffle_dataset(self):
        combined = list(zip(self.filenames, self.labels))
        random.shuffle(combined)
        self.filenames, self.labels = zip(*combined)
        
    def get_plate(self, filepath):
        filename = os.path.basename(filepath)  # Get just the filename from the full path
        return filename.split('_')[0]
    
    def __getitem__(self, index):
        label = self.labels[index]
        filename = self.filenames[index]
        img = self.load_image(filename)
        if self.transform:
            img = self.transform(img)
        return img, label, filename

def generate_loaders(src, train_mode='erm', mode='train', image_size=228, batch_size=32, classes=['nc','pc'], num_workers=None, validation_split=0.0, max_show=2, pin_memory=False, normalize=False, verbose=False):
    plate_to_filenames = defaultdict(list)
    plate_to_labels = defaultdict(list)
    train_loaders = []
    val_loaders = []
    plate_names = []

    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(image_size, image_size)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(image_size, image_size))])
    
    if mode == 'train':
        data_dir = os.path.join(src, 'train')
        shuffle = True
        print(f'Generating Train and validation datasets')
        
    elif mode == 'test':
        data_dir = os.path.join(src, 'test')
        val_loaders = []
        validation_split=0.0
        shuffle = True
        print(f'Generating test dataset')
    
    else:
        print(f'mode:{mode} is not valid, use mode = train or test')
        return
    
    if train_mode == 'erm':
        data = MyDataset(data_dir, classes, transform=transform, shuffle=shuffle, pin_memory=pin_memory)
        #train_loaders = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers if num_workers is not None else 0, pin_memory=pin_memory)
        if validation_split > 0:
            train_size = int((1 - validation_split) * len(data))
            val_size = len(data) - train_size

            print(f'Train data:{train_size}, Validation data:{val_size}')

            train_dataset, val_dataset = random_split(data, [train_size, val_size])

            train_loaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers if num_workers is not None else 0, pin_memory=pin_memory)
            val_loaders = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers if num_workers is not None else 0, pin_memory=pin_memory)
        else:
            train_loaders = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers if num_workers is not None else 0, pin_memory=pin_memory)
        
    elif train_mode == 'irm':
        data = MyDataset(data_dir, classes, transform=transform, shuffle=shuffle, pin_memory=pin_memory)
        
        for filename, label in zip(data.filenames, data.labels):
            plate = data.get_plate(filename)
            plate_to_filenames[plate].append(filename)
            plate_to_labels[plate].append(label)

        for plate, filenames in plate_to_filenames.items():
            labels = plate_to_labels[plate]
            plate_data = MyDataset(data_dir, classes, specific_files=filenames, specific_labels=labels, transform=transform, shuffle=False, pin_memory=pin_memory)
            plate_names.append(plate)

            if validation_split > 0:
                train_size = int((1 - validation_split) * len(plate_data))
                val_size = len(plate_data) - train_size

                print(f'Train data:{train_size}, Validation data:{val_size}')

                train_dataset, val_dataset = random_split(plate_data, [train_size, val_size])

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers if num_workers is not None else 0, pin_memory=pin_memory)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers if num_workers is not None else 0, pin_memory=pin_memory)

                train_loaders.append(train_loader)
                val_loaders.append(val_loader)
            else:
                train_loader = DataLoader(plate_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers if num_workers is not None else 0, pin_memory=pin_memory)
                train_loaders.append(train_loader)
                val_loaders.append(None)
    
    else:
        print(f'train_mode:{train_mode} is not valid, use: train_mode = irm or erm')
        return

    if verbose:
        if train_mode == 'erm':
            for idx, (images, labels, filenames) in enumerate(train_loaders):
                if idx >= max_show:
                    break
                images = images.cpu()
                label_strings = [str(label.item()) for label in labels]
                imshow(images, label_strings, nrow=20, fontsize=12)

        elif train_mode == 'irm':
            for plate_name, train_loader in zip(plate_names, train_loaders):
                print(f'Plate: {plate_name} with {len(train_loader.dataset)} images')
                for idx, (images, labels, filenames) in enumerate(train_loader):
                    if idx >= max_show:
                        break
                    images = images.cpu()
                    label_strings = [str(label.item()) for label in labels]
                    imshow(images, label_strings, nrow=20, fontsize=12)
    
    return train_loaders, val_loaders, plate_names
    
# Early Fusion Block
class EarlyFusion(nn.Module):
    def __init__(self, in_channels):
        super(EarlyFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = self.conv1(x)
        return x

# Spatial Attention Mechanism
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
# Multi-Scale Block with Attention
class MultiScaleBlockWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlockWithAttention, self).__init__()
        self.dilated_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=1, padding=1)
        self.spatial_attention = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
    def custom_forward(self, x):
        x1 = F.relu(self.dilated_conv1(x), inplace=True)
        x = self.spatial_attention(x1)
        return x

    def forward(self, x):
        return checkpoint(self.custom_forward, x)

# Final Classifier
class CustomCellClassifier(nn.Module):
    def __init__(self, num_classes, pathogen_channel, use_attention, use_checkpoint, dropout_rate):
        super(CustomCellClassifier, self).__init__()
        self.early_fusion = EarlyFusion(in_channels=3)
        
        self.multi_scale_block_1 = MultiScaleBlockWithAttention(in_channels=64, out_channels=64)
        
        self.fc1 = nn.Linear(64, num_classes)
        self.use_checkpoint = use_checkpoint
        # Explicitly require gradients for all parameters
        for param in self.parameters():
            param.requires_grad = True
        
    def custom_forward(self, x):
        x.requires_grad = True 
        x = self.early_fusion(x)
        x = self.multi_scale_block_1(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        return x

    def forward(self, x):
        if self.use_checkpoint:
            x.requires_grad = True 
            return checkpoint(self.custom_forward, x)
        else:
            return self.custom_forward(x)

#CNN and Transformer class, pick any Torch model.
class TorchModel(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, dropout_rate=None, use_checkpoint=False):
        super(TorchModel, self).__init__()
        self.model_name = model_name
        self.use_checkpoint = use_checkpoint
        self.base_model = self.init_base_model(pretrained)
        
        # Retain layers up to and including the (5): Linear layer for model 'maxvit_t'
        if model_name == 'maxvit_t':
            self.base_model.classifier = nn.Sequential(*list(self.base_model.classifier.children())[:-1])
        
        if dropout_rate is not None:
            self.apply_dropout_rate(self.base_model, dropout_rate)
            
        self.num_ftrs = self.get_num_ftrs()
        self.init_spacr_classifier(dropout_rate)

    def apply_dropout_rate(self, model, dropout_rate):
        """Apply dropout rate to all dropout layers in the model."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate

    def init_base_model(self, pretrained):
        """Initialize the base model from torchvision.models."""
        model_func = models.__dict__.get(self.model_name, None)
        if not model_func:
            raise ValueError(f"Model {self.model_name} is not recognized.")
        weight_choice = self.get_weight_choice()
        if weight_choice is not None:
            return model_func(weights=weight_choice)
        else:
            return model_func(pretrained=pretrained)

    def get_weight_choice(self):
        """Get weight choice if it exists for the model."""
        weight_enum = None
        for attr_name in dir(models):
            if attr_name.lower() == f"{self.model_name}_weights".lower():
                weight_enum = getattr(models, attr_name)
                break
        return weight_enum.DEFAULT if weight_enum else None

    def get_num_ftrs(self):
        """Determine the number of features output by the base model."""
        if hasattr(self.base_model, 'fc'):
            self.base_model.fc = nn.Identity()
        elif hasattr(self.base_model, 'classifier'):
            if self.model_name != 'maxvit_t':
                self.base_model.classifier = nn.Identity()

        # Forward a dummy input and check output size
        dummy_input = torch.randn(1, 3, 224, 224)
        output = self.base_model(dummy_input)
        return output.size(1)

    def init_spacr_classifier(self, dropout_rate):
        """Initialize the SPACR classifier."""
        self.use_dropout = dropout_rate is not None
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
        self.spacr_classifier = nn.Linear(self.num_ftrs, 1)

    def forward(self, x):
        """Define the forward pass of the model."""
        if self.use_checkpoint:
            x = checkpoint(self.base_model, x)
        else:
            x = self.base_model(x)
        if self.use_dropout:
            x = self.dropout(x)
        logits = self.spacr_classifier(x).flatten()
        return logits

class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLossWithLogits, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()
    
class ResNet(nn.Module):
    def __init__(self, resnet_type='resnet50', dropout_rate=None, use_checkpoint=False, init_weights='imagenet'):
        super(ResNet, self).__init__()

        resnet_map = {
            'resnet18': {'func': models.resnet18, 'weights': ResNet18_Weights.IMAGENET1K_V1},
            'resnet34': {'func': models.resnet34, 'weights': ResNet34_Weights.IMAGENET1K_V1},
            'resnet50': {'func': models.resnet50, 'weights': ResNet50_Weights.IMAGENET1K_V1},
            'resnet101': {'func': models.resnet101, 'weights': ResNet101_Weights.IMAGENET1K_V1},
            'resnet152': {'func': models.resnet152, 'weights': ResNet152_Weights.IMAGENET1K_V1}
        }

        if resnet_type not in resnet_map:
            raise ValueError(f"Invalid resnet_type. Choose from {list(resnet_map.keys())}")

        self.initialize_base(resnet_map[resnet_type], dropout_rate, use_checkpoint, init_weights)

    def initialize_base(self, base_model_dict, dropout_rate, use_checkpoint, init_weights):
        if init_weights == 'imagenet':
            self.resnet = base_model_dict['func'](weights=base_model_dict['weights'])
        elif init_weights == 'none':
            self.resnet = base_model_dict['func'](weights=None)
        else:
            raise ValueError("init_weights should be either 'imagenet' or 'none'")

        self.fc1 = nn.Linear(1000, 500)
        self.use_dropout = dropout_rate != None
        self.use_checkpoint = use_checkpoint

        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(500, 1)

    def forward(self, x):
        x.requires_grad = True  # Ensure that the tensor has requires_grad set to True

        if self.use_checkpoint:
            x = checkpoint(self.resnet, x)  # Use checkpointing for just the ResNet part
        else:
            x = self.resnet(x)
        
        x = F.relu(self.fc1(x))

        if self.use_dropout:
            x = self.dropout(x)

        logits = self.fc2(x).flatten()
        return logits

def split_my_dataset(dataset, split_ratio=0.1):
    num_samples = len(dataset)
    indices = list(range(num_samples))
    split_idx = int((1 - split_ratio) * num_samples)
    random.shuffle(indices)
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset

def classification_metrics(all_labels, prediction_pos_probs, loader_name, loss, epoch):
    
    if len(all_labels) != len(prediction_pos_probs):
        raise ValueError(f"all_labels ({len(all_labels)}) and pred_labels ({len(prediction_pos_probs)}) have different lengths")
    
    unique_labels = np.unique(all_labels)
    if len(unique_labels) >= 2:
        pr_labels = np.array(all_labels).astype(int)
        precision, recall, thresholds = precision_recall_curve(pr_labels, prediction_pos_probs, pos_label=1)
        pr_auc = auc(recall, precision)
        thresholds = np.append(thresholds, 0.0)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        optimal_idx = np.nanargmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        pred_labels = [int(p > 0.5) for p in prediction_pos_probs]
    if len(unique_labels) < 2:
        optimal_threshold = 0.5
        pred_labels = [int(p > optimal_threshold) for p in prediction_pos_probs]
        pr_auc = np.nan
    data = {'label': all_labels, 'pred': pred_labels}
    df = pd.DataFrame(data)
    pc_df = df[df['label'] == 1.0]
    nc_df = df[df['label'] == 0.0]
    correct = df[df['label'] == df['pred']]
    acc_all = len(correct) / len(df)
    if len(pc_df) > 0:
        correct_pc = pc_df[pc_df['label'] == pc_df['pred']]
        acc_pc = len(correct_pc) / len(pc_df)
    else:
        acc_pc = np.nan
    if len(nc_df) > 0:
        correct_nc = nc_df[nc_df['label'] == nc_df['pred']]
        acc_nc = len(correct_nc) / len(nc_df)
    else:
        acc_nc = np.nan
    data_dict = {'accuracy': acc_all, 'neg_accuracy': acc_nc, 'pos_accuracy': acc_pc, 'loss':loss.item(),'prauc':pr_auc, 'optimal_threshold':optimal_threshold}
    data_df = pd.DataFrame(data_dict, index=[str(epoch)+'_'+loader_name]) 
    return data_df
    
def evaluate_model_core(model, loader, loader_name, epoch, loss_type):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    loss = 0
    correct = 0
    total_samples = 0
    prediction_pos_probs = []
    all_labels = []
    model = model.to(device)
    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(loader, start=1):
            start_time = time.time()
            data, target = data.to(device), target.to(device).float()
            #data, target = data.to(torch.float).to(device), target.to(device).float()
            output = model(data)
            loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()
            loss = calculate_loss(output, target, loss_type=loss_type)
            loss += loss.item()
            total_samples += data.size(0)
            pred = torch.where(output >= 0.5,
                               torch.Tensor([1.0]).to(device).float(),
                               torch.Tensor([0.0]).to(device).float())
            correct += pred.eq(target.view_as(pred)).sum().item()
            batch_prediction_pos_prob = torch.sigmoid(output).cpu().numpy()
            prediction_pos_probs.extend(batch_prediction_pos_prob.tolist())
            all_labels.extend(target.cpu().numpy().tolist())
            mean_loss = loss / total_samples
            acc = correct / total_samples
            end_time = time.time()
            test_time = end_time - start_time
            print(f'\rTest: epoch: {epoch} Accuracy: {acc:.5f} batch: {batch_idx+1}/{len(loader)} loss: {mean_loss:.5f} loss: {mean_loss:.5f} time {test_time:.5f}', end='\r', flush=True)
    loss /= len(loader)
    data_df = classification_metrics(all_labels, prediction_pos_probs, loader_name, loss, epoch)
    return data_df, prediction_pos_probs, all_labels

def evaluate_model_performance(loaders, model, loader_name_list, epoch, train_mode, loss_type):
    start_time = time.time()
    df_list = []
    if train_mode == 'erm':
        result, _, _ = evaluate_model_core(model, loaders, loader_name_list, epoch, loss_type)
    if train_mode == 'irm':
        for loader_index in range(0, len(loaders)):
            loader = loaders[loader_index]
            loader_name = loader_name_list[loader_index]
            data_df, _, _ = evaluate_model_core(model, loader, loader_name, epoch, loss_type)
            torch.cuda.empty_cache()
            df_list.append(data_df)
        result = pd.concat(df_list)
        nc_mean = result['neg_accuracy'].mean(skipna=True)
        pc_mean = result['pos_accuracy'].mean(skipna=True)
        tot_mean = result['accuracy'].mean(skipna=True)
        loss_mean = result['loss'].mean(skipna=True)
        prauc_mean = result['prauc'].mean(skipna=True)
        data_mean = {'accuracy': tot_mean, 'neg_accuracy': nc_mean, 'pos_accuracy': pc_mean, 'loss': loss_mean, 'prauc': prauc_mean}
        result = pd.concat([pd.DataFrame(result), pd.DataFrame(data_mean, index=[str(epoch)+'_mean'])])
    end_time = time.time()
    test_time = end_time - start_time
    return result, test_time

def test_model_core(model, loader, loader_name, epoch, loss_type):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    loss = 0
    correct = 0
    total_samples = 0
    prediction_pos_probs = []
    all_labels = []
    filenames = []
    true_targets = []
    predicted_outputs = []

    model = model.to(device)
    with torch.no_grad():
        for batch_idx, (data, target, filename) in enumerate(loader, start=1):  # Assuming loader provides filenames
            start_time = time.time()
            data, target = data.to(device), target.to(device).float()
            output = model(data)
            loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()
            loss = calculate_loss(output, target, loss_type=loss_type)
            loss += loss.item()
            total_samples += data.size(0)
            pred = torch.where(output >= 0.5,
                               torch.Tensor([1.0]).to(device).float(),
                               torch.Tensor([0.0]).to(device).float())
            correct += pred.eq(target.view_as(pred)).sum().item()
            batch_prediction_pos_prob = torch.sigmoid(output).cpu().numpy()
            prediction_pos_probs.extend(batch_prediction_pos_prob.tolist())
            all_labels.extend(target.cpu().numpy().tolist())
            
            # Storing intermediate results in lists
            true_targets.extend(target.cpu().numpy().tolist())
            predicted_outputs.extend(pred.cpu().numpy().tolist())
            filenames.extend(filename)
            
            mean_loss = loss / total_samples
            acc = correct / total_samples
            end_time = time.time()
            test_time = end_time - start_time
            print(f'\rTest: epoch: {epoch} Accuracy: {acc:.5f} batch: {batch_idx}/{len(loader)} loss: {mean_loss:.5f} time {test_time:.5f}', end='\r', flush=True)
    
    # Constructing the DataFrame
    results_df = pd.DataFrame({
        'filename': filenames,
        'true_label': true_targets,
        'predicted_label': predicted_outputs,
        'class_1_probability':prediction_pos_probs})

    loss /= len(loader)
    data_df = classification_metrics(all_labels, prediction_pos_probs, loader_name, loss, epoch)
    return data_df, prediction_pos_probs, all_labels, results_df

def test_model_performance(loaders, model, loader_name_list, epoch, train_mode, loss_type):
    start_time = time.time()
    df_list = []
    if train_mode == 'erm':
        result, prediction_pos_probs, all_labels, results_df = test_model_core(model, loaders, loader_name_list, epoch, loss_type)
    if train_mode == 'irm':
        for loader_index in range(0, len(loaders)):
            loader = loaders[loader_index]
            loader_name = loader_name_list[loader_index]
            data_df, prediction_pos_probs, all_labels, results_df = test_model_core(model, loader, loader_name, epoch, loss_type)
            torch.cuda.empty_cache()
            df_list.append(data_df)
        result = pd.concat(df_list)
        nc_mean = result['neg_accuracy'].mean(skipna=True)
        pc_mean = result['pos_accuracy'].mean(skipna=True)
        tot_mean = result['accuracy'].mean(skipna=True)
        loss_mean = result['loss'].mean(skipna=True)
        prauc_mean = result['prauc'].mean(skipna=True)
        data_mean = {'accuracy': tot_mean, 'neg_accuracy': nc_mean, 'pos_accuracy': pc_mean, 'loss': loss_mean, 'prauc': prauc_mean}
        result = pd.concat([pd.DataFrame(result), pd.DataFrame(data_mean, index=[str(epoch)+'_mean'])])
    end_time = time.time()
    test_time = end_time - start_time
    return result, results_df


def save_model(model, model_type, results_df, dst, epoch, epochs, intermedeate_save=[0.99,0.98,0.95,0.94]):
    
    if epoch % 100 == 0:
        #torch.save(model, dst+'/'+str(epoch)+f'epoch{epoch}_model.pth')
        torch.save(model, f'{dst}/{model_type}_epoch_{str(epoch)}.pth')
        
    if epoch == epochs:
        #torch.save(model, dst+'/'+str(epoch)+f'epoch{epoch}_model.pth')
        torch.save(model, f'{dst}/{model_type}_epoch_{str(epoch)}.pth')
    
    if results_df['neg_accuracy'].dropna().mean() >= intermedeate_save[0] and results_df['pos_accuracy'].dropna().mean() >= intermedeate_save[0]:
        percentile = str(intermedeate_save[0]*100)
        print(f'\rfound: {percentile}% accurate model', end='\r', flush=True)
        #torch.save(model.state_dict(), dst+'/'+str(epoch)+'_model'+str(percentile)+'.pth')
        torch.save(model, f'{dst}/{model_type}_epoch_{str(epoch)}_acc_{str(percentile)}.pth')
        #torch.save(model.state_dict(), 'path/to/save/model.pth')

    elif results_df['neg_accuracy'].dropna().mean() >= intermedeate_save[1] and results_df['pos_accuracy'].dropna().mean() >= intermedeate_save[1]:
        percentile = str(intermedeate_save[1]*100)
        print(f'\rfound: {percentile}% accurate model', end='\r', flush=True)
        torch.save(model, f'{dst}/{model_type}_epoch_{str(epoch)}_acc_{str(percentile)}.pth')
    
    elif results_df['neg_accuracy'].dropna().mean() >= intermedeate_save[2] and results_df['pos_accuracy'].dropna().mean() >= intermedeate_save[2]:
        percentile = str(intermedeate_save[2]*100)
        print(f'\rfound: {percentile}% accurate model', end='\r', flush=True)
        torch.save(model, f'{dst}/{model_type}_epoch_{str(epoch)}_acc_{str(percentile)}.pth')
    
    elif results_df['neg_accuracy'].dropna().mean() >= intermedeate_save[3] and results_df['pos_accuracy'].dropna().mean() >= intermedeate_save[3]:
        percentile = str(intermedeate_save[3]*100)
        print(f'\rfound: {percentile}% accurate model', end='\r', flush=True)
        torch.save(model, f'{dst}/{model_type}_epoch_{str(epoch)}_acc_{str(percentile)}.pth')

def save_progress(dst, results_df, train_metrics_df):
    #Save accuracy, loss, PRAUC
    os.makedirs(dst, exist_ok=True)
    results_path = os.path.join(dst, 'acc_loss_prauc.csv')
    if not os.path.exists(results_path):
        results_df.to_csv(results_path, index=True, header=True, mode='w')
    else:
        results_df.to_csv(results_path, index=True, header=False, mode='a')
    training_metrics_path = os.path.join(dst, 'training_metrics.csv')
    if not os.path.exists(training_metrics_path):
        train_metrics_df.to_csv(training_metrics_path, index=True, header=True, mode='w')
    else:
        train_metrics_df.to_csv(training_metrics_path, index=True, header=False, mode='a')
    return

def save_settings(settings, src):
    dst = os.path.join(src,'model')
    settings_loc =  os.path.join(dst,'settings.csv')
    os.makedirs(dst, exist_ok=True)
    settings_df = pd.DataFrame(list(settings.items()), columns=['setting_key', 'setting_value'])
    display(settings_df)
    settings_df.to_csv(settings_loc, index=False)
    return

def compute_irm_penalty(losses, dummy_w, device):
    weighted_losses = [loss.clone().detach().requires_grad_(True).to(device) * dummy_w for loss in losses]
    gradients = [grad(w_loss, dummy_w, create_graph=True)[0] for w_loss in weighted_losses]
    irm_penalty = 0.0
    for g1, g2 in combinations(gradients, 2):
        irm_penalty += (g1.dot(g2))**2
    return irm_penalty

def print_model_summary(base_model, channels, height, width):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_model.to(device)
    summary(base_model, (channels, height,width))
    return

def choose_model(model_type, device, init_weights=True, dropout_rate=0, use_checkpoint=False, channels=3, height=224, width=224, chan_dict=None):
    
    torch_model_types = torchvision.models.list_models(module=torchvision.models)
    model_types = torch_model_types + ['custom']
    
    if not chan_dict is None:
        pathogen_channel = chan_dict['pathogen_channel']
        nucleus_channel = chan_dict['nucleus_channel']
        protein_channel = chan_dict['protein_channel']
    
    if model_type not in model_types:
        print(f'Invalid model_type: {model_type}. Compatible model_types: {model_types}')
        return

    print(f'\rModel parameters: Architecture: {model_type} init_weights: {init_weights} dropout_rate: {dropout_rate} use_checkpoint: {use_checkpoint}', end='\r', flush=True)
    
    if model_type == 'custom':
        
        base_model = CustomCellClassifier(num_classes, pathogen_channel=pathogen_channel, use_attention=True, use_checkpoint=use_checkpoint, dropout_rate=dropout_rate)
        #base_model = CustomCellClassifier(num_classes=2, pathogen_channel=pathogen_channel, nucleus_channel=nucleus_channel, protein_channel=protein_channel, dropout_rate=dropout_rate, use_checkpoint=use_checkpoint)
    elif model_type in torch_model_types:
        base_model = TorchModel(model_name=model_type, pretrained=init_weights, dropout_rate=dropout_rate)
    else:
        print(f'Compatible model_types: {model_types}')
        raise ValueError(f"Invalid model_type: {model_type}")

    print(base_model)
    
    return base_model

def calculate_loss(output, target, loss_type='binary_cross_entropy_with_logits'):
    if loss_type == 'binary_cross_entropy_with_logits':
        loss = F.binary_cross_entropy_with_logits(output, target)
    elif loss_type == 'focal_loss':
        focal_loss_fn = FocalLossWithLogits(alpha=1, gamma=2)
        loss = focal_loss_fn(output, target)
    return loss

def train_model(dst, model_type, train_loaders, train_loader_names, train_mode='erm', epochs=100, learning_rate=0.0001, weight_decay=0.05, amsgrad=False, optimizer_type='adamw', use_checkpoint=False, dropout_rate=0, num_workers=20, val_loaders=None, test_loaders=None, init_weights='imagenet', intermedeate_save=None, chan_dict=None, schedule = None, loss_type='binary_cross_entropy_with_logits', gradient_accumulation=False, gradient_accumulation_steps=4):
    
    print(f'Train batches:{len(train_loaders)}, Validation batches:{len(val_loaders)}')
    
    if test_loaders != None:
        print(f'Test batches:{len(test_loaders)}')
        
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    
    for idx, (images, labels, filenames) in enumerate(train_loaders):
        batch, channels, height, width = images.shape
        break

    model = choose_model(model_type, device, init_weights, dropout_rate, use_checkpoint)
    model.to(device)
    
    if optimizer_type == 'adamw':
        optimizer = AdamW(model.parameters(), lr=learning_rate,  betas=(0.9, 0.999), weight_decay=weight_decay, amsgrad=amsgrad)
    
    if optimizer_type == 'adagrad':
        optimizer = Adagrad(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=weight_decay)
    
    if schedule == 'step_lr':
        StepLR_step_size = int(epochs/5)
        StepLR_gamma = 0.75
        scheduler = StepLR(optimizer, step_size=StepLR_step_size, gamma=StepLR_gamma)
    elif schedule == 'reduce_lr_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    else:
        scheduler = None

    if train_mode == 'erm':
        for epoch in range(1, epochs+1):
            model.train()
            start_time = time.time()
            running_loss = 0.0

            # Initialize gradients if using gradient accumulation
            if gradient_accumulation:
                optimizer.zero_grad()

            for batch_idx, (data, target, filenames) in enumerate(train_loaders, start=1):
                data, target = data.to(device), target.to(device).float()
                output = model(data)
                loss = calculate_loss(output, target, loss_type=loss_type)
                # Normalize loss if using gradient accumulation
                if gradient_accumulation:
                    loss /= gradient_accumulation_steps
                running_loss += loss.item() * gradient_accumulation_steps  # correct the running_loss
                loss.backward()

                # Step optimizer if not using gradient accumulation or every gradient_accumulation_steps
                if not gradient_accumulation or (batch_idx % gradient_accumulation_steps == 0):
                    optimizer.step()
                    optimizer.zero_grad()

                avg_loss = running_loss / batch_idx
                print(f'\rTrain: epoch: {epoch} batch: {batch_idx}/{len(train_loaders)} avg_loss: {avg_loss:.5f} time: {(time.time()-start_time):.5f}', end='\r', flush=True)

            end_time = time.time()
            train_time = end_time - start_time
            train_metrics = {'epoch':epoch,'loss':loss.cpu().item(), 'train_time':train_time}
            train_metrics_df = pd.DataFrame(train_metrics, index=[epoch])
            train_names = 'train'
            results_df, train_test_time = evaluate_model_performance(train_loaders, model, train_names, epoch, train_mode='erm', loss_type=loss_type)
            train_metrics_df['train_test_time'] = train_test_time
            if val_loaders != None:
                val_names = 'val'
                result, val_time = evaluate_model_performance(val_loaders, model, val_names, epoch, train_mode='erm', loss_type=loss_type)
                
                if schedule == 'reduce_lr_on_plateau':
                    val_loss = result['loss']
                
                results_df = pd.concat([results_df, result])
                train_metrics_df['val_time'] = val_time
            if test_loaders != None:
                test_names = 'test'
                result, test_test_time = evaluate_model_performance(test_loaders, model, test_names, epoch, train_mode='erm', loss_type=loss_type)
                results_df = pd.concat([results_df, result])
                test_time = (train_test_time+val_time+test_test_time)/3
                train_metrics_df['test_time'] = test_time
            
            if scheduler:
                if schedule == 'reduce_lr_on_plateau':
                    scheduler.step(val_loss)
                if schedule == 'step_lr':
                    scheduler.step()
            
            save_progress(dst, results_df, train_metrics_df)
            clear_output(wait=True)
            display(results_df)
            save_model(model, model_type, results_df, dst, epoch, epochs, intermedeate_save=[0.99,0.98,0.95,0.94])
            
    if train_mode == 'irm':
        dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to(device)
        phi = torch.nn.Parameter (torch.ones(4,1))
        for epoch in range(1, epochs):
            model.train()
            penalty_factor = epoch * 1e-5
            epoch_names = [str(epoch) + '_' + item for item in train_loader_names]
            loader_erm_loss_list = []
            total_erm_loss_mean = 0
            for loader_index in range(0, len(train_loaders)):
                start_time = time.time()
                loader = train_loaders[loader_index]
                loader_erm_loss_mean = 0
                batch_count = 0
                batch_erm_loss_list = []
                for batch_idx, (data, target, filenames) in enumerate(loader, start=1):
                    optimizer.zero_grad()
                    data, target = data.to(device), target.to(device).float()
                    
                    output = model(data)
                    erm_loss = F.binary_cross_entropy_with_logits(output * dummy_w, target, reduction='none')
                    
                    batch_erm_loss_list.append(erm_loss.mean())
                    print(f'\repoch: {epoch} loader: {loader_index} batch: {batch_idx+1}/{len(loader)}', end='\r', flush=True)
                loader_erm_loss_mean = torch.stack(batch_erm_loss_list).mean()
                loader_erm_loss_list.append(loader_erm_loss_mean)
            total_erm_loss_mean = torch.stack(loader_erm_loss_list).mean()
            irm_loss = compute_irm_penalty(loader_erm_loss_list, dummy_w, device)
            
            (total_erm_loss_mean + penalty_factor * irm_loss).backward()
            optimizer.step()
            
            end_time = time.time()
            train_time = end_time - start_time
            
            train_metrics = {'epoch': epoch, 'irm_loss': irm_loss, 'erm_loss': total_erm_loss_mean, 'penalty_factor': penalty_factor, 'train_time': train_time}
            #train_metrics = {'epoch':epoch,'irm_loss':irm_loss.cpu().item(),'erm_loss':total_erm_loss_mean.cpu().item(),'penalty_factor':penalty_factor, 'train_time':train_time}
            train_metrics_df = pd.DataFrame(train_metrics, index=[epoch])
            print(f'\rTrain: epoch: {epoch} loader: {loader_index} batch: {batch_idx+1}/{len(loader)} irm_loss: {irm_loss:.5f} mean_erm_loss: {total_erm_loss_mean:.5f} train time {train_time:.5f}', end='\r', flush=True)            
            
            train_names = [item + '_train' for item in train_loader_names]
            results_df, train_test_time = evaluate_model_performance(train_loaders, model, train_names, epoch, train_mode='irm', loss_type=loss_type)
            train_metrics_df['train_test_time'] = train_test_time
            
            if val_loaders != None:
                val_names = [item + '_val' for item in train_loader_names]
                result, val_time = evaluate_model_performance(val_loaders, model, val_names, epoch, train_mode='irm', loss_type=loss_type)
                
                if schedule == 'reduce_lr_on_plateau':
                    val_loss = result['loss']
                
                results_df = pd.concat([results_df, result])
                train_metrics_df['val_time'] = val_time
            
            if test_loaders != None:
                test_names = [item + '_test' for item in test_loader_names]
                result, test_test_time = evaluate_model_performance(test_loaders, model, test_names, epoch, train_mode='irm', loss_type=loss_type)
                results_df = pd.concat([results_df, result])
                train_metrics_df['test_test_time'] = test_test_time
                
            if scheduler:
                if schedule == 'reduce_lr_on_plateau':
                    scheduler.step(val_loss)
                if schedule == 'step_lr':
                    scheduler.step()
            
            clear_output(wait=True)
            display(results_df)
            save_progress(dst, results_df, train_metrics_df)
            save_model(model, model_type, results_df, dst, epoch, epochs, intermedeate_save=[0.99,0.98,0.95,0.94])
            print(f'Saved model: {dst}')
    return

def pick_best_model(src):
    all_files = os.listdir(src)
    pth_files = [f for f in all_files if f.endswith('.pth')]
    pattern = re.compile(r'_epoch_(\d+)_acc_(\d+(?:\.\d+)?)')

    def sort_key(x):
        match = pattern.search(x)
        if not match:
            return (0.0, 0)  # Make the primary sorting key float for consistency
        g1, g2 = match.groups()
        return (float(g2), int(g1))  # Primary sort by accuracy (g2) and secondary sort by epoch (g1)
    
    sorted_files = sorted(pth_files, key=sort_key, reverse=True)
    best_model = sorted_files[0]
    return os.path.join(src, best_model)

def copy_missclassified(df):
    misclassified = df[df['true_label'] != df['predicted_label']]
    for _, row in misclassified.iterrows():
        original_path = row['filename']
        filename = os.path.basename(original_path)
        dest_folder = os.path.dirname(os.path.dirname(original_path))
        if "pc" in original_path:
            new_path = os.path.join(dest_folder, "missclassified/pc", filename)
        else:
            new_path = os.path.join(dest_folder, "missclassified/nc", filename)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.copy(original_path, new_path)
    print(f"Copied {len(misclassified)} misclassified images.")
    return

def train_test_model(src, settings, custom_model=False, custom_model_path=None):
    if custom_model:
        model = torch.load(custom_model_path) #if using a custom trained model
    
    if settings['train']:
        save_settings(settings, src)
    torch.cuda.empty_cache()
    torch.cuda.memory.empty_cache()
    gc.collect()
    dst = os.path.join(src,'model')
    os.makedirs(dst, exist_ok=True)
    settings['src'] = src
    settings['dst'] = dst
    if settings['train']:
        train, val, plate_names  = generate_loaders(src, 
                                                    train_mode=settings['train_mode'], 
                                                    mode='train', 
                                                    image_size=settings['image_size'],
                                                    batch_size=settings['batch_size'], 
                                                    classes=settings['classes'], 
                                                    num_workers=settings['num_workers'],
                                                    validation_split=settings['val_split'],
                                                    pin_memory=settings['pin_memory'],
                                                    normalize=settings['normalize'],
                                                    verbose=settings['verbose']) 

    if settings['test']:
        test, _, plate_names_test = generate_loaders(src, 
                                   train_mode=settings['train_mode'], 
                                   mode='test', 
                                   image_size=settings['image_size'],
                                   batch_size=settings['batch_size'], 
                                   classes=settings['classes'], 
                                   num_workers=settings['num_workers'],
                                   validation_split=0.0,
                                   pin_memory=settings['pin_memory'],
                                   normalize=settings['normalize'],
                                   verbose=settings['verbose'])
        if model == None:
            model_path = pick_best_model(src+'/model')
            print(f'Best model: {model_path}')

            model = torch.load(model_path, map_location=lambda storage, loc: storage)

            model_type = settings['model_type']
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(type(model))
            print(model)
        
        model_fldr = os.path.join(src,'model')
        time_now = datetime.date.today().strftime('%y%m%d')
        result_loc = f'{model_fldr}/{model_type}_time_{time_now}_result.csv'
        acc_loc = f'{model_fldr}/{model_type}_time_{time_now}_acc.csv'
        print(f'Results wil be saved in: {result_loc}')
        
        result, accuracy = test_model_performance(loaders=test,
                                                  model=model,
                                                  loader_name_list='test',
                                                  epoch=1,
                                                  train_mode=settings['train_mode'],
                                                  loss_type=settings['loss_type'])
        
        result.to_csv(result_loc, index=True, header=True, mode='w')
        accuracy.to_csv(acc_loc, index=True, header=True, mode='w')
        copy_missclassified(accuracy)
    else:
        test = None
    
    if settings['train']:
        train_model(dst = settings['dst'],
                    model_type=settings['model_type'],
                    train_loaders = train, 
                    train_loader_names = plate_names, 
                    train_mode = settings['train_mode'], 
                    epochs = settings['epochs'], 
                    learning_rate = settings['learning_rate'],
                    init_weights = settings['init_weights'],
                    weight_decay = settings['weight_decay'], 
                    amsgrad = settings['amsgrad'], 
                    optimizer_type = settings['optimizer_type'], 
                    use_checkpoint = settings['use_checkpoint'], 
                    dropout_rate = settings['dropout_rate'], 
                    num_workers = settings['num_workers'], 
                    val_loaders = val, 
                    test_loaders = test, 
                    intermedeate_save = settings['intermedeate_save'],
                    schedule = settings['schedule'],
                    loss_type=settings['loss_type'], 
                    gradient_accumulation=settings['gradient_accumulation'], 
                    gradient_accumulation_steps=settings['gradient_accumulation_steps'])

    torch.cuda.empty_cache()
    torch.cuda.memory.empty_cache()
    gc.collect()

def read_db(db_loc, tables):
    conn = sqlite3.connect(db_loc) # Create a connection to the database
    dfs = []
    for table in tables:
        query = f'SELECT * FROM {table}' # Write a SQL query to get the data from the database
        df = pd.read_sql_query(query, conn) # Use the read_sql_query function to get the data and save it as a DataFrame
        dfs.append(df)
    conn.close() # Close the connection
    return dfs

# Function to apply to each row
def map_values(row, dict_, type_='col'):
    for values, cols in dict_.items():
        if row[type_] in cols:
            return values
    return None

def split_data(df, group_by, object_type):
    
    df['prcfo'] = df['prcf'] + '_' + df[object_type]
    
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
            if verbose:
                print(f'plate: {i+1} cells:{len(cell)}')
	# see parasites logic, copy logic to other tables #here
        if 'nucleus' in tables:
            nucleus = dfs[1]
            if verbose:
                print(f'plate: {i+1} nuclei:{len(nucleus)} ')

        if 'parasite' in tables:
            if len(tables) == 1:
                parasite = dfs[0]
            else:
                parasite = dfs[2]
            if verbose:
                print(f'plate: {i+1} parasites:{len(parasite)}')
        
        if 'cytoplasm' in tables:
            if not 'parasite' in tables:
                cytoplasm = dfs[2]
            else:
                cytoplasm = dfs[3]
            if verbose:
                print(f'plate: {i+1} cytoplasms: {len(cytoplasm)}')

        if i > 0:
            if 'cell' in tables:
                cells = pd.concat([cells, cell], axis = 0)
            if 'nucleus' in tables:
                nuclei = pd.concat([nuclei, nucleus], axis = 0)
            if 'parasite' in tables:
                parasites = pd.concat([parasites, parasite], axis = 0)
            if 'cytoplasm' in tables:
                cytoplasms = pd.concat([cytoplasms, cytoplasm], axis = 0)
        else:
            if 'cell' in tables:
                cells = cell.copy()
            if 'nucleus' in tables:
                nuclei = nucleus.copy()
            if 'parasite' in tables:
                parasites = parasite.copy()
            if 'cytoplasm' in tables:
                cytoplasms = cytoplasm.copy()
    
    #Add an o in front of all object and cell lables to convert them to strings
    if 'cell' in tables:
        cells = cells.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        cells = cells.assign(prcfo = lambda x: x['prcf'] + '_' + x['object_label'])
        cells_g_df, metadata = split_data(cells, 'prcfo', 'object_label')
        merged_df = cells_g_df.copy()
        if verbose:
            print(f'cells: {len(cells)}')
            print(f'cells grouped: {len(cells_g_df)}')
		
    if 'cytoplasm' in tables:
        cytoplasms = cytoplasms.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        cytoplasms = cytoplasms.assign(prcfo = lambda x: x['prcf'] + '_' + x['object_label'])
        cytoplasms_g_df, _ = split_data(cytoplasms, 'prcfo', 'object_label')
        merged_df = cells_g_df.merge(cytoplasms_g_df, left_index=True, right_index=True)
        if verbose:
            print(f'cytoplasms: {len(cytoplasms)}')
            print(f'cytoplasms grouped: {len(cytoplasms_g_df)}')
		
    if 'nucleus' in tables:
        if not 'cell' in tables:
            cells_g_df = pd.DataFrame()
        nuclei = nuclei.dropna(subset=['cell_id'])
        nuclei = nuclei.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        nuclei = nuclei.assign(cell_id=lambda x: 'o' + x['cell_id'].astype(int).astype(str))
        nuclei = nuclei.assign(prcfo = lambda x: x['prcf'] + '_' + x['cell_id'])
        nuclei['nuclei_prcfo_count'] = nuclei.groupby('prcfo')['prcfo'].transform('count')
        if include_multinucleated == False:
            nuclei = nuclei[nuclei['nuclei_prcfo_count']==1]
        nuclei_g_df, _ = split_data(nuclei, 'prcfo', 'cell_id')
        if verbose:
            print(f'nuclei: {len(nuclei)}')
            print(f'nuclei grouped: {len(nuclei_g_df)}')
        if 'cytoplasm' in tables:
            merged_df = merged_df.merge(nuclei_g_df, left_index=True, right_index=True)
        else:
            merged_df = cells_g_df.merge(nuclei_g_df, left_index=True, right_index=True)
		
    if 'parasite' in tables:
        if not 'cell' in tables:
            cells_g_df = pd.DataFrame()
        parasites = parasites.dropna(subset=['cell_id'])
        parasites = parasites.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        parasites = parasites.assign(cell_id=lambda x: 'o' + x['cell_id'].astype(int).astype(str))
        parasites = parasites.assign(prcfo = lambda x: x['prcf'] + '_' + x['cell_id'])
        parasites['parasite_prcfo_count'] = parasites.groupby('prcfo')['prcfo'].transform('count')
        if include_noninfected == False:
            parasites = parasites[parasites['parasite_prcfo_count']>=1]
        if isinstance(include_multiinfected, bool):
            if include_multiinfected == False:
                parasites = parasites[parasites['parasite_prcfo_count']<=1]
        if isinstance(include_multiinfected, float):
            parasites = parasites[parasites['parasite_prcfo_count']<=include_multiinfected]
        parasites_g_df, _ = split_data(parasites, 'prcfo', 'cell_id')
        if verbose:
            print(f'parasites: {len(parasites)}')
            print(f'parasites grouped: {len(parasites_g_df)}')
        merged_df = merged_df.merge(parasites_g_df, left_index=True, right_index=True)
    
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
    if verbose:
        print(f'Generated dataframe with: {len(merged_df.columns)} columns and {len(merged_df)} rows')
    
    obj_df_ls = []
    if 'cell' in tables:
        obj_df_ls.append(cells)
    if 'cytoplasm' in tables:
        obj_df_ls.append(cytoplasms)
    if 'nucleus' in tables:
        obj_df_ls.append(nuclei)
    if 'parasite' in tables:
        obj_df_ls.append(parasites)
        
    return merged_df, obj_df_ls

def annotate_conditions(df, cells=['HeLa'], cell_loc=None, parasites=['rh'], parasite_loc=None, treatments=['cm'], treatment_loc=None, types = ['col','col','col']):
    if cell_loc is None:
        df['host_cells'] = cells[0]
    else:
        cells_dict = dict(zip(cells, cell_loc))
        df['host_cells'] = df.apply(lambda row: map_values(row, cells_dict, type_=types[0]), axis=1)
    if parasite_loc is None:
        if parasites != None:
            df['parasite'] = parasites[0]
    else:
        parasites_dict = dict(zip(parasites, parasite_loc))
        df['parasite'] = df.apply(lambda row: map_values(row, parasites_dict, type_=types[1]), axis=1)
    if treatment_loc is None:
        df['treatment'] = treatments[0]
    else:
        treatments_dict = dict(zip(treatments, treatment_loc))
        df['treatment'] = df.apply(lambda row: map_values(row, treatments_dict, type_=types[2]), axis=1)
    if parasites != None:
        df['condition'] = df['parasite']+'_'+df['treatment']
    else:
        df['condition'] = df['treatment']
    return df

def group_by_well(df):
    # Separate numeric and non-numeric columns
    numeric_cols = df._get_numeric_data().columns
    non_numeric_cols = df.select_dtypes(include=['object']).columns

    # Apply mean function to numeric columns and first to non-numeric
    df_grouped = df.groupby(['plate', 'row', 'col']).agg({**{col: np.mean for col in numeric_cols}, **{col: 'first' for col in non_numeric_cols}})
    return df_grouped

def calculate_slope(df, channel, object_type):
    # Find all columns for a specific channel
    cols = [col for col in df.columns if f'{object_type}_rad_dist_channel_{channel}_bin_' in col]
    # Create an array with the number of bins, assuming bins are consecutively numbered starting from 0
    x = np.arange(len(cols))
    # Apply polyfit on each row and get the slope (degree=1 for a line)
    slopes = df[cols].apply(lambda row: np.polyfit(x, row, 1)[0], axis=1)
    return slopes

def calculate_recruitment(df, channel):
    df['parasite_cell_mean_mean'] = df[f'parasite_channel_{channel}_mean_intensity']/df[f'cytoplasm_channel_{channel}_mean_intensity']
    df['parasite_cytoplasm_mean_mean'] = df[f'parasite_channel_{channel}_mean_intensity']/df[f'cytoplasm_channel_{channel}_mean_intensity']
    df['parasite_nucleus_mean_mean'] = df[f'parasite_channel_{channel}_mean_intensity']/df[f'nucleus_channel_{channel}_mean_intensity']
    
    df['parasite_cell_q75_mean'] = df[f'parasite_channel_{channel}_percentile_75']/df[f'cytoplasm_channel_{channel}_mean_intensity']
    df['parasite_cytoplasm_q75_mean'] = df[f'parasite_channel_{channel}_percentile_75']/df[f'cytoplasm_channel_{channel}_mean_intensity']
    df['parasite_nucleus_q75_mean'] = df[f'parasite_channel_{channel}_percentile_75']/df[f'nucleus_channel_{channel}_mean_intensity']
    
    df['parasite_outside_cell_mean_mean'] = df[f'parasite_channel_{channel}_outside_mean']/df[f'cytoplasm_channel_{channel}_mean_intensity']
    df['parasite_outside_cytoplasm_mean_mean'] = df[f'parasite_channel_{channel}_outside_mean']/df[f'cytoplasm_channel_{channel}_mean_intensity']
    df['parasite_outside_nucleus_mean_mean'] = df[f'parasite_channel_{channel}_outside_mean']/df[f'nucleus_channel_{channel}_mean_intensity']
    
    df['parasite_outside_cell_q75_mean'] = df[f'parasite_channel_{channel}_outside_75_percentile']/df[f'cytoplasm_channel_{channel}_mean_intensity']
    df['parasite_outside_cytoplasm_q75_mean'] = df[f'parasite_channel_{channel}_outside_75_percentile']/df[f'cytoplasm_channel_{channel}_mean_intensity']
    df['parasite_outside_nucleus_q75_mean'] = df[f'parasite_channel_{channel}_outside_75_percentile']/df[f'nucleus_channel_{channel}_mean_intensity']

    channels = [0,1,2,3]
    object_type = 'parasite'
    for chan in channels:
        df[f'{object_type}_slope_channel_{chan}'] = calculate_slope(df, f'{chan}', object_type)
    
    object_type = 'nucleus'
    for chan in channels:
        df[f'{object_type}_slope_channel_{chan}'] = calculate_slope(df, f'{chan}', object_type)
    
    for chan in channels:
        df[f'nucleus_coordinates_{chan}'] = df[[f'nucleus_channel_{chan}_centroid_weighted_local-0', f'nucleus_channel_{chan}_centroid_weighted_local-1']].values.tolist()
        df[f'parasite_coordinates_{chan}'] = df[[f'parasite_channel_{chan}_centroid_weighted_local-0', f'parasite_channel_{chan}_centroid_weighted_local-1']].values.tolist()
        df[f'cell_coordinates_{chan}'] = df[[f'cell_channel_{chan}_centroid_weighted_local-0', f'cell_channel_{chan}_centroid_weighted_local-1']].values.tolist()
        df[f'cytoplasm_coordinates_{chan}'] = df[[f'cytoplasm_channel_{chan}_centroid_weighted_local-0', f'cytoplasm_channel_{chan}_centroid_weighted_local-1']].values.tolist()

        df[f'parasite_cell_distance_channel_{chan}'] = df.apply(lambda row: np.sqrt((row[f'parasite_coordinates_{chan}'][0] - row[f'cell_coordinates_{chan}'][0])**2 + 
                                                      (row[f'parasite_coordinates_{chan}'][1] - row[f'cell_coordinates_{chan}'][1])**2), axis=1)
        df[f'nucleus_cell_distance_channel_{chan}'] = df.apply(lambda row: np.sqrt((row[f'nucleus_coordinates_{chan}'][0] - row[f'cell_coordinates_{chan}'][0])**2 + 
                                                      (row[f'nucleus_coordinates_{chan}'][1] - row[f'cell_coordinates_{chan}'][1])**2), axis=1)
    return df

def save_filtered_cells_to_csv(src, cell_dim=4, nucleus_dim=5, parasite_dim=6, include_multinucleated=True, include_multiinfected=True, include_noninfected=False, include_border_parasites=False, verbose=False):
    mask_dims = [cell_dim, nucleus_dim, parasite_dim]
    dest = os.path.join(src, 'measurements')
    os.makedirs(dest, exist_ok=True)
    csv_file = dest+'/filtered_filelist.csv'
    mask_dim = cell_dim
    for file in os.listdir(src+'/merged'):
        path = os.path.join(src+'/merged', file)
        stack = np.load(path, allow_pickle=True)
        if not include_noninfected:
            stack = remove_noninfected(stack, cell_dim, nucleus_dim, parasite_dim)
        if include_multinucleated is not True:
            stack = remove_multiobject_cells(stack, mask_dim, cell_dim, nucleus_dim, parasite_dim, object_dim=parasite_dim)
        if include_multiinfected is not True:
            stack = remove_multiobject_cells(stack, mask_dim, cell_dim, nucleus_dim, parasite_dim, object_dim=nucleus_dim)
        if include_border_parasites is not True:
            stack = remove_border_parasites(stack, cell_dim, nucleus_dim, parasite_dim)
        for i, mask_dim in enumerate(mask_dims):
            mask = np.take(stack, mask_dim, axis=2)
            unique_labels = np.unique(mask)
            with open(csv_file, 'a', newline='') as csvfile:
                fieldnames = ['filename', 'object_label']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for label in unique_labels[1:]:
                    writer.writerow({'filename': file, 'object_label': label})
    return print(f'filtered file list saved at: {csv_file}')    

def generate_training_data_file_list(src, 
                        target='protein of interest', 
                        cell_dim=4, 
                        nucleus_dim=5, 
                        parasite_dim=6,
                        channel_of_interest=1,
                        parasite_size_min=0, 
                        nucleus_size_min=0, 
                        cell_size_min=0, 
                        parasite_min=0, 
                        nucleus_min=0, 
                        cell_min=0, 
                        target_min=0, 
                        mask_chans=[0,1,2], 
                        filter_data=False,
                        include_noninfected=False,
                        include_multiinfected=False,
                        include_multinucleated=False, 
                        cells_per_well=10, 
                        save_filtered_filelist=False):
    
    mask_dims=[cell_dim,nucleus_dim,parasite_dim]
    sns.color_palette("mako", as_cmap=True)
    print(f'channel:{channel_of_interest} = {target}')
    overlay_channels = [0, 1, 2, 3]
    overlay_channels.remove(channel_of_interest)
    overlay_channels.reverse()

    db_loc = [src+'/measurements/measurements.db']
    tables = ['cell', 'nucleus', 'parasite','cytoplasm']
    df, object_dfs = read_and_merge_data(db_loc,
                                         tables,
                                         verbose=True,
                                         include_multinucleated=include_multinucleated,
                                         include_multiinfected=include_multiinfected,
                                         include_noninfected=include_noninfected)

    if filter_data:
        df = df[df['cell_area'] > cell_size_min]
        df = df[df[f'cell_channel_{mask_chans[2]}_mean_intensity'] > cell_min]
        print(f'After cell filtration {len(df)}')
        df = df[df['nucleus_area'] > nucleus_size_min]
        df = df[df[f'nucleus_channel_{mask_chans[0]}_mean_intensity'] > nucleus_min]
        print(f'After nucleus filtration {len(df)}')
        df = df[df['parasite_area'] > parasite_size_min]
        df=df[df[f'parasite_channel_{mask_chans[1]}_mean_intensity'] > parasite_min]
        print(f'After parasite filtration {len(df)}')
        df = df[df[f'cell_channel_{channel_of_interest}_percentile_95'] > target_min]
        print(f'After channel {channel_of_interest} filtration', len(df))

    df['recruitment'] = df[f'parasite_channel_{channel_of_interest}_mean_intensity']/df[f'cytoplasm_channel_{channel_of_interest}_mean_intensity']
    return df

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

def plot_arrays(src, figuresize=50, cmap='inferno', nr=1, normalize=True, q1=1, q2=99):
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

def remove_multiobject_cells(stack, mask_dim, cell_dim, nucleus_dim, parasite_dim, object_dim):
    cell_mask = stack[:, :, mask_dim]
    nucleus_mask = stack[:, :, nucleus_dim]
    parasite_mask = stack[:, :, parasite_dim]
    object_mask = stack[:, :, object_dim]

    for cell_label in np.unique(cell_mask)[1:]:
        cell_region = cell_mask == cell_label
        labels_in_cell = np.unique(object_mask[cell_region])
        if len(labels_in_cell) > 2:
            cell_mask[cell_region] = 0
            nucleus_mask[cell_region] = 0
            for parasite_label in labels_in_cell[1:]:  # Skip the first label (0)
                parasite_mask[parasite_mask == parasite_label] = 0

    stack[:, :, cell_dim] = cell_mask
    stack[:, :, nucleus_dim] = nucleus_mask
    stack[:, :, parasite_dim] = parasite_mask
    return stack

def remove_noninfected(stack, cell_dim, nucleus_dim, parasite_dim):
    cell_mask = stack[:, :, cell_dim]
    nucleus_mask = stack[:, :, nucleus_dim]
    parasite_mask = stack[:, :, parasite_dim]

    for cell_label in np.unique(cell_mask)[1:]:
        cell_region = cell_mask == cell_label
        labels_in_cell = np.unique(parasite_mask[cell_region])
        if len(labels_in_cell) <= 1:
            cell_mask[cell_region] = 0
            nucleus_mask[cell_region] = 0

    stack[:, :, cell_dim] = cell_mask
    stack[:, :, nucleus_dim] = nucleus_mask
    return stack

def remove_border_parasites(stack, cell_dim, nucleus_dim, parasite_dim):
    cell_mask = stack[:, :, cell_dim]
    nucleus_mask = stack[:, :, nucleus_dim]
    parasite_mask = stack[:, :, parasite_dim]

    cell_labels = np.unique(cell_mask)[1:]  # Get unique cell labels, excluding background
    for cell_label in cell_labels:
        cell_region = cell_mask == cell_label
        parasites_in_cell = np.unique(parasite_mask[cell_region])
        parasites_in_cell = parasites_in_cell[parasites_in_cell != 0]  # Exclude background

        # Create a border for the cell using dilation and subtract the original cell mask
        cell_border = binary_dilation(cell_region) & ~cell_region

        for parasite_label in parasites_in_cell:
            parasite_region = parasite_mask == parasite_label
            
            # If the parasite is touching the border of the cell, remove the cell, corresponding nucleus and the parasite
            if np.any(parasite_region & cell_border):
                cell_mask[cell_region] = 0
                nucleus_mask[cell_region] = 0
                parasite_mask[parasite_region] = 0
                break

    stack[:, :, cell_dim] = cell_mask
    stack[:, :, nucleus_dim] = nucleus_mask
    stack[:, :, parasite_dim] = parasite_mask
    return stack

def remove_outside_objects(stack, cell_dim, nucleus_dim, parasite_dim):
    cell_mask = stack[:, :, cell_dim]
    nucleus_mask = stack[:, :, nucleus_dim]
    parasite_mask = stack[:, :, parasite_dim]
    parasite_labels = np.unique(parasite_mask)[1:]
    for parasite_label in parasite_labels:
        parasite_region = parasite_mask == parasite_label
        cell_in_parasite_region = np.unique(cell_mask[parasite_region])
        cell_in_parasite_region = cell_in_parasite_region[cell_in_parasite_region != 0]  # Exclude background
        if len(cell_in_parasite_region) == 0:
            parasite_mask[parasite_region] = 0
            corresponding_nucleus_region = nucleus_mask == parasite_label
            nucleus_mask[corresponding_nucleus_region] = 0
    stack[:, :, cell_dim] = cell_mask
    stack[:, :, nucleus_dim] = nucleus_mask
    stack[:, :, parasite_dim] = parasite_mask
    return stack

def plot_merged(src, cmap='inferno', cell_dim=4, nucleus_dim=5, parasite_dim=6, channel_dims=[0,1,2,3], figuresize=20, nr=1, print_object_number=True, normalize=False, normalization_percentiles=[1,99], overlay=True, overlay_chans=[3,2,0], outline_thickness=3, outline_color='gbr', backgrounds=[100,100,100,100], remove_background=False, filter_objects=False, filter_min_max=[[0,100000],[0,100000],[0,100000],[0,100000]], include_multinucleated=True, include_multiinfected=True, include_noninfected=True, include_border_parasites=True, interactive=False, verbose=False):
    mask_dims = [cell_dim, nucleus_dim, parasite_dim]
    
    if verbose:
        print(f'src:{src}, cmap:{cmap}, mask_dims:{mask_dims}, channel_dims:{channel_dims}, figuresize:{figuresize}, nr:{nr}, print_object_number:{print_object_number}, normalize:{normalize}, normalization_percentiles:{normalization_percentiles}, overlay:{overlay}, overlay_chans:{overlay_chans}, outline_thickness:{outline_thickness}, outline_color:{outline_color}, backgrounds:{backgrounds}, remove_background:{remove_background},filter_objects:{filter_objects},filter_min_max:{filter_min_max},verbose:{verbose}')
    font = figuresize/2
    index = 0
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
    for file in os.listdir(src):
        path = os.path.join(src, file)
        print(f'{path}')
        stack = np.load(path)
        if not include_noninfected:
            stack = remove_noninfected(stack, cell_dim, nucleus_dim, parasite_dim)
        if filter_objects:
            stack = remove_outside_objects(stack, cell_dim, nucleus_dim, parasite_dim)

            for i, mask_dim in enumerate(mask_dims):
                min_max = filter_min_max[i]
                mask = np.take(stack, mask_dim, axis=2)
                props = measure.regionprops_table(mask, properties=['label', 'area'])  # Measure properties of labeled image regions.
                avg_size_before = np.mean(props['area'])
                total_count_before = len(props['label'])

                valid_labels = props['label'][np.logical_and(props['area'] > min_max[0], props['area'] < min_max[1])]  # Select labels of valid size.
                stack[:, :, mask_dim] = np.isin(mask, valid_labels) * mask  # Keep only valid objects.

                props_after = measure.regionprops_table(stack[:, :, mask_dim], properties=['label', 'area']) 
                avg_size_after = np.mean(props_after['area'])
                total_count_after = len(props_after['label'])
                if mask_dim == cell_dim:
                    if include_multinucleated is not True:
                        stack = remove_multiobject_cells(stack, mask_dim, cell_dim, nucleus_dim, parasite_dim, object_dim=parasite_dim)
                    if include_multiinfected is not True:
                        stack = remove_multiobject_cells(stack, mask_dim, cell_dim, nucleus_dim, parasite_dim, object_dim=nucleus_dim)
                    if include_border_parasites is not True:
                        stack = remove_border_parasites(stack, cell_dim, nucleus_dim, parasite_dim)
                    cell_area_before = avg_size_before
                    cell_count_before = total_count_before
                    cell_area_after = avg_size_after
                    cell_count_after = total_count_after
                if mask_dim == nucleus_dim:
                    nucleus_area_before = avg_size_before
                    nucleus_count_before = total_count_before
                    nucleus_area_after = avg_size_after
                    nucleus_count_after = total_count_after
                if mask_dim == parasite_dim:
                    parasite_area_before = avg_size_before
                    parasite_count_before = total_count_before
                    parasite_area_after = avg_size_after
                    parasite_count_after = total_count_after
        image = np.take(stack, channel_dims, axis=2)
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

                # Overlay the outlines onto the RGB image
                for j in np.unique(outline)[1:]:
                    overlayed_image[outline == j] = outline_colors[i % len(outline_colors)]  # Use different color for each mask
        if index < nr:
            index += 1
            if overlay:
                fig, ax = plt.subplots(1, image.shape[-1] + len(mask_dims) + 1, figsize=(4 * figuresize, figuresize)) # Changed here
            else:
                fig, ax = plt.subplots(1, image.shape[-1] + len(mask_dims), figsize=(4 * figuresize, figuresize))
            
            if overlay:
                ax[0].imshow(overlayed_image)
                ax[0].set_title('Overlayed Image')
                ax_index = 1
            else:
                ax_index = 0
            
            if filter_objects:
                if cell_dim is not None:
                    print(f'removed {cell_count_before-cell_count_after} cells, cell size from {cell_area_before} to {cell_area_after}')
                if nucleus_dim is not None:
                    print(f'removed {nucleus_count_before-nucleus_count_after} nuclei, nuclei size from {nucleus_area_before} to {nucleus_area_after}')
                if parasite_dim is not None:
                    print(f'removed {parasite_count_before-parasite_count_after} parasites, parasite size from {parasite_area_before} to {parasite_area_after}')
            
            for v in range(0, image.shape[-1]):
                ax[v+ax_index].imshow(image[..., v], cmap=cmap)  # display first channel
                ax[v+ax_index].set_title('Image - Channel'+str(v))

            for i, mask_dim in enumerate(mask_dims):
                mask = np.take(stack, mask_dim, axis=2)
                unique_labels = np.unique(mask)
                num_objects = len(unique_labels[unique_labels != 0])
                random_colors = np.random.rand(num_objects+1, 4)
                random_colors[:, 3] = 1
                random_colors[0, :] = [0, 0, 0, 1]
                random_cmap = mpl.colors.ListedColormap(random_colors)
                
                ax[i + image.shape[-1]+ax_index].imshow(mask, cmap=random_cmap)
                ax[i + image.shape[-1]+ax_index].set_title('Mask '+ str(i))
                if print_object_number:
                    unique_objects = np.unique(mask)[1:]
                    for obj in unique_objects:
                        cy, cx = ndi.center_of_mass(mask == obj)
                        ax[i + image.shape[-1]+ax_index].text(cx, cy, str(obj), color='white', fontsize=font, ha='center', va='center')
            plt.tight_layout()
            if interactive:
                def on_click(event):
                    if event.inaxes in [ax[i + image.shape[-1]+ax_index] for i in range(len(mask_dims))]:
                        for i, ax_i in enumerate([ax[i + image.shape[-1]+ax_index] for i in range(len(mask_dims))]):
                            if event.inaxes == ax_i:
                                mask = np.take(stack, mask_dims[i], axis=2)
                                y = int(event.ydata)
                                x = int(event.xdata)
                                label = mask[y, x]
                                if label != 0:
                                    mask[mask == label] = 0
                                    stack[:, :, mask_dims[i]] = mask
                                    ax_i.imshow(mask, cmap=random_cmap)
                                    ax_i.figure.canvas.draw()

                cid = fig.canvas.mpl_connect('button_press_event', on_click)
            plt.show()
        else:
            return
        
def filter_for_tsg101_screen(df):
    
    df['cond'] = np.nan
    df['cond'] = df['cond'].astype('object')
    
    plate_list = ['p6','p7','p8','p9']
    col_list = ['c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19','c20','c21','c22','c23','c24']
    df.loc[(df['plate'].isin(plate_list)) & (df['col'].isin(col_list)), 'cond'] = 'nc'
    
    plate_list = ['p1','p2','p3','p4','p6','p7','p8','p9']
    col_list = ['c1','c2','c3']
    df.loc[(df['plate'].isin(plate_list)) & (df['col'].isin(col_list)), 'cond'] = 'pc'

    plate_list = ['p1','p2','p3','p4']
    col_list = ['c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19','c20','c21','c22','c23','c24']
    df.loc[(df['plate'].isin(plate_list)) & (df['col'].isin(col_list)), 'cond'] = 'screen'

    screen = df[df['cond']=='screen']
    nc = df[df['cond']=='nc']
    pc = df[df['cond']=='pc']
    print(f'screen:{len(screen)}, nc:{len(nc)}, pc:{len(pc)}')
    #display(df)

    all_plates = ['p1','p2','p3','p4','p6','p7','p8','p9']
    nc_ls = []
    pc_ls = []

    for plate in all_plates:
        p = nc[nc['plate']==plate]
        print(f'NC: plate {plate}: {len(p)}')
        nc_ls.append(len(p))

        p = pc[pc['plate']==plate]
        print(f'PC: plate {plate}: {len(p)}')
        pc_ls.append(len(p))

    all_ls = nc_ls+pc_ls
    filtered_list = [x for x in all_ls if x != 0]
    min_value = min(filtered_list)

    nc_df = pd.DataFrame()
    pc_df = pd.DataFrame()
    
    for i,plate in enumerate(all_plates):
        p = nc[nc['plate']==plate]
        if len(p) >= min_value:
            rows = p.sample(min_value)
            nc_df = pd.concat([nc_df, rows], axis=0)

        p = pc[pc['plate']==plate]
        if len(p) >= min_value:
            rows = p.sample(int(min_value/2))
            pc_df = pd.concat([pc_df, rows], axis=0)
    
    nc_not_in_trainset = ~nc.index.isin(nc_df.index)
    nc_not_in_trainset = nc[nc_not_in_trainset]
    
    pc_not_in_trainset = ~pc.index.isin(pc_df.index)
    pc_not_in_trainset = pc[pc_not_in_trainset]
    
    return nc_df, pc_df, screen, nc_not_in_trainset, pc_not_in_trainset

def get_paths_from_db(df, png_df, image_type='cell_png'):
    objects = df.index.tolist()
    filtered_df = png_df[png_df['png_path'].str.contains(image_type) & png_df['prcfo'].isin(objects)]
    return filtered_df

def training_dataset_from_annotation(db_path, dst, annotation_column='test', annotated_classes=(1, 2)):
    all_paths = []
    
    # Connect to the database and retrieve the image paths and annotations
    print(f'Reading DataBase: {db_path}')
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Prepare the query with parameterized placeholders for annotated_classes
        placeholders = ','.join('?' * len(annotated_classes))
        query = f"SELECT png_path, {annotation_column} FROM png_list WHERE {annotation_column} IN ({placeholders})"
        cursor.execute(query, annotated_classes)

        while True:
            rows = cursor.fetchmany(1000)
            if not rows:
                break
            for row in rows:
                all_paths.append(row)

    # Filter paths based on annotation
    class_paths = []
    for class_ in annotated_classes:
        class_paths_temp = [path for path, annotation in all_paths if annotation == class_]
        class_paths.append(class_paths_temp)

    print(f'Generated a list of lists from annotation of {len(class_paths)} classes')
    return class_paths

def generate_dataset_from_lists(dst, class_data, classes, test_split=0.1):
    # Make sure that the length of class_data matches the length of classes
    if len(class_data) != len(classes):
        raise ValueError("class_data and classes must have the same length.")

    total_files = sum(len(data) for data in class_data)
    processed_files = 0
    
    for cls, data in zip(classes, class_data):
        # Create directories
        train_class_dir = os.path.join(dst, f'train/{cls}')
        test_class_dir = os.path.join(dst, f'test/{cls}')
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Split the data
        train_data, test_data = train_test_split(data, test_size=test_split, shuffle=True, random_state=42)
        
        # Copy train files
        for path in train_data:
            shutil.copy(path, os.path.join(train_class_dir, os.path.basename(path)))
            processed_files += 1
            print(f'{processed_files}/{total_files}', end='\r', flush=True)

        # Copy test files
        for path in test_data:
            shutil.copy(path, os.path.join(test_class_dir, os.path.basename(path)))
            processed_files += 1
            print(f'{processed_files}/{total_files}', end='\r', flush=True)

    # Print summary
    for cls in classes:
        train_class_dir = os.path.join(dst, f'train/{cls}')
        test_class_dir = os.path.join(dst, f'test/{cls}')
        print(f'Train class {cls}: {len(os.listdir(train_class_dir))}, Test class {cls}: {len(os.listdir(test_class_dir))}')

    return

def generate_training_dataset(src, mode='annotation', annotation_column='test', annotated_classes=[1,2], classes=['nc','pc'], size=200, test_split=0.1, class_metadata=[['c1'],['c2']], metadata_type_by='col', channel_of_interest=3, custom_measurement=None, tables=None):
    
    db_path = os.path.join(src, 'measurements','measurements.db')
    dst = os.path.join(src, 'datasets', 'training')
    
    if mode == 'annotation':
        class_paths_ls_2 = []
        class_paths_ls = training_dataset_from_annotation(db_path, dst, annotation_column, annotated_classes=annotated_classes)
        for class_paths in class_paths_ls:
            class_paths_temp = random.sample(class_paths, size)
            class_paths_ls_2.append(class_paths_temp)
        class_paths_ls = class_paths_ls_2

    elif mode == 'metadata':
        class_paths_ls = []
        [df] = read_db(db_loc=db_path, tables=['png_list'])
        df['metadata_based_class'] = pd.NA
        for i, class_ in enumerate(classes):
            ls = class_metadata[i]
            df.loc[df[metadata_type_by].isin(ls), 'metadata_based_class'] = class_
            
        for class_ in classes:
            class_temp_df = df[df['metadata_based_class'] == class_]
            class_paths_temp = random.sample(class_temp_df['png_path'].tolist(), size)
            class_paths_ls.append(class_paths_temp)
    
    elif mode == 'recruitment':
        class_paths_ls = []
        if not isinstance(tables, list):
            tables = ['cell', 'nucleus', 'parasite','cytoplasm']
        
        df, _ = read_and_merge_data(locs=[db_path],
                                    tables=tables,
                                    verbose=False,
                                    include_multinucleated=True,
                                    include_multiinfected=True,
                                    include_noninfected=True)
        
        df = annotate_conditions(df, cells=['HeLa'], cell_loc=None, parasites=['parasite'], parasite_loc=None, treatments=classes, treatment_loc=class_metadata, types = ['col','col',metadata_type_by])
        [png_list_df] = read_db(db_loc=db_path, tables=['png_list'])
	    
        if custom_measurement != None:
            if isinstance(custom_measurement, list):
                if len() == 2:
                    print(f'Classes will be defined by the Q1 and Q3 quantiles of recruitment ({custom_measurement[0]}/{custom_measurement[1]})')
                    df['recruitment'] = df[f'{custom_measurement[0]}']/df[f'{custom_measurement[1]}']
                if len() == 1:
                    print(f'Classes will be defined by the Q1 and Q3 quantiles of recruitment ({custom_measurement[0]})')
                    df['recruitment'] = df[f'{custom_measurement[0]}']
        else:
            print(f'Classes will be defined by the Q1 and Q3 quantiles of recruitment (parasite/cytoplasm for channel {channel_of_interest})')
            df['recruitment'] = df[f'parasite_channel_{channel_of_interest}_mean_intensity']/df[f'cytoplasm_channel_{channel_of_interest}_mean_intensity']
		
        q25 = df['recruitment'].quantile(0.25)
        q75 = df['recruitment'].quantile(0.75)
        df_lower = df[df['recruitment'] <= q25]
        df_upper = df[df['recruitment'] >= q75]
        
        class_paths_lower = get_paths_from_db(df=df_lower, png_df=png_list_df, image_type='cell_png')
        class_paths_lower = random.sample(class_paths_lower['png_path'].tolist(), size)
        class_paths_ls.append(class_paths_lower)
        
        class_paths_upper = get_paths_from_db(df=df_upper, png_df=png_list_df, image_type='cell_png')
        class_paths_upper = random.sample(class_paths_upper['png_path'].tolist(), size)
        class_paths_ls.append(class_paths_upper)
    
    generate_dataset_from_lists(dst, class_data=class_paths_ls, classes=classes, test_split=0.1)
    
    return

def generate_training_data_file_list(src, 
                        target='protein of interest', 
                        cell_dim=4, 
                        nucleus_dim=5, 
                        parasite_dim=6,
                        channel_of_interest=1,
                        parasite_size_min=0, 
                        nucleus_size_min=0, 
                        cell_size_min=0, 
                        parasite_min=0, 
                        nucleus_min=0, 
                        cell_min=0, 
                        target_min=0, 
                        mask_chans=[0,1,2], 
                        filter_data=False,
                        include_noninfected=False,
                        include_multiinfected=False,
                        include_multinucleated=False, 
                        cells_per_well=10, 
                        save_filtered_filelist=False):
    
    mask_dims=[cell_dim,nucleus_dim,parasite_dim]
    sns.color_palette("mako", as_cmap=True)
    print(f'channel:{channel_of_interest} = {target}')
    overlay_channels = [0, 1, 2, 3]
    overlay_channels.remove(channel_of_interest)
    overlay_channels.reverse()

    db_loc = [src+'/measurements/measurements.db']
    tables = ['cell', 'nucleus', 'parasite','cytoplasm']
    df, object_dfs = read_and_merge_data(db_loc,
                                         tables,
                                         verbose=True,
                                         include_multinucleated=include_multinucleated,
                                         include_multiinfected=include_multiinfected,
                                         include_noninfected=include_noninfected)
    if filter_data:
        df = df[df['cell_area'] > cell_size_min]
        df = df[df[f'cell_channel_{mask_chans[2]}_mean_intensity'] > cell_min]
        print(f'After cell filtration {len(df)}')
        df = df[df['nucleus_area'] > nucleus_size_min]
        df = df[df[f'nucleus_channel_{mask_chans[0]}_mean_intensity'] > nucleus_min]
        print(f'After nucleus filtration {len(df)}')
        df = df[df['parasite_area'] > parasite_size_min]
        df=df[df[f'parasite_channel_{mask_chans[1]}_mean_intensity'] > parasite_min]
        print(f'After parasite filtration {len(df)}')
        df = df[df[f'cell_channel_{channel_of_interest}_percentile_95'] > target_min]
        print(f'After channel {channel_of_interest} filtration', len(df))

    df['recruitment'] = df[f'parasite_channel_{channel_of_interest}_mean_intensity']/df[f'cytoplasm_channel_{channel_of_interest}_mean_intensity']
    
    png_list_df = read_db(db_loc=db_loc[0], tables=['png_list'])
    png_list_df = png_list_df[0]
    nc_df, pc_df, screen, nc_not_in_trainset, pc_not_in_trainset = filter_for_tsg101_screen(df=df)
    nc_files = get_paths_from_db(df=nc_df,png_df=png_list_df, image_type='cell_png')
    pc_files = get_paths_from_db(df=pc_df,png_df=png_list_df, image_type='cell_png')
    screen_files = get_paths_from_db(df=screen,png_df=png_list_df, image_type='cell_png')
    nc_files_not_in_train = get_paths_from_db(df=nc_not_in_trainset,png_df=png_list_df, image_type='cell_png')    
    pc_files_not_in_train = get_paths_from_db(df=pc_not_in_trainset,png_df=png_list_df, image_type='cell_png')
    
    print(f'NC: {len(nc_files)}, PC:{len(pc_files)}, Screen:{len(screen_files)}, ~NC:{len(nc_files_not_in_train)}, ~PC:{len(pc_files_not_in_train)}')
    return nc_files.png_path.tolist(), pc_files.png_path.tolist(), screen_files.png_path.tolist(), nc_files_not_in_train.png_path.tolist(), pc_files_not_in_train.png_path.tolist()

def save_file_lists(dst, data_set, ls):
    df = pd.DataFrame(ls, columns=[data_set])  
    df.to_csv(f'{dst}/{data_set}.csv', index=False)
    return

def augment_single_image(args):
    img_path, dst = args
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    filename = os.path.basename(img_path).split('.')[0]

    # Original Image
    cv2.imwrite(os.path.join(dst, f"{filename}_original.png"), img)
    
    # 90 degree rotation
    img_rot_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(os.path.join(dst, f"{filename}_rot_90.png"), img_rot_90)
    
    # 180 degree rotation
    img_rot_180 = cv2.rotate(img, cv2.ROTATE_180)
    cv2.imwrite(os.path.join(dst, f"{filename}_rot_180.png"), img_rot_180)

    # 270 degree rotation
    img_rot_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(os.path.join(dst, f"{filename}_rot_270.png"), img_rot_270)

    # Horizontal Flip
    img_flip_hor = cv2.flip(img, 1)
    cv2.imwrite(os.path.join(dst, f"{filename}_flip_hor.png"), img_flip_hor)

    # Vertical Flip
    img_flip_ver = cv2.flip(img, 0)
    cv2.imwrite(os.path.join(dst, f"{filename}_flip_ver.png"), img_flip_ver)

def augment_images(file_paths, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)

    args_list = [(img_path, dst) for img_path in file_paths]

    with Pool(cpu_count()) as pool:
        pool.map(augment_single_image, args_list)

def augment_classes(dst, nc, pc, generate=True,move=True):
    aug_nc = os.path.join(dst,'aug_nc')
    aug_pc = os.path.join(dst,'aug_pc')
    all_ = len(nc)+len(pc)
    if generate == True:
        os.makedirs(aug_nc, exist_ok=True)
        if __name__ == '__main__':
            augment_images(file_paths=nc, dst=aug_nc)

        os.makedirs(aug_pc, exist_ok=True)
        if __name__ == '__main__':
            augment_images(file_paths=pc, dst=aug_pc)

    if move == True:
        aug = os.path.join(dst,'aug')
        aug_train_nc = os.path.join(aug,'train/nc')
        aug_train_pc = os.path.join(aug,'train/pc')
        aug_test_nc = os.path.join(aug,'test/nc')
        aug_test_pc = os.path.join(aug,'test/pc')

        os.makedirs(aug_train_nc, exist_ok=True)
        os.makedirs(aug_train_pc, exist_ok=True)
        os.makedirs(aug_test_nc, exist_ok=True)
        os.makedirs(aug_test_pc, exist_ok=True)

        aug_nc_list = [os.path.join(aug_nc, file) for file in os.listdir(aug_nc)]
        aug_pc_list = [os.path.join(aug_pc, file) for file in os.listdir(aug_pc)]

        nc_train_data, nc_test_data = train_test_split(aug_nc_list, test_size=0.1, shuffle=True, random_state=42)
        pc_train_data, pc_test_data = train_test_split(aug_pc_list, test_size=0.1, shuffle=True, random_state=42)

        i=0
        for path in nc_train_data:
            i+=1
            shutil.move(path, os.path.join(aug_train_nc, os.path.basename(path)))
            print(f'{i}/{all_}', end='\r', flush=True)
        for path in nc_test_data:
            i+=1
            shutil.move(path, os.path.join(aug_test_nc, os.path.basename(path)))
            print(f'{i}/{all_}', end='\r', flush=True)
        for path in pc_train_data:
            i+=1
            shutil.move(path, os.path.join(aug_train_pc, os.path.basename(path)))
            print(f'{i}/{all_}', end='\r', flush=True)
        for path in pc_test_data:
            i+=1
            shutil.move(path, os.path.join(aug_test_pc, os.path.basename(path)))
            print(f'{i}/{all_}', end='\r', flush=True)
        print(f'Train nc: {len(os.listdir(aug_train_nc))}, Train pc:{len(os.listdir(aug_test_nc))}, Test nc:{len(os.listdir(aug_train_pc))}, Test pc:{len(os.listdir(aug_test_pc))}')
        return
    
class Image_viewingApp:
    def __init__(self, root, img_size=(200, 200), rows=5, columns=5, channels=None):
        self.root = root
        self.index = 0
        self.grid_rows = rows
        self.grid_cols = columns
        self.image_size = img_size
        self.channels = channels
        self.images = []
        self.photo_images = []
        self.labels = []

        for i in range(self.grid_rows * self.grid_cols):
            label = tk.Label(self.root)
            label.grid(row=i // self.grid_cols, column=i % self.grid_cols)
            self.labels.append(label)

        self.folder_path = filedialog.askdirectory(title="Select the folder containing images")
        if self.folder_path:
            self.images = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def filter_channels(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        r, g, b = img.split()
        if self.channels:
            if 'r' not in self.channels:
                r = r.point(lambda _: 0)
            if 'g' not in self.channels:
                g = g.point(lambda _: 0)
            if 'b' not in self.channels:
                b = b.point(lambda _: 0)
        return Image.merge("RGB", (r, g, b))

    def load_images(self):
        start = self.index
        end = start + (self.grid_rows * self.grid_cols)
        images_to_load = self.images[start:end]

        with ThreadPoolExecutor() as executor:
            loaded_images = list(executor.map(self.load_single_image, images_to_load))

        self.photo_images = []
        for i, img in enumerate(loaded_images):
            photo = ImageTk.PhotoImage(img)
            self.photo_images.append(photo)
            label = self.labels[i]
            label.config(image=photo)

        self.root.update()

    def load_single_image(self, path):
        img = Image.open(path)
        img = self.filter_channels(img)
        img = img.resize(self.image_size)
        return img

    def next_page(self):
        self.index += self.grid_rows * self.grid_cols
        if self.index >= len(self.images):
            self.index = len(self.images) - (self.grid_rows * self.grid_cols)
        self.load_images()

    def previous_page(self):
        self.index -= self.grid_rows * self.grid_cols
        if self.index < 0:
            self.index = 0
        self.load_images()

def view_images(geom="1000x1100", img_size=(200, 200), rows=5, columns=5, channels=None):
    root = tk.Tk()
    root.geometry(geom)
    app = Image_viewingApp(root, img_size=img_size, rows=rows, columns=columns, channels=channels)
    
    next_button = tk.Button(root, text="Next", command=app.next_page)
    next_button.grid(row=app.grid_rows, column=app.grid_cols - 1)
    back_button = tk.Button(root, text="Back", command=app.previous_page)
    back_button.grid(row=app.grid_rows, column=app.grid_cols - 2)
    exit_button = tk.Button(root, text="Exit", command=root.quit)
    exit_button.grid(row=app.grid_rows, column=app.grid_cols - 3)

    app.load_images()
    root.mainloop()
    
def apply_model(src, model_path, image_size=224, batch_size=64, normalize=True, num_workers=10):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(image_size, image_size)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(image_size, image_size))])
    
    model = torch.load(model_path)
    print(model)
    
    print(f'Loading dataset in {src} with {len(src)} images')
    dataset = NoClassDataset(data_dir=src, transform=transform, shuffle=True, load_to_memory=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print(f'Loaded {len(src)} images')
    
    result_loc = os.path.splitext(model_path)[0]+datetime.date.today().strftime('%y%m%d')+'_'+os.path.splitext(model_path)[1]+'_test_result.csv'
    print(f'Results wil be saved in: {result_loc}')
    
    model.eval()
    model = model.to(device)
    prediction_pos_probs = []
    filenames_list = []
    with torch.no_grad():
        for batch_idx, (batch_images, filenames) in enumerate(data_loader, start=1):
            images = batch_images.to(torch.float).to(device)
            outputs = model(images)
            batch_prediction_pos_prob = torch.sigmoid(outputs).cpu().numpy()
            prediction_pos_probs.extend(batch_prediction_pos_prob.tolist())
            filenames_list.extend(filenames)
            print(f'\rbatch: {batch_idx}/{len(data_loader)}', end='\r', flush=True)
    data = {'path':filenames_list, 'pred':prediction_pos_probs}
    df = pd.DataFrame(data, index=None)
    df.to_csv(result_loc, index=True, header=True, mode='w')
    torch.cuda.empty_cache()
    torch.cuda.memory.empty_cache()
    return df

def annotate_predictions(csv_loc):
    df = pd.read_csv(csv_loc)
    df['filename'] = df['path'].apply(lambda x: x.split('/')[-1])
    df[['plate', 'well', 'field', 'object']] = df['filename'].str.split('_', expand=True)
    df['object'] = df['object'].str.replace('.png', '')
    
    def assign_condition(row):
        plate = int(row['plate'])
        col = int(row['well'][1:])
        
        if col > 3:
            if plate in [1, 2, 3, 4]:
                return 'screen'
            elif plate in [5, 6, 7, 8]:
                return 'pc'
        elif col in [1, 2, 3]:
            return 'nc'
        else:
            return ''

    df['cond'] = df.apply(assign_condition, axis=1)
    return df

class TarImageDataset(Dataset):
    def __init__(self, tar_path, transform=None):
        self.tar_path = tar_path
        self.transform = transform

        # Open the tar file just to build the list of members
        with tarfile.open(self.tar_path, 'r') as f:
            self.members = [m for m in f.getmembers() if m.isfile()]

    def __len__(self):
        return len(self.members)

    def __getitem__(self, idx):
        with tarfile.open(self.tar_path, 'r') as f:
            m = self.members[idx]
            img_file = f.extractfile(m)
            img = Image.open(BytesIO(img_file.read())).convert("RGB")
            
        if self.transform:
            img = self.transform(img)
        
        return img, m.name

def apply_model_to_tar(tar_path, model_path, file_type='cell_png', image_size=224, batch_size=64, normalize=True, preload='images', num_workers=10, verbose=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(image_size, image_size)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(image_size, image_size))])
    
    if verbose:
        print(f'Loading model from {model_path}')
        print(f'Loading dataset from {tar_path}')
        
    model = torch.load(model_path)
    
    dataset = TarImageDataset(tar_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    model_name = os.path.splitext(os.path.basename(model_path))[0] 
    dataset_name = os.path.splitext(os.path.basename(tar_path))[0]  
    date_name = datetime.date.today().strftime('%y%m%d')
    dst = os.path.dirname(tar_path)
    result_loc = f'{dst}/{date_name}_{dataset_name}_{model_name}_result.csv'

    model.eval()
    model = model.to(device)
    
    if verbose:
        print(model)
        print(f'Generated dataset with {len(dataset)} images')
        print(f'Generating loader from {len(data_loader)} batches')
        print(f'Results wil be saved in: {result_loc}')
        print(f'Model is in eval mode')
        print(f'Model loaded to device')
        
    prediction_pos_probs = []
    filenames_list = []
    gc.collect()
    with torch.no_grad():
        for batch_idx, (batch_images, filenames) in enumerate(data_loader, start=1):
            images = batch_images.to(torch.float).to(device)
            outputs = model(images)
            batch_prediction_pos_prob = torch.sigmoid(outputs).cpu().numpy()
            prediction_pos_probs.extend(batch_prediction_pos_prob.tolist())
            filenames_list.extend(filenames)
            print(f'\rbatch: {batch_idx}/{len(data_loader)}', end='\r', flush=True)

    data = {'path':filenames_list, 'pred':prediction_pos_probs}
    df = pd.DataFrame(data, index=None)
    df.to_csv(result_loc, index=True, header=True, mode='w')
    torch.cuda.empty_cache()
    torch.cuda.memory.empty_cache()
    return df

def init_globals(counter_, lock_):
    global counter, lock
    counter = counter_
    lock = lock_

def add_images_to_tar(args):
    global counter, lock, total_images
    paths_chunk, tar_path = args
    with tarfile.open(tar_path, 'w') as tar:
        for img_path in paths_chunk:
            arcname = os.path.basename(img_path)
            try:
                tar.add(img_path, arcname=arcname)
                with lock:
                    counter.value += 1
                    print(f"\rProcessed: {counter.value}/{total_images}", end='', flush=True)
            except FileNotFoundError:
                print(f"File not found: {img_path}")
    return tar_path

def generate_dataset(src, file_type=None, experiment='TSG101_screen', sample=None):
	
    db_path = os.path.join(src, 'measurements','measurements.db')
    dst = os.path.join(src, 'datasets')
	
    global total_images
    all_paths = []
    
    # Connect to the database and retrieve the image paths
    print(f'Reading DataBase: {db_path}')
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        if file_type:
            cursor.execute("SELECT png_path FROM png_list WHERE png_path LIKE ?", (f"%{file_type}%",))
        else:
            cursor.execute("SELECT png_path FROM png_list")
        while True:
            rows = cursor.fetchmany(1000)
            if not rows:
                break
            all_paths.extend([row[0] for row in rows])
    
    if isinstance(sample, int):
        selected_paths = random.sample(all_paths, sample)
        print(f'Random selection of {len(selected_paths)} paths')
    else:
        selected_paths = all_paths
        random.shuffle(selected_paths)
        print(f'All paths: {len(selected_paths)} paths')
        
    total_images = len(selected_paths)
    print(f'found {total_images} images')
    
    # Create a temp folder in dst
    temp_dir = os.path.join(dst, "temp_tars")
    os.makedirs(temp_dir, exist_ok=True)

    # Chunking the data
    if len(selected_paths) > 10000:
        num_procs = cpu_count()-2
        chunk_size = len(selected_paths) // num_procs
        remainder = len(selected_paths) % num_procs
    else:
        num_procs = 2
        chunk_size = len(selected_paths) // 2
        remainder = 0

    paths_chunks = []
    start = 0
    for i in range(num_procs):
        end = start + chunk_size + (1 if i < remainder else 0)
        paths_chunks.append(selected_paths[start:end])
        start = end

    temp_tar_files = [os.path.join(temp_dir, f'temp_{i}.tar') for i in range(num_procs)]
    
    # Initialize the shared objects
    counter_ = Value('i', 0)
    lock_ = Lock()

    ctx = multiprocessing.get_context('spawn')
    
    print(f'Generating temporary tar files in {dst}')
    
    # Combine the temporary tar files into a final tar
    date_name = datetime.date.today().strftime('%y%m%d')
    tar_name = f'{date_name}_{experiment}_{file_type}.tar'
    if os.path.exists(tar_name):
        number = random.randint(1, 100)
        tar_name_2 = f'{date_name}_{experiment}_{file_type}_{number}.tar'
        print(f'Warning: {os.path.basename(tar_name)} exists saving as {os.path.basename(tar_name_2)} ')
        tar_name = tar_name_2
    
    # Add the counter and lock to the arguments for pool.map
    print(f'Merging temporary files')
    #with Pool(processes=num_procs, initializer=init_globals, initargs=(counter_, lock_)) as pool:
    #    results = pool.map(add_images_to_tar, zip(paths_chunks, temp_tar_files))

    with ctx.Pool(processes=num_procs, initializer=init_globals, initargs=(counter_, lock_)) as pool:
        results = pool.map(add_images_to_tar, zip(paths_chunks, temp_tar_files))
    
    with tarfile.open(os.path.join(dst, tar_name), 'w') as final_tar:
        for tar_path in results:
            with tarfile.open(tar_path, 'r') as t:
                for member in t.getmembers():
                    t.extract(member, path=dst)
                    final_tar.add(os.path.join(dst, member.name), arcname=member.name)
                    os.remove(os.path.join(dst, member.name))
            os.remove(tar_path)

    # Delete the temp folder
    shutil.rmtree(temp_dir)
    print(f"\nSaved {total_images} images to {os.path.join(dst, tar_name)}")
    
def map_wells_png(file_name):
    try:
        root, ext = os.path.splitext(file_name)
        parts = root.split('_')
        plate = 'p' + parts[0][5:]
        field = 'f' + str(int(parts[2]))
        well = parts[1]
        row = 'r' + str(string.ascii_uppercase.index(well[0]) + 1)
        column = 'c' + str(int(well[1:]))
        cell_id = 'o' + str(int(parts[3]))
        prcfo = '_'.join([plate, row, column, field, cell_id])
    except Exception as e:
        print(f"Error processing filename: {file_name}")
        print(f"Error: {e}")
        plate, row, column, field, cell_id, prcfo = 'error','error','error','error','error','error'
    return plate, row, column, field, cell_id, prcfo

def annotate_results(pred_loc):
    df = pd.read_csv(pred_loc)
    df = df.copy()
    pc_col_list = ['c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19','c20','c21','c22','c23','c24']
    pc_plate_list = ['p6','p7','p8', 'p9']
        
    nc_col_list = ['c1','c2','c3']
    nc_plate_list = ['p1','p2','p3','p4','p6','p7','p8', 'p9']
    
    screen_col_list = ['c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19','c20','c21','c22','c23','c24']
    screen_plate_list = ['p1','p2','p3','p4']
    
    df[['plate', 'row', 'col', 'field', 'cell_id', 'prcfo']] = df['path'].apply(lambda x: pd.Series(map_wells_png(x)))
    
    df.loc[(df['col'].isin(pc_col_list)) & (df['plate'].isin(pc_plate_list)), 'condition'] = 'pc'
    df.loc[(df['col'].isin(nc_col_list)) & (df['plate'].isin(nc_plate_list)), 'condition'] = 'nc'
    df.loc[(df['col'].isin(screen_col_list)) & (df['plate'].isin(screen_plate_list)), 'condition'] = 'screen'

    df = df.dropna(subset=['condition'])
    display(df)
    return df

def merge_pred_mes(src,
                   pred_loc,
                   target='protein of interest', 
                   cell_dim=4, 
                   nucleus_dim=5, 
                   parasite_dim=6,
                   channel_of_interest=1,
                   parasite_size_min=0, 
                   nucleus_size_min=0, 
                   cell_size_min=0, 
                   parasite_min=0, 
                   nucleus_min=0, 
                   cell_min=0, 
                   target_min=0, 
                   mask_chans=[0,1,2], 
                   filter_data=False,
                   include_noninfected=False,
                   include_multiinfected=False,
                   include_multinucleated=False, 
                   cells_per_well=10, 
                   save_filtered_filelist=False,
                   verbose=False):
    
    mask_chans=[cell_dim,nucleus_dim,parasite_dim]
    sns.color_palette("mako", as_cmap=True)
    print(f'channel:{channel_of_interest} = {target}')
    overlay_channels = [0, 1, 2, 3]
    overlay_channels.remove(channel_of_interest)
    overlay_channels.reverse()

    db_loc = [src+'/measurements/measurements.db']
    tables = ['cell', 'nucleus', 'parasite','cytoplasm']
    df, object_dfs = read_and_merge_data(db_loc,
                                         tables,
                                         verbose=True,
                                         include_multinucleated=include_multinucleated,
                                         include_multiinfected=include_multiinfected,
                                         include_noninfected=include_noninfected)
    if filter_data:
        df = df[df['cell_area'] > cell_size_min]
        df = df[df[f'cell_channel_{mask_chans[2]}_mean_intensity'] > cell_min]
        print(f'After cell filtration {len(df)}')
        df = df[df['nucleus_area'] > nucleus_size_min]
        df = df[df[f'nucleus_channel_{mask_chans[0]}_mean_intensity'] > nucleus_min]
        print(f'After nucleus filtration {len(df)}')
        df = df[df['parasite_area'] > parasite_size_min]
        df=df[df[f'parasite_channel_{mask_chans[1]}_mean_intensity'] > parasite_min]
        print(f'After parasite filtration {len(df)}')
        df = df[df[f'cell_channel_{channel_of_interest}_percentile_95'] > target_min]
        print(f'After channel {channel_of_interest} filtration', len(df))

    df['recruitment'] = df[f'parasite_channel_{channel_of_interest}_mean_intensity']/df[f'cytoplasm_channel_{channel_of_interest}_mean_intensity']
    
    pred_df = annotate_results(pred_loc=pred_loc)
    
    if verbose:
        plot_histograms_and_stats(df=pred_df)
        
    pred_df.set_index('prcfo', inplace=True)
    pred_df = pred_df.drop(columns=['plate', 'row', 'col', 'field'])

    joined_df = df.join(pred_df, how='inner')
    
    if verbose:
        plot_histograms_and_stats(df=joined_df)
        
    #dv = joined_df.copy()
    #if 'prc' not in dv.columns:
    #dv['prc'] = dv['plate'] + '_' + dv['row'] + '_' + dv['col']
    #dv = dv[['pred']].groupby('prc').mean()
    #dv.set_index('prc', inplace=True)
    
    #loc = '/mnt/data/CellVoyager/20x/tsg101/crispr_screen/all/measurements/dv.csv'
    #dv.to_csv(loc, index=True, header=True, mode='w')

    return joined_df

def plot_histograms_and_stats(df):
    conditions = df['condition'].unique()
    
    for condition in conditions:
        subset = df[df['condition'] == condition]
        
        # Calculate the statistics
        mean_pred = subset['pred'].mean()
        over_0_5 = sum(subset['pred'] > 0.5)
        under_0_5 = sum(subset['pred'] <= 0.5)

        # Print the statistics
        print(f"Condition: {condition}")
        print(f"Number of rows: {len(subset)}")
        print(f"Mean of pred: {mean_pred}")
        print(f"Count of pred values over 0.5: {over_0_5}")
        print(f"Count of pred values under 0.5: {under_0_5}")
        print(f"Percent positive: {(over_0_5/(over_0_5+under_0_5))*100}")
        print(f"Percent negative: {(under_0_5/(over_0_5+under_0_5))*100}")
        print('-'*40)
        
        # Plot the histogram
        plt.figure(figsize=(10,6))
        plt.hist(subset['pred'], bins=30, edgecolor='black')
        plt.axvline(mean_pred, color='red', linestyle='dashed', linewidth=1, label=f"Mean = {mean_pred:.2f}")
        plt.title(f'Histogram for pred - Condition: {condition}')
        plt.xlabel('Pred Value')
        plt.ylabel('Count')
        plt.legend()
        plt.show()
        
def process_reads(df, min_reads, min_wells, max_wells, gene_column, remove_outliers=False):
    print('start',len(df))
    df = df[df['count'] >= min_reads]
    print('after filtering min reads',min_reads, len(df))
    reads_ls = df['count']
    stats_dict = {}
    stats_dict['screen_reads_mean'] = np.mean(reads_ls)
    stats_dict['screen_reads_sd'] = np.std(reads_ls)
    stats_dict['screen_reads_var'] = np.var(reads_ls)
    
    well_read_sum = pd.DataFrame(df.groupby(['prc']).sum())
    well_read_sum = well_read_sum.rename({'count': 'well_read_sum'}, axis=1)
    well_sgRNA_count = pd.DataFrame(df.groupby(['prc']).count()[gene_column])
    well_sgRNA_count = well_sgRNA_count.rename({gene_column: 'gRNAs_per_well'}, axis=1)
    well_seq = pd.merge(well_read_sum, well_sgRNA_count, how='inner', suffixes=('', '_right'), left_index=True, right_index=True)
    gRNA_well_count = pd.DataFrame(df.groupby([gene_column]).count()['prc'])
    gRNA_well_count = gRNA_well_count.rename({'prc': 'gRNA_well_count'}, axis=1)
    df = pd.merge(df, well_seq, on='prc', how='inner', suffixes=('', '_right'))
    df = pd.merge(df, gRNA_well_count, on=gene_column, how='inner', suffixes=('', '_right'))

    df = df[df['gRNA_well_count'] >= min_wells]
    df = df[df['gRNA_well_count'] <= max_wells]
    
    if remove_outliers:
        clf = IsolationForest(contamination='auto', random_state=42, n_jobs=20)
        #clf.fit(df.select_dtypes(include=['int', 'float']))
        clf.fit(df[["gRNA_well_count", "count"]])
        outlier_array = clf.predict(df[["gRNA_well_count", "count"]])
        #outlier_array = clf.predict(df.select_dtypes(include=['int', 'float']))
        outlier_df = pd.DataFrame(outlier_array, columns=['outlier'])
        df['outlier'] =  outlier_df['outlier']
        outliers = pd.DataFrame(df[df['outlier']==-1])
        df = pd.DataFrame(df[df['outlier']==1])
        print('removed',len(outliers), 'outliers', 'inlers',len(df))
    
    columns_to_drop = ['gRNA_well_count','gRNAs_per_well', 'well_read_sum']#, 'outlier']
    df = df.drop(columns_to_drop, axis=1)

    plates = ['p1', 'p2', 'p3', 'p4']
    df = df[df.plate.isin(plates) == True]
    print('after filtering out p5,p6,p7,p8',len(df))

    gRNA_well_count = pd.DataFrame(df.groupby([gene_column]).count()['prc'])
    gRNA_well_count = gRNA_well_count.rename({'prc': 'gRNA_well_count'}, axis=1)
    df = pd.merge(df, gRNA_well_count, on=gene_column, how='inner', suffixes=('', '_right'))
    well_read_sum = pd.DataFrame(df.groupby(['prc']).sum())
    well_read_sum = well_read_sum.rename({'count': 'well_read_sum'}, axis=1)
    well_sgRNA_count = pd.DataFrame(df.groupby(['prc']).count()[gene_column])
    well_sgRNA_count = well_sgRNA_count.rename({gene_column: 'gRNAs_per_well'}, axis=1)
    well_seq = pd.merge(well_read_sum, well_sgRNA_count, how='inner', suffixes=('', '_right'), left_index=True, right_index=True)
    df = pd.merge(df, well_seq, on='prc', how='inner', suffixes=('', '_right'))

    columns_to_drop = [col for col in df.columns if col.endswith('_right')]
    columns_to_drop2 = [col for col in df.columns if col.endswith('0')]
    columns_to_drop = columns_to_drop + columns_to_drop2
    df = df.drop(columns_to_drop, axis=1)
    return df, stats_dict

def generate_fraction_map(df, gene_column, min_frequency=0.0):
    df['fraction'] = df['count']/df['well_read_sum']
    genes = df[gene_column].unique().tolist()
    wells = df['prc'].unique().tolist()
    print(len(genes),len(wells))
    independent_variables = pd.DataFrame(columns=genes, index = wells)
    for index, row in df.iterrows():
        prc = row['prc']
        gene = row[gene_column]
        fraction = row['fraction']
        independent_variables.loc[prc,gene]=fraction
    independent_variables = independent_variables.dropna(axis=1, how='all')
    independent_variables = independent_variables.dropna(axis=0, how='all')
    independent_variables['sum'] = independent_variables.sum(axis=1)
    #sums = independent_variables['sum'].unique().tolist()
    #print(sums)
    #independent_variables = independent_variables[(independent_variables['sum'] == 0.0) | (independent_variables['sum'] == 1.0)]
    independent_variables = independent_variables.fillna(0.0)
    independent_variables = independent_variables.drop(columns=[col for col in independent_variables.columns if independent_variables[col].max() < min_frequency])
    independent_variables = independent_variables.drop('sum', axis=1)
    independent_variables.index.name = 'prc'
    loc = '/mnt/data/CellVoyager/20x/tsg101/crispr_screen/all/measurements/iv.csv'
    independent_variables.to_csv(loc, index=True, header=True, mode='w')
    return independent_variables

def check_multicollinearity(x):
    """Checks multicollinearity of the predictors by computing the VIF."""
    vif_data = pd.DataFrame()
    vif_data["Variable"] = x.columns
    vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    return vif_data

def show_residules(model):

    # Get the residuals
    residuals = model.resid

    # Histogram of residuals
    plt.hist(residuals, bins=30)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.show()

    # QQ plot
    sm.qqplot(residuals, fit=True, line='45')
    plt.title('QQ Plot')
    plt.show()

    # Residuals vs. Fitted values
    plt.scatter(model.fittedvalues, residuals)
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Fitted Values')
    plt.axhline(y=0, color='red')
    plt.show()

    # Shapiro-Wilk test for normality
    W, p_value = stats.shapiro(residuals)
    print(f'Shapiro-Wilk Test W-statistic: {W}, p-value: {p_value}')

def fishers_odds(df, threshold=0.5, phenotyp_col='mean_pred'):
    # Binning based on phenotype score (e.g., above 0.8 as high)
    df['high_phenotype'] = df[phenotyp_col] < threshold

    results = []
    mutants = df.columns[:-2]
    mutants = [item for item in mutants if item not in ['count_prc','mean_parasite_area']]
    print(f'fishers df')
    display(df)
    # Perform Fisher's exact test for each mutant
    for mutant in mutants:
        contingency_table = pd.crosstab(df[mutant] > 0, df['high_phenotype'])
        if contingency_table.shape == (2, 2):  # Check for 2x2 shape
            odds_ratio, p_value = fisher_exact(contingency_table)
            results.append((mutant, odds_ratio, p_value))
        else:
            # Optionally handle non-2x2 tables (e.g., append NaN or other placeholders)
            results.append((mutant, float('nan'), float('nan')))
    
    # Convert results to DataFrame for easier handling
    results_df = pd.DataFrame(results, columns=['Mutant', 'OddsRatio', 'PValue'])
    # Remove rows with undefined odds ratios or p-values
    filtered_results_df = results_df.dropna(subset=['OddsRatio', 'PValue'])
    
    pvalues = filtered_results_df['PValue'].values

    # Check if pvalues array is empty
    if len(pvalues) > 0:
        # Apply Benjamini-Hochberg correction
        adjusted_pvalues = multipletests(pvalues, method='fdr_bh')[1]
        # Add adjusted p-values back to the dataframe
        filtered_results_df['AdjustedPValue'] = adjusted_pvalues
        # Filter significant results
        significant_mutants = filtered_results_df[filtered_results_df['AdjustedPValue'] < 0.05]
    else:
        print("No p-values to adjust. Check your data filtering steps.")
        significant_mutants = pd.DataFrame()  # return empty DataFrame in this case
    
    return filtered_results_df

def model_metrics(model):

    # Calculate additional metrics
    rmse = np.sqrt(model.mse_resid)
    mae = np.mean(np.abs(model.resid))
    durbin_w_value = durbin_watson(model.resid)

    # Display the additional metrics
    print("\nAdditional Metrics:")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Durbin-Watson: {durbin_w_value}")

    # Residual Plots
    fig, ax = plt.subplots(2, 2, figsize=(15, 12))

    # Residual vs. Fitted
    ax[0, 0].scatter(model.fittedvalues, model.resid, edgecolors = 'k', facecolors = 'none')
    ax[0, 0].set_title('Residuals vs Fitted')
    ax[0, 0].set_xlabel('Fitted values')
    ax[0, 0].set_ylabel('Residuals')

    # Histogram
    sns.histplot(model.resid, kde=True, ax=ax[0, 1])
    ax[0, 1].set_title('Histogram of Residuals')
    ax[0, 1].set_xlabel('Residuals')

    # QQ Plot
    sm.qqplot(model.resid, fit=True, line='45', ax=ax[1, 0])
    ax[1, 0].set_title('QQ Plot')

    # Scale-Location
    standardized_resid = model.get_influence().resid_studentized_internal
    ax[1, 1].scatter(model.fittedvalues, np.sqrt(np.abs(standardized_resid)), edgecolors = 'k', facecolors = 'none')
    ax[1, 1].set_title('Scale-Location')
    ax[1, 1].set_xlabel('Fitted values')
    ax[1, 1].set_ylabel('$\sqrt{|Standardized Residuals|}$')

    plt.tight_layout()
    plt.show()

def regression_analasys(dv_df,sequencing_loc, min_reads=75, min_wells=2, max_wells=0, model_type = 'mlr', min_cells=100, transform='logit', min_frequency=0.05, gene_column='gene', effect_size_threshold=0.25, fishers=True, clean_regression=False, VIF_threshold=10):
    
    sequencing_df = pd.read_csv(sequencing_loc)
    columns_list = ['c1','c2','c3', 'c15']
    sequencing_df = sequencing_df[~sequencing_df['col'].isin(columns_list)]

    reads_df, stats_dict = process_reads(df=sequencing_df,
                                   min_reads=min_reads,
                                   min_wells=min_wells,
                                   max_wells=max_wells,
                                   gene_column='gene')
    
    display(reads_df)
    
    iv_df = generate_fraction_map(df=reads_df, 
                              gene_column=gene_column, 
                              min_frequency=min_frequency)
    
    display(iv_df)
    
    dv_df = dv_df[dv_df['count_prc']>min_cells]
    display(dv_df)
    merged_df = iv_df.join(dv_df, on='prc', how='inner')
    display(merged_df)
    fisher_df = merged_df.copy()
    
    merged_df.reset_index(inplace=True)
    merged_df[['plate', 'row', 'col']] = merged_df['prc'].str.split('_', expand=True)
    merged_df = merged_df.drop(columns=['prc'])
    merged_df.dropna(inplace=True)
    merged_df = pd.get_dummies(merged_df, columns=['plate', 'row', 'col'], drop_first=True)
    
    y = merged_df['mean_pred']
    
    if model_type == 'mlr':
        merged_df = merged_df.drop(columns=['count_prc'])
        
    elif model_type == 'wls':
        weights = merged_df['count_prc']
    
    elif model_type == 'glm':
        merged_df = merged_df.drop(columns=['count_prc'])
    
    if transform == 'logit':
    # logit transformation
        epsilon = 1e-15
        y = np.log(y + epsilon) - np.log(1 - y + epsilon)
    
    elif transform == 'log':
    # log transformation
        y = np.log10(y+1)
    
    elif transform == 'center':
    # Centering the y around 0
        y_mean = y.mean()
        y = y - y_mean
    
    x = merged_df.drop('mean_pred', axis=1)
    x = x.select_dtypes(include=[np.number])
    #x = sm.add_constant(x)
    x['const'] = 0.0

    if model_type == 'mlr':
        model = sm.OLS(y, x).fit()
        model_metrics(model)

        # Check for Multicollinearity
        vif_data = check_multicollinearity(x.drop('const', axis=1))  # assuming you've added a constant to x
        high_vif_columns = vif_data[vif_data["VIF"] > VIF_threshold]["Variable"].values  # VIF threshold of 10 is common, but this can vary based on context

        print(f"Columns with high VIF: {high_vif_columns}")
        x = x.drop(columns=high_vif_columns)  # dropping columns with high VIF

        if clean_regression:
            # 1. Filter by standardized residuals
            std_resid = model.get_influence().resid_studentized_internal
            outliers_resid = np.where(np.abs(std_resid) > 3)[0]

            # 2. Filter by leverage
            influence = model.get_influence().hat_matrix_diag
            outliers_lev = np.where(influence > 2*(x.shape[1])/len(y))[0]

            # 3. Filter by Cook's distance
            (c, p) = model.get_influence().cooks_distance
            outliers_cooks = np.where(c > 4/(len(y)-x.shape[1]-1))[0]

            # Combine all identified outliers
            outliers = reduce(np.union1d, (outliers_resid, outliers_lev, outliers_cooks))

            # Filter out outliers
            x_clean = x.drop(x.index[outliers])
            y_clean = y.drop(y.index[outliers])

            # Re-run the regression with the filtered data
            model = sm.OLS(y_clean, x_clean).fit()
            model_metrics(model)
    
    elif model_type == 'wls':
        model = sm.WLS(y, x, weights=weights).fit()
    
    elif model_type == 'glm':
        model = sm.GLM(y, x, family=sm.families.Binomial()).fit()

    print(model.summary())
    
    results_summary = model.summary()
        
    results_as_html = results_summary.tables[1].as_html()
    results_df = pd.read_html(results_as_html, header=0, index_col=0)[0]
    results_df = results_df.sort_values(by='coef', ascending=False)
    
    if model_type == 'mlr':
        results_df['p'] = results_df['P>|t|']
    elif model_type == 'wls':
        results_df['p'] = results_df['P>|t|']
    elif model_type == 'glm':    
        results_df['p'] = results_df['P>|z|']
    
    results_df['type'] = 1
    results_df.loc[results_df['p'] == 0.000, 'p'] = 0.005
    results_df['-log10(p)'] = -np.log10(results_df['p'])
    
    display(results_df)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))

    # Plot histogram on ax1
    sns.histplot(data=y, kde=False, element="step", ax=ax1, color='teal')
    ax1.set_xlim([0, 1])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Prepare data for volcano plot on ax2
    results_df['-log10(p)'] = -np.log10(results_df['p'])

    # Assuming the 'type' column is in the merged_df
    sc = ax2.scatter(results_df['coef'], results_df['-log10(p)'], c=results_df['type'], cmap='coolwarm')
    ax2.set_title('Volcano Plot')
    ax2.set_xlabel('Coefficient')
    ax2.set_ylabel('-log10(P-value)')

    # Adjust colorbar
    cbar = plt.colorbar(sc, ax=ax2, ticks=[-1, 1])
    cbar.set_label('Sign of Coefficient')
    cbar.set_ticklabels(['-ve', '+ve'])

    # Add text for specified points
    for idx, row in results_df.iterrows():
        if row['p'] < 0.05 and row['coef'] > effect_size_threshold:
            ax2.text(row['coef'], -np.log10(row['p']), idx, fontsize=8, ha='center', va='bottom', color='black')

    ax2.axhline(y=-np.log10(0.05), color='gray', linestyle='--')

    plt.show()
    
    #if model_type == 'mlr':
    #    show_residules(model)
    
    if fishers:
        threshold = 2*effect_size_threshold
        fishers_odds(df=fisher_df, threshold=threshold, phenotyp_col='mean_pred')
    
    return

def reg_v_plot(df):
    df['-log10(p)'] = -np.log10(df['p'])

    # Create the volcano plot
    plt.figure(figsize=(40, 30))
    sc = plt.scatter(df['effect'], df['-log10(p)'], c=np.sign(df['effect']), cmap='coolwarm')
    plt.title('Volcano Plot', fontsize=12)
    plt.xlabel('Coefficient', fontsize=12)
    plt.ylabel('-log10(P-value)', fontsize=12)

    # Add text for specified points
    for idx, row in df.iterrows():
        if row['p'] < 0.05:# and abs(row['effect']) > 0.1:
            plt.text(row['effect'], -np.log10(row['p']), idx, fontsize=12, ha='center', va='bottom', color='black')

    plt.axhline(y=-np.log10(0.05), color='gray', linestyle='--')  # line for p=0.05
    plt.show()

def analyze_data_reg(sequencing_loc, dv_loc, agg_type = 'mean', min_cell_count=50, min_reads=100, min_wells=2, max_wells=1000, remove_outlier_genes=False, refine_model=False,by_plate=False, threshold=0.5):
    
    def qstring_to_float(qstr):
        number = int(qstr[1:])  # Remove the "q" and convert the rest to an integer
        return number / 100.0
    
    columns_list = ['c1', 'c2', 'c3', 'c15']
    plate_list = ['p1','p2','p3','p4']
    
    dv_df = pd.read_csv(dv_loc)#, index_col='prc')    
    
    if agg_type.startswith('q'):
        val = qstring_to_float(agg_type)
        agg_type = lambda x: x.quantile(val)
    
    # Aggregating for mean prediction, total count and count of values > 0.95
    dv_df = dv_df.groupby('prc').agg(
        pred=('pred', agg_type),
        count_prc=('prc', 'size'),
        #count_above_95=('pred', lambda x: (x > 0.95).sum()),
        mean_parasite_area=('parasite_area', 'mean')
    )
    
    dv_df = dv_df[dv_df['count_prc'] >= min_cell_count]
    sequencing_df = pd.read_csv(sequencing_loc)

    reads_df, stats_dict = process_reads(df=sequencing_df,
                                         min_reads=min_reads,
                                         min_wells=min_wells,
                                         max_wells=max_wells,
                                         gene_column='gene',
                                         remove_outliers=remove_outlier_genes)

    iv_df = generate_fraction_map(df=reads_df, 
                                  gene_column='grna', 
                                  min_frequency=0.0)

    # Melt the iv_df to long format
    df_long = iv_df.reset_index().melt(id_vars=["prc"], 
                                       value_vars=iv_df.columns, 
                                       var_name="gene_grna", 
                                       value_name="value")

    # Extract gene and grna info from gene_grna column
    df_long["gene"] = df_long["gene_grna"].str.split("_").str[1]
    df_long["grna"] = df_long["gene_grna"].str.split("_").str[2]

    merged_df = df_long.merge(dv_df, left_on='prc', right_index=True)
    merged_df = merged_df[merged_df['value'] > 0]
    merged_df['plate'] = merged_df['prc'].str.split('_').str[0]
    merged_df['row'] = merged_df['prc'].str.split('_').str[1]
    merged_df['column'] = merged_df['prc'].str.split('_').str[2]
    
    merged_df = merged_df[~merged_df['column'].isin(columns_list)]
    merged_df = merged_df[merged_df['plate'].isin(plate_list)]
    
    # Printing the unique values in 'col' and 'plate' columns
    print("Unique values in col:", merged_df['column'].unique())
    print("Unique values in plate:", merged_df['plate'].unique())
    
    if not by_plate:
        if fishers:
            fishers_odds(df=merged_df, threshold=threshold, phenotyp_col='pred')
    
    if by_plate:
        merged_df2 = merged_df.copy()
        for plate in merged_df2['plate'].unique():
            merged_df = merged_df2[merged_df2['plate'] == plate]
            print(f'merged_df: {len(merged_df)}, plate: {plate}')
            if len(merged_df) <100:
                break
            display(merged_df)

            model = smf.ols("pred ~ gene + grna + gene:grna + plate + row + column", merged_df).fit()
            #model = smf.ols("pred ~ infection_time + gene + grna + gene:grna + plate + row + column", merged_df).fit()
            
            # Display model metrics and summary
            model_metrics(model)
            #print(model.summary())

            if refine_model:
                # Filter outliers
                std_resid = model.get_influence().resid_studentized_internal
                outliers_resid = np.where(np.abs(std_resid) > 3)[0]
                (c, p) = model.get_influence().cooks_distance
                outliers_cooks = np.where(c > 4/(len(merged_df)-merged_df.shape[1]-1))[0]
                outliers = reduce(np.union1d, (outliers_resid, outliers_cooks))
                merged_df_filtered = merged_df.drop(merged_df.index[outliers])

                display(merged_df_filtered)

                # Refit the model with filtered data
                model = smf.ols("pred ~ gene + grna + gene:grna + row + column", merged_df_filtered).fit()
                print("Number of outliers detected by standardized residuals:", len(outliers_resid))
                print("Number of outliers detected by Cook's distance:", len(outliers_cooks))

                model_metrics(model)

            # Extract interaction coefficients and determine the maximum effect size
            interaction_coeffs = {key: val for key, val in model.params.items() if "gene[T." in key and ":grna[T." in key}
            interaction_pvalues = {key: val for key, val in model.pvalues.items() if "gene[T." in key and ":grna[T." in key}

            max_effects = {}
            max_effects_pvalues = {}
            for key, val in interaction_coeffs.items():
                gene_name = key.split(":")[0].replace("gene[T.", "").replace("]", "")
                if gene_name not in max_effects or abs(max_effects[gene_name]) < abs(val):
                    max_effects[gene_name] = val
                    max_effects_pvalues[gene_name] = interaction_pvalues[key]

            for key in max_effects:
                print(f"Key: {key}: {max_effects[key]}, p:{max_effects_pvalues[key]}")

            df = pd.DataFrame([max_effects, max_effects_pvalues])
            df = df.transpose()
            df = df.rename(columns={df.columns[0]: 'effect', df.columns[1]: 'p'})
            df = df.sort_values(by=['effect', 'p'], ascending=[False, True])

            reg_v_plot(df)
            
            if fishers:
                fishers_odds(df=merged_df, threshold=threshold, phenotyp_col='pred')
    else:
        display(merged_df)

        model = smf.ols("pred ~ gene + grna + gene:grna + plate + row + column", merged_df).fit()

        # Display model metrics and summary
        model_metrics(model)

        if refine_model:
            # Filter outliers
            std_resid = model.get_influence().resid_studentized_internal
            outliers_resid = np.where(np.abs(std_resid) > 3)[0]
            (c, p) = model.get_influence().cooks_distance
            outliers_cooks = np.where(c > 4/(len(merged_df)-merged_df.shape[1]-1))[0]
            outliers = reduce(np.union1d, (outliers_resid, outliers_cooks))
            merged_df_filtered = merged_df.drop(merged_df.index[outliers])

            display(merged_df_filtered)

            # Refit the model with filtered data
            model = smf.ols("pred ~ gene + grna + gene:grna + plate + row + column", merged_df_filtered).fit()
            print("Number of outliers detected by standardized residuals:", len(outliers_resid))
            print("Number of outliers detected by Cook's distance:", len(outliers_cooks))

            model_metrics(model)

        # Extract interaction coefficients and determine the maximum effect size
        interaction_coeffs = {key: val for key, val in model.params.items() if "gene[T." in key and ":grna[T." in key}
        interaction_pvalues = {key: val for key, val in model.pvalues.items() if "gene[T." in key and ":grna[T." in key}

        max_effects = {}
        max_effects_pvalues = {}
        for key, val in interaction_coeffs.items():
            gene_name = key.split(":")[0].replace("gene[T.", "").replace("]", "")
            if gene_name not in max_effects or abs(max_effects[gene_name]) < abs(val):
                max_effects[gene_name] = val
                max_effects_pvalues[gene_name] = interaction_pvalues[key]

        for key in max_effects:
            print(f"Key: {key}: {max_effects[key]}, p:{max_effects_pvalues[key]}")

        df = pd.DataFrame([max_effects, max_effects_pvalues])
        df = df.transpose()
        df = df.rename(columns={df.columns[0]: 'effect', df.columns[1]: 'p'})
        df = df.sort_values(by=['effect', 'p'], ascending=[False, True])

        reg_v_plot(df)
        
        if fishers:
            fishers_odds(df=merged_df, threshold=threshold, phenotyp_col='pred')

    return max_effects, max_effects_pvalues, model, df

def generate_plate_heatmap(df, plate_number, variable, grouping, min_max):
    df = df.copy()  # Work on a copy to avoid SettingWithCopyWarning
    df['plate'], df['row'], df['col'] = zip(*df['prc'].str.split('_'))
    
    # Filtering the dataframe based on the plate_number
    df = df[df['plate'] == plate_number].copy()  # Create another copy after filtering
    
    # Ensure proper ordering
    row_order = [f'r{i}' for i in range(1, 17)]
    col_order = [f'c{i}' for i in range(1, 28)]  # Exclude c15 as per your earlier code
    
    df['row'] = pd.Categorical(df['row'], categories=row_order, ordered=True)
    df['col'] = pd.Categorical(df['col'], categories=col_order, ordered=True)
    
    # Explicitly set observed=True to avoid FutureWarning
    grouped = df.groupby(['row', 'col'], observed=True)  
    
    if grouping == 'mean':
        plate = grouped[variable].mean().reset_index()
    elif grouping == 'sum':
        plate = grouped[variable].sum().reset_index()
    elif grouping == 'count':
        plate = grouped[variable].count().reset_index()
    else:
        raise ValueError(f"Unsupported grouping: {grouping}")
        
    plate_map = pd.pivot_table(plate, values=variable, index='row', columns='col').fillna(0)
    
    if min_max == 'all':
        min_max = [plate_map.min().min(), plate_map.max().max()]
    elif min_max == 'allq':
        min_max = np.quantile(plate_map.values, [0.2, 0.98])
    elif min_max == 'plate':
        min_max = [plate_map.min().min(), plate_map.max().max()]
        
    return plate_map, min_max

def plot_plates(df, variable, grouping, min_max, cmap):
    plates = df['prc'].str.split('_', expand=True)[0].unique()
    n_rows, n_cols = (len(plates) + 3) // 4, 4
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(40, 5 * n_rows))
    ax = ax.flatten()

    for index, plate in enumerate(plates):
        plate_map, min_max_values = generate_plate_heatmap(df, plate, variable, grouping, min_max)
        sns.heatmap(plate_map, cmap=cmap, vmin=0, vmax=2, ax=ax[index])
        ax[index].set_title(plate)
        
    for i in range(len(plates), n_rows * n_cols):
        fig.delaxes(ax[i])
    
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    plt.show()
    return

def generate_dependent_variable(df, dv_loc, pc_min=0.95, nc_max=0.05, agg_type='mean'):
    
    def qstring_to_float(qstr):
        number = int(qstr[1:])  # Remove the "q" and convert the rest to an integer
        return number / 100.0
    
    print("Unique values in plate:", df['plate'].unique())
    dv_cell_loc = f'{dv_loc}/dv_cell.csv'
    dv_well_loc = f'{dv_loc}/dv_well.csv'
    
    df['pred'] = 1-df['pred'] #if you swiched pc and nc
    df = df[(df['pred'] <= nc_max) | (df['pred'] >= pc_min)]
    
    if 'prc' not in df.columns:
        df['prc'] = df['plate'] + '_' + df['row'] + '_' + df['col']
    
    if agg_type.startswith('q'):
        val = qstring_to_float(agg_type)
        agg_type = lambda x: x.quantile(val)
    
    # Aggregating for mean prediction and total count
    df_grouped = df.groupby('prc').agg(
        pred=('pred', agg_type),
        recruitment=('recruitment', agg_type),
        count_prc=('prc', 'size'),
        #count_above_95=('pred', lambda x: (x > 0.95).sum()),
        mean_parasite_area=('parasite_area', 'mean')
    )
    
    df_cell = df[['prc', 'pred', 'parasite_area', 'recruitment']]
    
    df_cell.to_csv(dv_cell_loc, index=True, header=True, mode='w')
    df_grouped.to_csv(dv_well_loc, index=True, header=True, mode='w')  # Changed from loc to dv_loc
    display(df)
    plot_histograms_and_stats(df)
    df_grouped = df_grouped.sort_values(by='count_prc', ascending=True)
    display(df_grouped)
    print('pred')
    plot_plates(df=df_cell, variable='pred', grouping='mean', min_max='allq', cmap='viridis')
    print('recruitment')
    plot_plates(df=df_cell, variable='recruitment', grouping='mean', min_max='allq', cmap='viridis')
    
    return df_grouped

def lasso_reg(merged_df, alpha_value=0.01, reg_type='lasso'):
    # Separate predictors and response
    X = merged_df[['gene', 'grna', 'plate', 'row', 'column']]
    y = merged_df['pred']

    # One-hot encode the categorical predictors
    encoder = OneHotEncoder(drop='first')  # drop one category to avoid the dummy variable trap
    X_encoded = encoder.fit_transform(X).toarray()
    feature_names = encoder.get_feature_names_out(input_features=X.columns)
    
    if reg_type == 'ridge':
        # Fit ridge regression
        ridge = Ridge(alpha=alpha_value)
        ridge.fit(X_encoded, y)
        coefficients = ridge.coef_
        coeff_dict = dict(zip(feature_names, ridge.coef_))
        
    if reg_type == 'lasso':
        # Fit Lasso regression
        lasso = Lasso(alpha=alpha_value)
        lasso.fit(X_encoded, y)
        coefficients = lasso.coef_
        coeff_dict = dict(zip(feature_names, lasso.coef_))
    coeff_df = pd.DataFrame(list(coeff_dict.items()), columns=['Feature', 'Coefficient'])
    return coeff_df

def MLR(merged_df, refine_model):
    #model = smf.ols("pred ~ gene + grna + gene:grna + plate + row + column", merged_df).fit()
    model = smf.ols("pred ~ gene:grna + plate + row + column", merged_df).fit()
    # Display model metrics and summary
    model_metrics(model)

    if refine_model:
        # Filter outliers
        std_resid = model.get_influence().resid_studentized_internal
        outliers_resid = np.where(np.abs(std_resid) > 3)[0]
        (c, p) = model.get_influence().cooks_distance
        outliers_cooks = np.where(c > 4/(len(merged_df)-merged_df.shape[1]-1))[0]
        outliers = reduce(np.union1d, (outliers_resid, outliers_cooks))
        merged_df_filtered = merged_df.drop(merged_df.index[outliers])

        display(merged_df_filtered)

        # Refit the model with filtered data
        model = smf.ols("pred ~ gene + grna + gene:grna + row + column", merged_df_filtered).fit()
        print("Number of outliers detected by standardized residuals:", len(outliers_resid))
        print("Number of outliers detected by Cook's distance:", len(outliers_cooks))

        model_metrics(model)
        print(model.summary())

    # Extract interaction coefficients and determine the maximum effect size
    interaction_coeffs = {key: val for key, val in model.params.items() if "gene[T." in key and ":grna[T." in key}
    interaction_pvalues = {key: val for key, val in model.pvalues.items() if "gene[T." in key and ":grna[T." in key}

    max_effects = {}
    max_effects_pvalues = {}
    for key, val in interaction_coeffs.items():
        gene_name = key.split(":")[0].replace("gene[T.", "").replace("]", "")
        if gene_name not in max_effects or abs(max_effects[gene_name]) < abs(val):
            max_effects[gene_name] = val
            max_effects_pvalues[gene_name] = interaction_pvalues[key]

    for key in max_effects:
        print(f"Key: {key}: {max_effects[key]}, p:{max_effects_pvalues[key]}")

    df = pd.DataFrame([max_effects, max_effects_pvalues])
    df = df.transpose()
    df = df.rename(columns={df.columns[0]: 'effect', df.columns[1]: 'p'})
    df = df.sort_values(by=['effect', 'p'], ascending=[False, True])

    reg_v_plot(df)
    
    return max_effects, max_effects_pvalues, model, df

def analyze_data_reg(sequencing_loc, dv_loc, agg_type = 'mean', dv_col='pred', transform=None, min_cell_count=50, min_reads=100, min_wells=2, max_wells=1000, min_frequency=0.0,remove_outlier_genes=False, refine_model=False,by_plate=False, regression_type='mlr', alpha_value=0.01, fishers=False, fisher_threshold=0.9):
    
    def qstring_to_float(qstr):
        number = int(qstr[1:])  # Remove the "q" and convert the rest to an integer
        return number / 100.0
    
    columns_list = ['c1', 'c2', 'c3']
    plate_list = ['p1','p3','p4']
    
    dv_df = pd.read_csv(dv_loc)#, index_col='prc')    
    
    if agg_type.startswith('q'):
        val = qstring_to_float(agg_type)
        agg_type = lambda x: x.quantile(val)
    
    # Aggregating for mean prediction, total count and count of values > 0.95
    dv_df = dv_df.groupby('prc').agg(
        pred=(dv_col, agg_type),
        count_prc=('prc', 'size'),
        mean_parasite_area=('parasite_area', 'mean')
    )
    
    dv_df = dv_df[dv_df['count_prc'] >= min_cell_count]
    sequencing_df = pd.read_csv(sequencing_loc)
    

    reads_df, stats_dict = process_reads(df=sequencing_df,
                                         min_reads=min_reads,
                                         min_wells=min_wells,
                                         max_wells=max_wells,
                                         gene_column='gene',
                                         remove_outliers=remove_outlier_genes)
    
    reads_df['value'] = reads_df['count']/reads_df['well_read_sum']
    reads_df['gene_grna'] = reads_df['gene']+'_'+reads_df['grna']
    
    display(reads_df)
    
    df_long = reads_df
    
    df_long = df_long[df_long['value'] > min_frequency] # removes gRNAs under a certain proportion
    #df_long = df_long[df_long['value']<1.0] # removes gRNAs in wells with only one gRNA

    # Extract gene and grna info from gene_grna column
    df_long["gene"] = df_long["grna"].str.split("_").str[1]
    df_long["grna"] = df_long["grna"].str.split("_").str[2]
    
    agg_df = df_long.groupby('prc')['count'].sum().reset_index()
    agg_df = agg_df.rename(columns={'count': 'count_sum'})
    df_long = pd.merge(df_long, agg_df, on='prc', how='left')
    df_long['value'] = df_long['count']/df_long['count_sum']
    
    merged_df = df_long.merge(dv_df, left_on='prc', right_index=True)
    merged_df = merged_df[merged_df['value'] > 0]
    merged_df['plate'] = merged_df['prc'].str.split('_').str[0]
    merged_df['row'] = merged_df['prc'].str.split('_').str[1]
    merged_df['column'] = merged_df['prc'].str.split('_').str[2]
    
    merged_df = merged_df[~merged_df['column'].isin(columns_list)]
    merged_df = merged_df[merged_df['plate'].isin(plate_list)]
    
    if transform == 'log':
        merged_df['pred'] = np.log(merged_df['pred'] + 1e-10)
    
    # Printing the unique values in 'col' and 'plate' columns
    print("Unique values in col:", merged_df['column'].unique())
    print("Unique values in plate:", merged_df['plate'].unique())
    display(merged_df)

    if fishers:
        iv_df = generate_fraction_map(df=reads_df, 
                                      gene_column='grna', 
                                      min_frequency=min_frequency)

        fishers_df = iv_df.join(dv_df, on='prc', how='inner')
        
        significant_mutants = fishers_odds(df=fishers_df, threshold=fisher_threshold, phenotyp_col='pred')
        significant_mutants = significant_mutants.sort_values(by='OddsRatio', ascending=False) 
        display(significant_mutants)
        
    if regression_type == 'mlr':
        if by_plate:
            merged_df2 = merged_df.copy()
            for plate in merged_df2['plate'].unique():
                merged_df = merged_df2[merged_df2['plate'] == plate]
                print(f'merged_df: {len(merged_df)}, plate: {plate}')
                if len(merged_df) <100:
                    break
                
                max_effects, max_effects_pvalues, model, df = MLR(merged_df, refine_model)
        else:
            
            max_effects, max_effects_pvalues, model, df = MLR(merged_df, refine_model)
        return max_effects, max_effects_pvalues, model, df
            
    if regression_type == 'ridge' or regression_type == 'lasso':
        coeffs = lasso_reg(merged_df, alpha_value=alpha_value, reg_type=regression_type)
        return coeffs
    
    if regression_type == 'mixed':
        model = smf.mixedlm("pred ~ gene_grna - 1", merged_df, groups=merged_df["plate"], re_formula="~1")
        result = model.fit(method="bfgs")
        print(result.summary())

        # Print AIC and BIC
        print("AIC:", result.aic)
        print("BIC:", result.bic)
    

        results_df = pd.DataFrame({
            'effect': result.params,
            'Standard Error': result.bse,
            'T-Value': result.tvalues,
            'p': result.pvalues
        })
        
        display(results_df)
        reg_v_plot(df=results_df)
        
        std_resid = result.resid

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Histogram of Residuals
        axes[0].hist(std_resid, bins=50, edgecolor='k')
        axes[0].set_xlabel('Residuals')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Histogram of Residuals')

        # Boxplot of Residuals
        axes[1].boxplot(std_resid)
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Boxplot of Residuals')

        # QQ Plot
        sm.qqplot(std_resid, line='45', ax=axes[2])
        axes[2].set_title('QQ Plot')

        # Show plots
        plt.tight_layout()
        plt.show()
        
        return result
