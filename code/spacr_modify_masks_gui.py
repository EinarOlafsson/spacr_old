import subprocess
import sys
import os
import json
import shutil
import platform

def get_paths(env_name):
    conda_executable = "conda.exe" if sys.platform == "win32" else "conda"
    python_executable = "python.exe" if sys.platform == "win32" else "python"
    pip_executable = "pip.exe" if sys.platform == "win32" else "pip"

    conda_path = shutil.which(conda_executable)
    if not conda_path:
        print("Conda is not found in the system PATH")
        return None, None, None, None

    conda_dir = os.path.dirname(os.path.dirname(conda_path))
    env_path = os.path.join(conda_dir, 'envs', env_name)
    python_path = os.path.join(env_path, 'bin', python_executable)
    pip_path = os.path.join(env_path, 'bin', pip_executable)

    if sys.platform == "win32":
        python_path = python_path.replace('bin/', 'Scripts/')
        pip_path = pip_path.replace('bin/', 'Scripts/')

    return conda_path, python_path, pip_path, env_path

# create new kernel
def add_kernel(env_name, display_name):
    _, _, python_path, env_path = get_paths(env_name)

    if not python_path or not env_path:
        print(f"Failed to locate the environment or Python executable for '{env_name}'")
        return

    kernel_spec_path = os.path.join(env_path, ".local", "share", "jupyter", "kernels", env_name)

    kernel_spec = {
        "argv": [python_path, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": display_name,
        "language": "python"
    }

    os.makedirs(kernel_spec_path, exist_ok=True)
    with open(os.path.join(kernel_spec_path, "kernel.json"), "w") as f:
        json.dump(kernel_spec, f)

    print(f"Kernel for '{env_name}' added successfully.")

def create_environment(env_name):
    print(f"Creating environment {env_name}...")
    subprocess.run(["conda", "create", "-n", env_name, "python=3.9", "-y"])

# Install dependencies in a specified kernel environment.
def install_dependencies_in_kernel(dependencies, env_name):
    
    # Ensure Python version is 3.9 or above
    if sys.version_info < (3, 9):
        raise EnvironmentError("Python version 3.9 or higher is required.")
    
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
    
    for package in dependencies:
        print(f"Installing {package}")
        subprocess.run([conda_PATH, "install", "-n", env_name, package, "-y"])
    
    # Install additional packages
    subprocess.run([pip_PATH, "install", "opencv-python-headless"])
    subprocess.run([pip_PATH, "install", "PyQt5"])
    subprocess.run([pip_PATH, "install", "screeninfo"])

    print("Dependencies installation complete.")

env_name = "spacr_modify_masks_gui"

conda_PATH, python_PATH, pip_PATH, env_PATH = get_paths(env_name)

dependencies = ["matplotlib==3.7.1", "scipy", "pillow", "scikit-image", "ipykernel", "requests", "h5py", "pyzmq"]

if not os.path.exists(env_PATH):

    print(f'System type: {sys.platform}')
    print(f'PATH to conda: {conda_PATH}')
    print(f'PATH to python: {python_PATH}')
    print(f'PATH to pip: {pip_PATH}')
    print(f'PATH to new environment: {env_PATH}')

    create_environment(env_name)
    install_dependencies_in_kernel(dependencies, env_name)
    add_kernel(env_name, env_name)
    print(f"Environment '{env_name}' created and added as a Jupyter kernel.")
    print(f"Refresh the page, set {env_name} as the kernel and run cell again")
    #sys.exit()
    SystemExit()

################################################################################################################################################################################

import os, time, warnings, matplotlib, random
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from skimage.measure import label
import skimage.morphology as morph
from skimage import feature, morphology
from skimage.morphology import disk
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from skimage.draw import polygon

from scipy.ndimage import binary_dilation, binary_fill_holes

from PIL import Image
import imageio.v2 as imageio
from collections import deque
import matplotlib as mpl
from matplotlib.widgets import CheckButtons, Button, Slider, TextBox
from screeninfo import get_monitors

warnings.filterwarnings('ignore', category=RuntimeWarning, message='QCoreApplication::exec: The event loop is already running')
Image.MAX_IMAGE_PIXELS = None

plt.style.use('ggplot')

# Function to determine an appropriate font size based on screen resolution
def get_font_size():
    try:
        monitor = get_monitors()[0]
        width = monitor.width

        # Define font size based on screen width (you can adjust the conditions and sizes)
        if width <= 1280:  # Smaller screens
            return 8
        elif width <= 1920:  # Medium screens
            return 10
        else:  # Larger screens
            return 12
		
    except Exception as e:
        print(f"Error occurred: {e}")
        return 10

def get_line_width():
    try:
        monitor = get_monitors()[0]
        width = monitor.width

        # Define font size based on screen width (you can adjust the conditions and sizes)
        if width <= 1280:  # Smaller screens
            return 1
        elif width <= 1920:  # Medium screens
            return 2
        else:  # Larger screens
            return 3
		
    except Exception as e:
        print(f"Error occurred: {e}")
        return 2 
	    
# Style paramiters
plt.rcParams['axes.grid'] = False           		# Disable grid
plt.rcParams['axes.facecolor'] = 'white'    		# Change axis face color
plt.rcParams['lines.linewidth'] = get_line_width()     	# Change line width
plt.rcParams['font.size'] = get_font_size() 		# Font size
plt.rcParams['axes.labelsize'] = get_font_size() + 2   	# Axis label size
plt.rcParams['axes.titlesize'] = get_font_size() + 4   	# Axis title size
plt.rcParams['axes.spines.top'] = True      		# Hide the top spine
plt.rcParams['axes.spines.right'] = True    		# Hide the right spine

# To display all style paramiters: run the following in jupyter
#%matplotlib inline
#import matplotlib.pyplot as plt

## Set the ggplot style
#plt.style.use('ggplot')

## Display all style parameters for ggplot
#for param, value in plt.rcParams.items():
#    print(f"{param}: {value}")

# Function to normalize the image
def normalize_to_dtype(array, lower_quantile, upper_quantile):
    if len(array.shape) == 2:
        array = np.expand_dims(array, axis=-1)

    num_channels = array.shape[-1]
    new_stack = np.empty_like(array)

    for channel in range(num_channels):
        img = array[..., channel]
        non_zero_img = img[img > 0]

        if non_zero_img.size > 0:
            img_min = np.percentile(non_zero_img, lower_quantile)
            img_max = np.percentile(non_zero_img, upper_quantile)
        else:
            img_min, img_max = img.min(), img.max()

        new_stack[..., channel] = rescale_intensity(img, in_range=(img_min, img_max), out_range='dtype')

    if new_stack.shape[-1] == 1:
        new_stack = np.squeeze(new_stack, axis=-1)

    return new_stack
    
# Edge detection and overlay
def overlay_edges(mask, normalized_image):
    global displayed_image  # Make sure this is necessary for your function logic

    # Detect edges in the mask
    edges = feature.canny(mask.astype(float), sigma=1)

    # If normalized_image is in the range [0, 255], normalize it to [0, 1]
    if normalized_image.max() > 1:
        normalized_image = normalized_image / 255.0

    # Create an RGB version of the image for overlay
    rgb_image = np.repeat(normalized_image[:, :, np.newaxis], 3, axis=2)

    # Overlay red color on the edges
    rgb_image[edges, 0] = 1  # Red channel (assuming normalized_image is now in [0, 1] range)
    rgb_image[edges, 1] = 0  # Green channel
    rgb_image[edges, 2] = 0  # Blue channel

    return rgb_image

def find_nearest_nonzero_pixel(mask, seed_point):
    y, x = seed_point
    non_zero_coords = np.argwhere(mask > 0)  # Find all non-zero pixels

    if len(non_zero_coords) == 0:
        return 0  # No non-zero pixels in the mask

    # Calculate distances to the clicked point
    distances = np.sqrt((non_zero_coords[:, 0] - y) ** 2 + (non_zero_coords[:, 1] - x) ** 2)
    nearest_pixel_index = np.argmin(distances)
    nearest_pixel_value = mask[tuple(non_zero_coords[nearest_pixel_index])]
    
    if nearest_pixel_value == 0:
        nearest_pixel_value = 255  # Change to 255 if the nearest pixel value is 0

    return nearest_pixel_value

# Magic Wand function
def magic_wand(image, mask, seed_point, intensity_tolerance=25, max_pixels=1000, remove=False):
    x, y = seed_point
    initial_value = np.float32(image[y, x])
    to_check = deque([(x, y)])
    checked = set()
    fill_value = 255 if mask.sum() == 0 else find_nearest_nonzero_pixel(mask, seed_point)

    if remove:
        fill_value = 0

    while to_check and len(checked) < max_pixels:
        x, y = to_check.popleft()
        if (x, y) in checked or not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            continue

        checked.add((x, y))

        current_value = np.float32(image[y, x])
        if abs(current_value - initial_value) <= intensity_tolerance:
            mask[y, x] = fill_value

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and (nx, ny) not in checked:
                    next_value = np.float32(image[ny, nx])
                    if abs(next_value - initial_value) <= intensity_tolerance:
                        to_check.append((nx, ny))

    return mask

def clear_freehand_lines():
    global freehand_lines
    for line in freehand_lines:
        line.remove()
    freehand_lines = []
    fig.canvas.draw_idle()

# Mouse click event handler
def on_click(event):
    global save_clicked, mask, slider_itol, slider_mpixels, slider_radius, mode_magic_wand, mode_remove_object
    save_clicked = False
    if fig.canvas.toolbar.mode != '':
        return
    if event.xdata is not None and event.ydata is not None and event.inaxes == ax:
        x, y = int(event.xdata), int(event.ydata)
        intensity_tolerance = slider_itol.val
        max_pixels = slider_mpixels.val
        radius = int(slider_radius.val)

        if mode_remove_object and event.xdata is not None and event.ydata is not None:
            
            # Relabel the mask
            binary_mask = mask > 0
            labeled_mask = label(binary_mask)
            mask = labeled_mask

            if mask[y, x] != 0:
                mask[mask == mask[y, x]] = 0  # Set all pixels of the clicked object to 0
                overlay.set_data(mask)
                clear_freehand_lines()  # Function to clear freehand lines
                
                # Update displayed image without the edges
                displayed_image_with_edges = overlay_edges(mask, normalized_image)
                displayed_image.set_data(displayed_image_with_edges)

                fig.canvas.draw()

        elif mode_magic_wand:
            if event.button == 1:  # Left mouse button
                mask = magic_wand(image, mask, (x, y), slider_itol.val, slider_mpixels.val)
            elif event.button == 3:  # Right mouse button
                mask = magic_wand(image, mask, (x, y), slider_itol.val, slider_mpixels.val, remove=True)

            overlay.set_data(mask)
            overlay.set_cmap(random_cmap)
            fig.canvas.draw()
        
        elif not (mode_freehand or mode_magic_wand or mode_remove_object):
            # Radius selection mode
            y_min, y_max = max(y - radius, 0), min(y + radius + 1, mask.shape[0])
            x_min, x_max = max(x - radius, 0), min(x + radius + 1, mask.shape[1])
            if event.button == 1:  # Left mouse button
                mask[y_min:y_max, x_min:x_max] = 255  # Set pixels to 255
            elif event.button == 3:  # Right mouse button
                mask[y_min:y_max, x_min:x_max] = 0  # Set pixels to 0

            overlay.set_data(mask)
            overlay.set_cmap(random_cmap)
            fig.canvas.draw()

# Function to remove small objects
def remove_small_objects(event):
    global mask, slider_min_size, random_cmap, overlay, normalized_image
    min_size = slider_min_size.val
    mask = morph.remove_small_objects(mask > 0, min_size)
    
    overlay.remove()
    overlay = ax.imshow(mask, cmap=random_cmap, alpha=0.5)
    
    # Update the displayed image with red outlines
    #normalized_image = normalize_to_dtype(image, lower_q, upper_q)
    displayed_image_with_edges = overlay_edges(mask, normalized_image)
    displayed_image.set_data(displayed_image_with_edges)
    
    fig.canvas.draw_idle()

## Function to relabel objects and add red outlines
def relabel_objects(event):
    global mask, overlay, fig, ax, random_cmap, displayed_image, normalized_image

    # Relabel the mask
    binary_mask = mask > 0
    labeled_mask = label(binary_mask)
    mask = labeled_mask

    # Regenerate the colormap for overlay
    n_labels = np.max(mask)
    
    # Ensure a unique color for each label
    random_colors = np.random.rand(n_labels + 1, 4)  # n_labels + 1 for background color
    random_colors[0, :] = [0, 0, 0, 0]  # Background color (label 0)
    random_cmap = mpl.colors.ListedColormap(random_colors)

    # Remove the old overlay and create a new one
    overlay.remove()
    overlay = ax.imshow(mask, cmap=random_cmap, alpha=0.5)

    # Update the displayed image with red outlines
    displayed_image_with_edges = overlay_edges(mask, normalized_image)
    displayed_image.set_data(displayed_image_with_edges)

    # Redraw the figure
    fig.canvas.draw_idle()

# Function to fill holes in objects and add red outlines
def fill_holes(event):
    global mask, overlay, fig, ax, random_cmap, displayed_image, normalized_image

    # Fill holes in the binary mask
    binary_mask = mask > 0
    filled_mask = binary_fill_holes(binary_mask)

    # Update the original mask
    mask = filled_mask.astype(mask.dtype) * 255
    
    # Relabel the mask
    binary_mask = mask > 0
    labeled_mask = label(binary_mask)
    mask = labeled_mask
    
    # Ensure a unique color for each label
    n_labels = np.max(mask)
    random_colors = np.random.rand(n_labels + 1, 4)  # n_labels + 1 for background color
    random_colors[0, :] = [0, 0, 0, 0]  # Background color (label 0)
    random_cmap = mpl.colors.ListedColormap(random_colors)
    
    # Remove the old overlay and create a new one
    overlay.remove()
    overlay = ax.imshow(mask, cmap=random_cmap, alpha=0.5)

    # Update the displayed image with red outlines
    displayed_image_with_edges = overlay_edges(mask, normalized_image)
    displayed_image.set_data(displayed_image_with_edges)

    # Redraw the figure
    fig.canvas.draw_idle()

    # Update the overlay
    #overlay.set_data(mask)

    # Update the displayed image with red outlines
    #displayed_image_with_edges = overlay_edges(mask, normalized_image)
    #displayed_image.set_data(displayed_image_with_edges)

    # Redraw the figure
    #fig.canvas.draw_idle()

# Function to save the modified mask
def save_mask(event, mask_path, image_path, img_src, mask_src, rescale_factor, original_dimensions):
    global mask, current_file_index
    
    # Relabel the mask
    binary_mask = mask > 0
    labeled_mask = label(binary_mask)
    mask = labeled_mask
    
    if mask_path is None:
    	dtype = 'uint16'
    else:
    	dtype = mask.dtype
    
    if rescale_factor != 1:
        # Resize the mask to match the original image dimensions
        resized_mask = resize(mask, original_dimensions, order=0, preserve_range=True, anti_aliasing=False).astype(dtype)
    else:
        resized_mask = mask
    
    # Rest of your code for saving the mask
    new_masks_folder = os.path.join(os.path.dirname(image_path if mask_path is None else mask_path), 'new_masks')
    os.makedirs(new_masks_folder, exist_ok=True)

    base_filename = os.path.basename(image_path if mask_path is None else mask_path)
    new_file_path = os.path.join(new_masks_folder, base_filename)

    # Save the resized mask
    imageio.imwrite(new_file_path, resized_mask)
    print(f'Mask saved to {new_file_path}')
    
    current_file_index += 1
    plt.close()  # Close current figure
    load_next_image(img_src, mask_src, rescale_factor)

# Function to convert RGB to Matplotlib color format
def rgb_to_mpl_color(r, g, b):
    return r / 255., g / 255., b / 255.

def hover(event):
    global ax, displayed_image  # Include displayed_image as well

    if event.inaxes == ax:
        x, y = int(event.xdata), int(event.ydata)
        if x >= 0 and y >= 0 and x < image.shape[1] and y < image.shape[0]:
            intensity_val = image[y, x]
            mask_val = mask[y, x]

            # Clear previous text
            for txt in ax.texts:
                txt.set_visible(False)

            # Display new text at desired location (e.g., top-left corner)
            ax.text(0.01, 0.99, f"Intensity: {intensity_val}, Mask: {mask_val}", transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.4))

            # Redraw the figure
            fig.canvas.draw_idle()

# Function to loade and optionally rescale the origional image
def downsample_tiff(image_path, scale_factor):
    with Image.open(image_path) as img:
        # Store original dimensions
        original_dimensions = (img.width, img.height)

        # Convert image to numpy array
        img_array = np.array(img)

        # Check the image mode and resize accordingly
        if img.mode in ['RGB', 'RGBA']:
            # For RGB and RGBA images, use PIL's resize
            resized_img = img.resize((int(img.width * scale_factor), int(img.height * scale_factor)), Image.Resampling.LANCZOS)
        elif img.mode == 'I;16':
            # For 16-bit grayscale images, use skimage's resize
            resized_img = resize(img_array, (int(img.height * scale_factor), int(img.width * scale_factor)), anti_aliasing=True, preserve_range=True)
        else:
            # For other modes, raise an error or handle as needed
            raise ValueError(f"Unsupported image mode: {img.mode}")

        return resized_img, original_dimensions
    
def invert_mask(event):
    global mask, overlay, displayed_image, normalized_image

    # Invert the mask: change non-zero values to 0 and zeros to 1
    mask = np.where(mask > 0, 0, 1).astype(np.int32)

    # Update the overlay and displayed image
    overlay.set_data(mask)
    displayed_image_with_edges = overlay_edges(mask, normalized_image)
    displayed_image.set_data(displayed_image_with_edges)

    # Redraw the figure
    fig.canvas.draw_idle()
    
def clear_mask(event):
    global mask, overlay, displayed_image, normalized_image, fig

    # Set all values in the mask to zero
    mask = np.zeros_like(mask)

    # Update the overlay and displayed image
    overlay.set_data(mask)
    displayed_image_with_edges = overlay_edges(mask, normalized_image)
    displayed_image.set_data(displayed_image_with_edges)

    # Redraw the figure
    fig.canvas.draw_idle()
    
# Function for freehand drawing
def freehand_draw(event):
    global freehand_points, mask, freehand_lines, mode_freehand

    if mode_freehand:  # Check if Freehand mode is active
        if event.inaxes != ax:
            return

        if event.button == 1:  # Left mouse click
            x, y = int(event.xdata), int(event.ydata)
            freehand_points.append((x, y))

            # Draw line segment between the last two points
            if len(freehand_points) > 1:
                line, = ax.plot([freehand_points[-2][0], freehand_points[-1][0]],
                                [freehand_points[-2][1], freehand_points[-1][1]],
                                color='red')
                freehand_lines.append(line)
                fig.canvas.draw()

        elif event.button == 3:  # Right mouse click to end drawing
            if len(freehand_points) > 2:
                # Close the shape and fill
                poly_points = np.array(freehand_points, dtype=np.int32)
                rr, cc = polygon(poly_points[:, 1], poly_points[:, 0], mask.shape)
                mask[rr, cc] = random.randint(1, 65535)
                overlay.set_data(mask)
                
                # Update displayed image with the edges
                displayed_image_with_edges = overlay_edges(mask, normalized_image)
                displayed_image.set_data(displayed_image_with_edges)
                fig.canvas.draw()
            
            # Reset points, remove lines
            freehand_points = []
            for line in freehand_lines:
                line.remove()
            freehand_lines = []
            fig.canvas.draw()
            
def update_overlay():
    global mask, overlay, displayed_image, normalized_image
    # Update the overlay
    overlay.set_data(mask)

    # Update the displayed image with edges or other features
    displayed_image_with_edges = overlay_edges(mask, normalized_image)
    displayed_image.set_data(displayed_image_with_edges)

    # Redraw the figure
    fig.canvas.draw_idle()
            
# Adding a short explanation buttons/sliders/boxes
def add_text_box_annotation(ax, text, x, y):
    ax.annotate(text, xy=(x, y), xycoords='axes fraction', ha='center', va='center',
                fontsize=8, color='gray')

def update_mode(new_mode):
    global mode_freehand, mode_magic_wand, mode_remove_object
    mode_freehand = (new_mode == 'freehand')
    mode_magic_wand = (new_mode == 'magic_wand')
    mode_remove_object = (new_mode == 'remove_object')

    # Update button colors or styles here to reflect the current mode
    if new_mode == 'freehand':
        highlight_selected_button(btn_freehand)
    elif new_mode == 'magic_wand':
        highlight_selected_button(btn_magic_wand)
    elif new_mode == 'remove_object':
        highlight_selected_button(btn_remove_object)
    else:
        highlight_selected_button(None)  # This will reset the "Deselect All" button


def on_deselect_all_clicked(event):
    # Reset all modes to False
    global mode_freehand, mode_magic_wand, mode_remove_object
    mode_freehand = False
    mode_magic_wand = False
    mode_remove_object = False

    # Highlight the "Deselect All" button
    highlight_selected_button(btn_deselect_all)

# Button callback functions
def on_freehand_clicked(event):
    update_mode('freehand')
    # Rest of the freehand functionality

def on_magic_wand_clicked(event):
    update_mode('magic_wand')
    # Rest of the magic wand functionality

def on_remove_object_clicked(event):
    update_mode('remove_object')
    # Rest of the remove object functionality
    
def highlight_selected_button(selected_button):
    # Reset all buttons to default color
    btn_freehand.color = default_button_color
    btn_magic_wand.color = default_button_color
    btn_remove_object.color = default_button_color
    btn_deselect_all.color = default_button_color

    # Highlight the selected button
    if selected_button is not None:
        selected_button.color = 'red'


def modify_mask(image_path, mask_path, itol, mpixels, min_size_for_removal, img_src, mask_src, rescale_factor):
    global image, mask, overlay, fig, ax, random_cmap, displayed_image, normalized_image
    global slider_itol, slider_mpixels, slider_min_size, slider_radius
    global freehand_points, freehand_lines, mode_freehand, mode_magic_wand, mode_remove_object, btn_remove_object, btn_freehand, btn_magic_wand, default_button_color, btn_deselect_all
    global btn_remove, btn_relabel, btn_fill_holes, btn_save, btn_invert, btn_clear, slider_lower_quantile, slider_upper_quantile
    
    # Initialize mode variables
    default_button_color = 'lightgrey'
    mode_freehand = False
    mode_magic_wand = False
    mode_remove_object = False
    
    save_clicked = False
    freehand_points = []
    freehand_lines = []
    
    # Modified _wrapper function
    def save_mask_wrapper(event):
        save_mask(event, mask_path, image_path, img_src, mask_src, rescale_factor, original_dimensions)
        
    # Callback function for updating the image based on slider values
    def update_image(val):
        global displayed_image, overlay

        lower_q = slider_lower_quantile.val
        upper_q = slider_upper_quantile.val
        normalized_image = normalize_to_dtype(image, lower_q, upper_q)

        # Update only the displayed intensity image, not the overlay
        displayed_image.set_data(normalized_image)
        fig.canvas.draw_idle()
	
    # Assign values to global variables
    image, original_dimensions = downsample_tiff(image_path, scale_factor=rescale_factor)
    
    # Calculate image area and max intensity
    height, width = image.shape[:2]
    image_area = height * width
    max_intensity = image.max()

    if mask_path != None:
        mask = imageio.imread(mask_path)
        mask = mask.astype(np.int32)
    else:
        mask = np.zeros_like(image)
        mask = mask.astype(np.int32)
        
    # Check if the mask is empty and modify it accordingly
    if np.max(mask) == 0:
        # If the mask is empty, initialize with a distinct value in a small area
        mask[0:2, 0:2] = 1  # Example initialization
        
    # Create a custom color map for the mask
    unique_labels = np.unique(mask)
    num_objects = len(unique_labels[unique_labels != 0])
    random_colors = np.random.rand(num_objects + 1, 4)
    random_colors[:, 3] = 1  # Set alpha to 1
    random_colors[0, :] = [0, 0, 0, 1]  # Background color
    random_cmap = mpl.colors.ListedColormap(random_colors)

    # Create a figure and display the image with the mask overlay
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.canvas.manager.set_window_title("SpaCr: modify mask")
    
    # Make the window full screen
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    
    fig.canvas.mpl_connect('button_press_event', freehand_draw)

    # Slider for lower quantile
    ax_lower_quantile = plt.axes([0.85, 0.7, 0.1, 0.02], figure=fig)
    slider_lower_quantile = Slider(ax_lower_quantile, 'Lower Quantile', 0, 25, valinit=0)

    # Slider for upper quantile
    ax_upper_quantile = plt.axes([0.85, 0.675, 0.1, 0.02], figure=fig)
    slider_upper_quantile = Slider(ax_upper_quantile, 'Upper Quantile', 75, 100, valinit=100)
    
    # Define quntile sliders
    slider_lower_quantile.on_changed(update_image)
    slider_upper_quantile.on_changed(update_image)

    # Normalize the image using default quantile values
    lower_q = slider_lower_quantile.val
    upper_q = slider_upper_quantile.val
    normalized_image = normalize_to_dtype(image, lower_q, upper_q)

    displayed_image = ax.imshow(normalized_image, cmap='gray')  # Store reference to the displayed image
    displayed_image_with_edges = overlay_edges(mask, normalized_image)  # Pass the required arguments
    displayed_image.set_data(displayed_image_with_edges)
    
    overlay = ax.imshow(mask, cmap=random_cmap, alpha=0.5)  # Store reference to the mask overlay
    
    # Define the button color
    button_color_1 = rgb_to_mpl_color(155, 55, 155)
    button_color_2 = rgb_to_mpl_color(55, 155, 155)
    
    # Define the button for deselecting all modes
    ax_deselect_all = plt.axes([0.75, 0.8, 0.05, 0.05]) 
    btn_deselect_all = Button(ax_deselect_all, 'Radius', color=default_button_color)
    btn_deselect_all.on_clicked(on_deselect_all_clicked)

    # Create buttons
    ax_freehand_btn = plt.axes([0.8, 0.8, 0.05, 0.05])
    btn_freehand = Button(ax_freehand_btn, 'Freehand')
    btn_freehand.on_clicked(on_freehand_clicked)

    ax_magic_wand_btn = plt.axes([0.85, 0.8, 0.05, 0.05])
    btn_magic_wand = Button(ax_magic_wand_btn, 'Magic Wand')
    btn_magic_wand.on_clicked(on_magic_wand_clicked)

    ax_remove_object_btn = plt.axes([0.9, 0.8, 0.05, 0.05])
    btn_remove_object = Button(ax_remove_object_btn, 'Remove Object')
    btn_remove_object.on_clicked(on_remove_object_clicked)
    
    # Slider for radius
    ax_radius = plt.axes([0.85, 0.6, 0.1, 0.02])
    slider_radius = Slider(ax_radius, 'Radius', 0, 10, valinit=1)
    
    # Slider for magic wand tolerance
    ax_itol = plt.axes([0.85, 0.575, 0.1, 0.02])
    slider_itol = Slider(ax_itol, 'Tolerance', 0, max_intensity, valinit=25)
    #ax_text_itol = plt.axes([0.975, 0.575, 0.01, 0.015])  # Adjust position and size
    #text_box_itol = TextBox(ax_text_itol, '', initial=str(slider_itol.val))
    #def submit_itol(text):
    #	slider_itol.set_val(float(text))
    #text_box_itol.on_submit(submit_itol)

    # Slider for magic wand max selection
    ax_mpixels = plt.axes([0.85, 0.55, 0.1, 0.02])
    slider_mpixels = Slider(ax_mpixels, 'Max Pixels', 0, image_area, valinit=500)
    
    # Slider for minimum size
    ax_min_size = plt.axes([0.85, 0.5, 0.1, 0.02])
    slider_min_size = Slider(ax_min_size, 'Min Size', 0, 2000, valinit=50)
    
    # Button to remove small objects
    ax_remove = plt.axes([0.85, 0.45, 0.1, 0.05])
    btn_remove = Button(ax_remove, 'Remove Small', color=button_color_1)
    btn_remove.on_clicked(remove_small_objects)
    btn_remove.label.set_fontsize(10)
    btn_remove.label.set_weight('bold')
    btn_remove.label.set_color('white')
    
    # Button for filling holes
    ax_fill_holes = plt.axes([0.85, 0.4, 0.1, 0.05])
    btn_fill_holes = Button(ax_fill_holes, 'Fill Holes', color=button_color_1)
    btn_fill_holes.on_clicked(fill_holes)
    btn_fill_holes.label.set_fontsize(10)
    btn_fill_holes.label.set_weight('bold')
    btn_fill_holes.label.set_color('white')
    
    # Button to relable objects
    ax_relabel = plt.axes([0.85, 0.35, 0.1, 0.05])
    btn_relabel = Button(ax_relabel, 'Relabel', color=button_color_1)
    btn_relabel.on_clicked(relabel_objects)
    btn_relabel.label.set_fontsize(10)
    btn_relabel.label.set_weight('bold')
    btn_relabel.label.set_color('white')
    
    # Button for inverting the mask
    ax_invert = plt.axes([0.85, 0.3, 0.1, 0.05])
    btn_invert = Button(ax_invert, 'Invert Mask', color=button_color_1)
    btn_invert.on_clicked(invert_mask)
    btn_invert.label.set_fontsize(10)
    btn_invert.label.set_weight('bold')
    btn_invert.label.set_color('white')
    
    # Define the button for clearing the mask
    ax_clear = plt.axes([0.85, 0.25, 0.1, 0.05])  # Adjust the position and size as needed
    btn_clear = Button(ax_clear, 'Clear Mask', color=button_color_1)
    btn_clear.on_clicked(clear_mask)
    btn_clear.label.set_fontsize(10)
    btn_clear.label.set_weight('bold')
    btn_clear.label.set_color('white')


    # Add a button for saving the mask
    ax_save = plt.axes([0.85, 0.2, 0.1, 0.05])
    btn_save = Button(ax_save, 'Save Mask', color=button_color_2)
    btn_save.on_clicked(save_mask_wrapper)
    btn_save.label.set_fontsize(10)
    btn_save.label.set_weight('bold')
    btn_save.label.set_color('white')

    #Text here

    # Connect the mouse click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Connect the mouse hover event
    fig.canvas.mpl_connect('motion_notify_event', hover)
    
    # Initialize default mode
    update_mode('none')

    plt.show()

# Add short text explanations
#add_text_box_annotation(ax_freehand, "Left: connect, Right: close", 0.85, 0.625)
#add_text_box_annotation(ax_lower_quantile, "Set the radius for drawing", 0.85, 0.7)
#add_text_box_annotation(ax_upper_quantile, "Set the radius for drawing", 0.85, 0.675)
#add_text_box_annotation(ax_check, "Left: select, Right: deselect", 0.85, 0.625)
#add_text_box_annotation(ax_radius, "Set the radius for drawing", 0.85, 0.6)
#add_text_box_annotation(ax_itol, "Set the radius for drawing", 0.85, 0.575)
#add_text_box_annotation(ax_mpixels, "Set the radius for drawing", 0.85, 0.55)
#add_text_box_annotation(ax_min_size, "Set the radius for drawing", 0.85, 0.5)
#add_text_box_annotation(ax_remove, "Set the radius for drawing", 0.85, 0.45)
#add_text_box_annotation(ax_fill_holes, "Set the radius for drawing", 0.85, 0.4)
#add_text_box_annotation(ax_relabel, "Set the radius for drawing", 0.85, 0.35)
#add_text_box_annotation(ax_invert, "Set the radius for drawing", 0.85, 0.3)
#add_text_box_annotation(ax_clear, "Set the radius for drawing", 0.85, 0.25)
#add_text_box_annotation(ax_save, "Set the radius for drawing", 0.85, 0.2)

def modify_masks(img_src, mask_src, rescale_factor):
    global save_clicked, current_file_index, file_list
    
    # Get list of .tif files in img_src
    img_files = {f for f in os.listdir(img_src) if f.lower().endswith('.tif')}
    
    new_masks_img = os.path.join(img_src,'new_masks')
    new_masks_mask = os.path.join(img_src,'new_masks')
    
    if mask_src != None:
        # Get list of files in mask_src (assuming same file names as in img_src)
        mask_files = {f for f in os.listdir(mask_src)}
    
        # Find intersection of both sets to get files present in both directories
        file_list = list(img_files & mask_files)
    else:
        file_list = img_files
        
    if os.path.exists(new_masks_img):
        # Get list of files in new_mask from images
        new_mask_files_img = {f for f in os.listdir(mask_src)}
        file_list = [elem for elem in file_list if elem not in new_mask_files_img]

    if os.path.exists(new_masks_mask):
        # Get list of files in new_mask from masks
        new_mask_files_mask = {f for f in os.listdir(mask_src)}
        file_list = [elem for elem in file_list if elem not in new_mask_files_mask]
    if len(file_list) == 0:
        return
    current_file_index = 0
    if file_list:
        load_next_image(img_src, mask_src, rescale_factor)

def initialize_file_list(img_src):
    global file_list
    file_list = [f for f in os.listdir(img_src) if f.lower().endswith('.tif')]
    return file_list

def load_next_image(img_src, mask_src, rescale_factor):
    global current_file_index, file_list
    # Update the file list
    file_list = initialize_file_list(img_src)  
    if current_file_index < len(file_list):
        file = file_list[current_file_index]
        image_path = os.path.join(img_src, file)
        print(f'Filename: {image_path}')
        if mask_src is not None and os.path.exists(mask_path):
            mask_path = os.path.join(mask_src, file)
        else:
            mask_path = None
        modify_mask(image_path, mask_path, itol=1000, mpixels=1000, min_size_for_removal=100, img_src=img_src, mask_src=mask_src, rescale_factor=rescale_factor)
    else:
        print(f"No corresponding mask found for {file}")
        current_file_index += 1
        load_next_image(img_src, mask_src, rescale_factor)
