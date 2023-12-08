## Run in notebook
#%matplotlib qt
#import matplotlib.pyplot as plt
#global save_clicked
#save_clicked = False

import os
import time
import numpy as np
import imageio.v2 as imageio
from skimage import morphology
from collections import deque
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
import matplotlib as mpl
from matplotlib.widgets import CheckButtons, Button, Slider
import skimage.morphology as morph
from skimage.measure import label
from scipy.ndimage import binary_fill_holes

# Function to normalize the image
def normalize_to_dtype(array, q1=2, q2=98):
    # Ensure array is at least 3D
    if len(array.shape) == 2:
        array = np.expand_dims(array, axis=-1)

    num_channels = array.shape[-1]
    new_stack = np.empty_like(array)

    for channel in range(num_channels):
        img = array[..., channel]
        non_zero_img = img[img > 0]

        # Determine min and max for intensity scaling
        if non_zero_img.size > 0:
            img_min = np.percentile(non_zero_img, q1)
            img_max = np.percentile(non_zero_img, q2)
        else:
            img_min, img_max = img.min(), img.max()

        # Rescale intensity
        new_stack[..., channel] = rescale_intensity(img, in_range=(img_min, img_max), out_range='dtype')

    # Remove the added dimension for 2D inputassociates phenotypes with genotypes by randomly inducing
    if new_stack.shape[-1] == 1:
        new_stack = np.squeeze(new_stack, axis=-1)

    return new_stack

def find_nearest_nonzero_pixel(mask, seed_point):
    y, x = seed_point
    non_zero_coords = np.argwhere(mask > 0)  # Find all non-zero pixels

    if len(non_zero_coords) == 0:
        return 0  # No non-zero pixels in the mask

    # Calculate distances to the clicked point
    distances = np.sqrt((non_zero_coords[:, 0] - y) ** 2 + (non_zero_coords[:, 1] - x) ** 2)
    nearest_pixel_index = np.argmin(distances)
    nearest_pixel_value = mask[tuple(non_zero_coords[nearest_pixel_index])]

    return nearest_pixel_value

# Magic Wand function
def magic_wand(image, mask, seed_point, intensity_tolerance=100, max_pixels=1000, remove=False):
    x, y = seed_point
    initial_value = np.float32(image[y, x])  # Cast to float to prevent overflow
    to_check = deque([(x, y)])
    checked = set()  # Reset for each new call
    
    if remove:
        fill_value = 0
    else:
        fill_value = find_nearest_nonzero_pixel(mask, seed_point)
        if fill_value == 0:
            fill_value = 65535 if mask.dtype == np.uint16 else 255  # Default fill value if no non-zero pixels are found

    filled_pixels = 0

    while to_check and filled_pixels < max_pixels:
        x, y = to_check.popleft()
        if (x, y) in checked or not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            continue

        checked.add((x, y))

        current_value = np.float32(image[y, x])  # Cast to float
        if abs(current_value - initial_value) <= intensity_tolerance:
            mask[y, x] = fill_value
            filled_pixels += 1

            # Check and add adjacent pixels within tolerance
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and (nx, ny) not in checked:
                    next_value = np.float32(image[ny, nx])  # Cast to float
                    if abs(next_value - initial_value) <= intensity_tolerance:
                        to_check.append((nx, ny))

    return mask

# Function to modify mask
def modify_mask(mask, position):
    y, x = position
    if mask[y, x] == 0:
        non_zero_coords = np.argwhere(mask)
        distances = np.sum((non_zero_coords - np.array(position))**2, axis=1)
        closest_pixel = non_zero_coords[np.argmin(distances)]
        mask_value = mask[closest_pixel[0], closest_pixel[1]]
    else:
        mask_value = mask[y, x]

    mask[y, x] = mask_value
    return mask

# Mouse click event handler
def on_click(event):
    global mask, slider_itol, slider_mpixels, slider_radius, check_magic_wand
    global save_clicked
    save_clicked = False
    if fig.canvas.toolbar.mode != '':
        return
    if event.xdata is not None and event.ydata is not None and event.inaxes == ax:
        x, y = int(event.xdata), int(event.ydata)
        intensity_tolerance = slider_itol.val
        max_pixels = slider_mpixels.val
        radius = int(slider_radius.val)
        use_magic_wand = check_magic_wand.get_status()[0]

        if use_magic_wand:
            if event.button == 1:  # Left mouse button
                mask = magic_wand(image, mask, (x, y), intensity_tolerance, max_pixels)
            elif event.button == 3:  # Right mouse button
                mask = magic_wand(image, mask, (x, y), intensity_tolerance, max_pixels, remove=True)
        else:
            # Define the area to be modified based on the radius
            y_min, y_max = max(y - radius, 0), min(y + radius + 1, mask.shape[0])
            x_min, x_max = max(x - radius, 0), min(x + radius + 1, mask.shape[1])
            if event.button == 1:  # Left mouse button
                mask[y_min:y_max, x_min:x_max] = 255
            elif event.button == 3:  # Right mouse button
                mask[y_min:y_max, x_min:x_max] = 0

        overlay.set_data(mask)
        overlay.set_cmap(random_cmap)
        fig.canvas.draw()


# Function to remove small objects
def remove_small_objects(event):
    global mask, slider_min_size
    min_size = slider_min_size.val
    mask = morph.remove_small_objects(mask > 0, min_size)
    overlay.set_data(mask)
    overlay.set_cmap(random_cmap)
    fig.canvas.draw()

# Function to relabel objects
def relabel_objects(event):
    global mask
    mask = label(mask)
    overlay.set_data(mask)
    overlay.set_cmap(random_cmap)
    fig.canvas.draw()
    
# Function to fill holes in objects
def fill_holes(event):
    global mask
    mask = binary_fill_holes(mask).astype(mask.dtype)
    overlay.set_data(mask)
    overlay.set_cmap(random_cmap)
    fig.canvas.draw()

#Function to save the modified mask
def save_mask(event, mask_path, img_src, mask_src):
    global mask, current_file_index

    # Create the 'new_masks' folder if it doesn't exist
    new_masks_folder = os.path.join(os.path.dirname(mask_path), 'new_masks')
    os.makedirs(new_masks_folder, exist_ok=True)

    # Construct the new file path
    base_filename = os.path.basename(mask_path)
    new_file_path = os.path.join(new_masks_folder, base_filename)

    # Save the mask
    imageio.imwrite(new_file_path, mask)
    print(f'Mask saved to {new_file_path}')
    
    current_file_index += 1
    plt.close()  # Close current figure
    load_next_image(img_src, mask_src)

# Function to convert RGB to Matplotlib color format
def rgb_to_mpl_color(r, g, b):
    return r / 255., g / 255., b / 255.

def hover(event):
    global ax  # Declare 'ax' as global here
    if event.inaxes == ax:
        x, y = int(event.xdata), int(event.ydata)
        if x >= 0 and y >= 0 and x < image.shape[1] and y < image.shape[0]:
            intensity_val = image[y, x]
            mask_val = mask[y, x]
            plt.gca().set_title(f"Intensity: {intensity_val}, Mask: {mask_val}")



def medify_mask(image_path, mask_path, itol, mpixels, min_size_for_removal):
    global image, mask, overlay, fig, random_cmap
    global slider_itol, slider_mpixels, slider_min_size, slider_radius, check_magic_wand
    global btn_remove, btn_relabel, btn_fill_holes, btn_save, ax  # Add 'ax' here
    #global mask_path
    #mask_path = os.path.join(mask_src, file)

    # Assign values to global variables
    intensity_tolerance = itol
    max_pixels = mpixels
    min_size = min_size_for_removal
    
    image = imageio.imread(image_path)
    mask = imageio.imread(mask_path)
    
    # Normalize the image
    normalized_image = normalize_to_dtype(image)

    # Create a custom color map for the mask
    unique_labels = np.unique(mask)
    num_objects = len(unique_labels[unique_labels != 0])
    random_colors = np.random.rand(num_objects + 1, 4)
    random_colors[:, 3] = 1  # Set alpha to 1
    random_colors[0, :] = [0, 0, 0, 1]  # Background color
    random_cmap = mpl.colors.ListedColormap(random_colors)
    
    # Create a figure and display the image with the mask overlay
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(normalized_image, cmap='gray')
    overlay = ax.imshow(mask, cmap=random_cmap, alpha=0.5)

    # Define the button color
    button_color_1 = rgb_to_mpl_color(155, 55, 155)
    button_color_2 = rgb_to_mpl_color(55, 155, 155)

    # Add buttons
    ax_remove = plt.axes([0.8, 0.05, 0.15, 0.075])
    btn_remove = Button(ax_remove, 'Remove Small', color=button_color_1)
    btn_remove.on_clicked(remove_small_objects)
    btn_remove.label.set_fontsize(10)
    btn_remove.label.set_weight('bold')
    btn_remove.label.set_color('white')

    ax_relabel = plt.axes([0.8, 0.15, 0.15, 0.075])
    btn_relabel = Button(ax_relabel, 'Relabel', color=button_color_1)
    btn_relabel.on_clicked(relabel_objects)
    btn_relabel.label.set_fontsize(10)
    btn_relabel.label.set_weight('bold')
    btn_relabel.label.set_color('white')

    # Button for filling holes
    ax_fill_holes = plt.axes([0.8, 0.25, 0.15, 0.075])
    btn_fill_holes = Button(ax_fill_holes, 'Fill Holes', color=button_color_1)
    btn_fill_holes.on_clicked(fill_holes)
    btn_fill_holes.label.set_fontsize(10)
    btn_fill_holes.label.set_weight('bold')
    btn_fill_holes.label.set_color('white')

    # Add a button for saving the mask
    ax_save = plt.axes([0.8, 0.35, 0.15, 0.075])
    btn_save = Button(ax_save, 'Save Mask', color=button_color_2)
    btn_save.on_clicked(lambda event: save_mask(event, mask_path, img_src, mask_src))
    btn_save.label.set_fontsize(10)
    btn_save.label.set_weight('bold')
    btn_save.label.set_color('white')

    # Connect the mouse click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Connect the mouse hover event
    fig.canvas.mpl_connect('motion_notify_event', hover)

    # Sliders
    ax_itol = plt.axes([0.8, 0.55, 0.15, 0.03])
    slider_itol = Slider(ax_itol, 'Tolerance', 0, 2000, valinit=100)

    ax_mpixels = plt.axes([0.8, 0.5, 0.15, 0.03])
    slider_mpixels = Slider(ax_mpixels, 'Max Pixels', 0, 2000, valinit=500)

    ax_min_size = plt.axes([0.8, 0.45, 0.15, 0.03])
    slider_min_size = Slider(ax_min_size, 'Min Size', 0, 2000, valinit=50)
    
    # Checkbox for toggling Magic Wand
    ax_check = plt.axes([0.8, 0.6, 0.15, 0.1])
    check_magic_wand = CheckButtons(ax_check, ['Magic Wand'], [True])
    
    # Slider for radius
    ax_radius = plt.axes([0.8, 0.65, 0.15, 0.03])
    slider_radius = Slider(ax_radius, 'Radius', 0, 10, valinit=1)

    plt.show()

def modify_masks(img_src, mask_src):
    global save_clicked

    for file in os.listdir(img_src):
        ext = os.path.splitext(file)[1]
        if ext.lower() == '.tif':
            image_path = os.path.join(img_src, file)
            mask_path = os.path.join(mask_src, file)

            if os.path.exists(mask_path):
                save_clicked = False
                medify_mask(image_path, mask_path, itol=1000, mpixels=1000, min_size_for_removal=100)

                # Update the GUI and wait for the save button to be clicked
                while not save_clicked:
                    plt.pause(0.1)  # Allow GUI to process events
                
                # Close the figure and proceed to next image
                plt.close()
                save_clicked = False
            else:
                print(f"No corresponding mask found for {file}")
                
def modify_masks(img_src, mask_src):
    global save_clicked, current_file_index, file_list
    file_list = [f for f in os.listdir(img_src) if f.lower().endswith('.tif')]
    current_file_index = 0

    if file_list:
        load_next_image(img_src, mask_src)

def initialize_file_list(img_src):
    global file_list
    file_list = [f for f in os.listdir(img_src) if f.lower().endswith('.tif')]
    return file_list

def load_next_image(img_src, mask_src):
    global current_file_index, file_list
    file_list = initialize_file_list(img_src)  # Update the file list

    if current_file_index < len(file_list):
        file = file_list[current_file_index]
        image_path = os.path.join(img_src, file)
        mask_path = os.path.join(mask_src, file)
        
        if os.path.exists(mask_path):
            medify_mask(image_path, mask_path, itol=1000, mpixels=1000, min_size_for_removal=100)
        else:
            print(f"No corresponding mask found for {file}")
            current_file_index += 1
            load_next_image(img_src, mask_src)
