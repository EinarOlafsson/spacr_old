## Run in notebook
#%matplotlib qt
#import matplotlib.pyplot as plt
#global save_clicked
#save_clicked = False
import matplotlib.pyplot as plt

import os
import time
import numpy as np
import imageio.v2 as imageio
from skimage import morphology
from collections import deque
from skimage.exposure import rescale_intensity
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
import matplotlib as mpl
from matplotlib.widgets import CheckButtons, Button, Slider
import skimage.morphology as morph
from skimage.measure import label
from scipy.ndimage import binary_fill_holes

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
                mask = magic_wand(image, mask, (x, y), slider_itol.val, slider_mpixels.val)
            elif event.button == 3:  # Right mouse button
                mask = magic_wand(image, mask, (x, y), slider_itol.val, slider_mpixels.val, remove=True)

            overlay.set_data(mask)
            overlay.set_cmap(random_cmap)
            fig.canvas.draw()
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
    global mask, overlay, fig, ax, random_cmap

    # Relabel the mask
    binary_mask = mask > 0
    labeled_mask = label(binary_mask)
    mask = labeled_mask

    # Regenerate the colormap
    n_labels = np.max(mask)
    random_colors = np.random.rand(n_labels + 1, 4)
    random_colors[0, :] = [0, 0, 0, 0]  # Background color
    random_cmap = mpl.colors.ListedColormap(random_colors)

    # Remove the old overlay and create a new one
    overlay.remove()
    overlay = ax.imshow(mask, cmap=random_cmap, alpha=0.5)

    # Redraw the figure
    fig.canvas.draw_idle()
    #fig.canvas.draw()

    
# Function to fill holes in objects
def fill_holes(event):
    global mask, overlay, fig, ax

    # Ensure the mask is boolean for binary_fill_holes
    binary_mask = mask > 0

    # Fill holes in the binary mask
    filled_mask = binary_fill_holes(binary_mask)

    # Update the original mask; convert back to original dtype if needed
    mask = filled_mask.astype(mask.dtype) * 255

    # Update the overlay
    overlay.set_data(mask)
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

def modify_mask(image_path, mask_path, itol, mpixels, min_size_for_removal, img_src, mask_src):
    global image, mask, overlay, fig, ax, random_cmap
    global displayed_image
    global slider_itol, slider_mpixels, slider_min_size, slider_radius, check_magic_wand
    global slider_lower_quantile, slider_upper_quantile
    global btn_remove, btn_relabel, btn_fill_holes, btn_save

    # Modified save_mask_wrapper function
    def save_mask_wrapper(event):
        save_mask(event, mask_path, img_src, mask_src)
        
    # Callback function for updating the image based on slider values
    # This takes to long add a delay in rendering here
    def update_image(val):
        global displayed_image, overlay

        lower_q = slider_lower_quantile.val
        upper_q = slider_upper_quantile.val
        normalized_image = normalize_to_dtype(image, lower_q, upper_q)

        # Update only the displayed intensity image, not the overlay
        displayed_image.set_data(normalized_image)
        fig.canvas.draw_idle()

    # Assign values to global variables
    image = imageio.imread(image_path)
    mask = imageio.imread(mask_path)
    mask = mask.astype(np.int32)
    
    # Check if the mask is empty and modify it accordingly
    if np.max(mask) == 0:
        # If the mask is empty, initialize with a distinct value in a small area
        mask[0:2, 0:2] = 1  # Example initialization

    # Normalize the image using default quantile values
    normalized_image = normalize_to_dtype(image, 2, 98)

    # Create a custom color map for the mask
    unique_labels = np.unique(mask)
    num_objects = len(unique_labels[unique_labels != 0])
    random_colors = np.random.rand(num_objects + 1, 4)
    random_colors[:, 3] = 1  # Set alpha to 1
    random_colors[0, :] = [0, 0, 0, 1]  # Background color
    random_cmap = mpl.colors.ListedColormap(random_colors)

    # Create a figure and display the image with the mask overlay
    fig, ax = plt.subplots(figsize=(8, 6))
    displayed_image = ax.imshow(normalized_image, cmap='gray')  # Store reference to the displayed image
    overlay = ax.imshow(mask, cmap=random_cmap, alpha=0.5)  # Store reference to the mask overlay

    # Slider for lower quantile
    ax_lower_quantile = plt.axes([0.8, 0.7, 0.1, 0.02], figure=fig)
    slider_lower_quantile = Slider(ax_lower_quantile, 'Lower Quantile', 0, 10, valinit=2)

    # Slider for upper quantile
    ax_upper_quantile = plt.axes([0.8, 0.75, 0.1, 0.02], figure=fig)
    slider_upper_quantile = Slider(ax_upper_quantile, 'Upper Quantile', 90, 100, valinit=98)

    slider_lower_quantile.on_changed(update_image)
    slider_upper_quantile.on_changed(update_image)

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
    btn_save.on_clicked(save_mask_wrapper)
    btn_save.label.set_fontsize(10)
    btn_save.label.set_weight('bold')
    btn_save.label.set_color('white')

    # Connect the mouse click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Connect the mouse hover event
    fig.canvas.mpl_connect('motion_notify_event', hover)

    # Sliders
    ax_itol = plt.axes([0.8, 0.55, 0.1, 0.02])
    slider_itol = Slider(ax_itol, 'Tolerance', 0, 2000, valinit=25)

    ax_mpixels = plt.axes([0.8, 0.5, 0.1, 0.02])
    slider_mpixels = Slider(ax_mpixels, 'Max Pixels', 0, 2000, valinit=500)

    ax_min_size = plt.axes([0.8, 0.45, 0.1, 0.02])
    slider_min_size = Slider(ax_min_size, 'Min Size', 0, 2000, valinit=50)
    
    # Checkbox for toggling Magic Wand
    ax_check = plt.axes([0.8, 0.6, 0.1, 0.02])
    check_magic_wand = CheckButtons(ax_check, ['Magic Wand'], [True])
    
    # Slider for radius
    ax_radius = plt.axes([0.8, 0.65, 0.1, 0.02])
    slider_radius = Slider(ax_radius, 'Radius', 0, 10, valinit=1)
    
    plt.show()
          
def modify_masks(img_src, mask_src):
    global save_clicked, current_file_index, file_list
    
    # Get list of .tif files in img_src
    img_files = {f for f in os.listdir(img_src) if f.lower().endswith('.tif')}

    # Get list of files in mask_src (assuming same file names as in img_src)
    mask_files = {f for f in os.listdir(mask_src)}

    # Find intersection of both sets to get files present in both directories
    file_list = list(img_files & mask_files)

    
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
            modify_mask(image_path, mask_path, itol=1000, mpixels=1000, min_size_for_removal=100, img_src=img_src, mask_src=mask_src)
        
        else:
            print(f"No corresponding mask found for {file}")
            current_file_index += 1
            load_next_image(img_src, mask_src)
