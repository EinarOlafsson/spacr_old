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
    
    for package in dependencies:
        print(f"Installing {package}")
        subprocess.run([conda_PATH, "install", "-n", env_name, package, "-y"])
    
    # Install additional packages
    subprocess.run([pip_PATH, "install", "opencv-python-headless"])
    subprocess.run([pip_PATH, "install", "PyQt5"])
    subprocess.run([pip_PATH, "install", "screeninfo"])

    print("Dependencies installation complete.")

env_name = "spacr_modify_masks_tkinter"

conda_PATH, python_PATH, pip_PATH, env_PATH = get_paths(env_name)

dependencies = ["tkinter", "imageio", "scipy", "pillow", "scikit-image", "ipykernel", "requests", "h5py", "pyzmq"]

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
    #SystemExit()

#######################################################################################################################

import os
import numpy as np
import tkinter as tk
import imageio.v2 as imageio
from collections import deque
from tkinter import filedialog
from PIL import Image, ImageTk
from skimage.draw import polygon
from skimage.transform import resize
from skimage.measure import label as sk_label
from scipy.ndimage import binary_fill_holes, label

class modify_masks:
    def __init__(self, root, folder_path, scale_factor, width, height):
        self.root = root
        self.folder_path = folder_path
        self.scale_factor = scale_factor
        self.image_filenames = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        self.masks_folder = os.path.join(folder_path, 'masks')
        self.current_image_index = 0
        self.canvas_width = width
        self.canvas_height = height
        #print(f'loaded image: {self.image_filenames[current_image_index]}')

        # Initialize these before the first call to load_image_and_mask
        self.zoom_rectangle_start = None
        self.zoom_rectangle_end = None
        self.zoom_rectangle_id = None
        self.zoom_x0 = None
        self.zoom_y0 = None
        self.zoom_x1 = None
        self.zoom_y1 = None
        self.zoom_mask = None
        self.zoom_image = None
        self.zoom_image_orig = None
        self.root.bind('<Return>', self.apply_zoom_on_enter)
        
        # Initialize mode flags
        self.drawing = False
        self.zoom_active = False
        self.magic_wand_active = False
        self.brush_active = False
        self.zoom_scale = 1
        
        # Initialize percentile values
        self.lower_quantile = tk.StringVar(value="1.0")
        self.upper_quantile = tk.StringVar(value="99.0")
        
        self.image, self.mask = self.load_image_and_mask(self.current_image_index)
        self.original_size = self.image.shape
        
        self.image, self.mask = self.resize_arrays(self.image,self.mask)
        
        self.setup_navigation_toolbar()
        
        # Initialize the canvas
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()
        self.canvas.bind("<Motion>", self.update_mouse_info)
        
        self.magic_wand_tolerance = tk.StringVar(value="10")
        
        self.setup_mode_toolbar()
        self.setup_function_toolbar()
        self.setup_zoom_toolbar()
        
        self.draw_coordinates = []
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.finish_drawing_if_active)
        self.display_image()
        
    def update_display(self):
        if self.zoom_active:
            self.display_zoomed_image()
        else:
            self.display_image()
        
    def update_mouse_info(self, event):
        x, y = event.x, event.y
        intensity = "N/A"
        mask_value = "N/A"
        pixel_count = "N/A"  # Variable to hold the pixel count

        if self.zoom_active:
            if 0 <= x < self.canvas_width and 0 <= y < self.canvas_height:
                intensity = self.zoom_image_orig[y, x] if self.zoom_image_orig is not None else "N/A"
                mask_value = self.zoom_mask[y, x] if self.zoom_mask is not None else "N/A"
        else:
            if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
                intensity = self.image[y, x]
                mask_value = self.mask[y, x]

        # Count the pixels with the same mask value, if the value is not 0
        if mask_value != "N/A" and mask_value != 0:
            pixel_count = np.sum(self.mask == mask_value)

        self.intensity_label.config(text=f"Intensity: {intensity}")
        self.mask_value_label.config(text=f"Mask: {mask_value}, Area: {pixel_count}")
        
        self.mask_value_label.config(text=f"Mask: {mask_value}")
        if mask_value != "N/A" and mask_value != 0:
            self.pixel_count_label.config(text=f"Area: {pixel_count}")
        else:
            self.pixel_count_label.config(text="Area: N/A")

    def apply_zoom_on_enter(self, event):
        if self.zoom_active and self.zoom_rectangle_start is not None:
            self.set_zoom_rectangle_end(event)
        
    def normalize_image(self, image, lower_quantile, upper_quantile):
        lower_bound = np.percentile(image, lower_quantile)
        upper_bound = np.percentile(image, upper_quantile)
        normalized = np.clip(image, lower_bound, upper_bound)
        normalized = (normalized - lower_bound) / (upper_bound - lower_bound)
        max_value = np.iinfo(image.dtype).max
        normalized = (normalized * max_value).astype(image.dtype)
        return normalized
    
    def resize_arrays(self, img, mask):
        original_dtype = img.dtype
        scaled_height = int(img.shape[0] * self.scale_factor)
        scaled_width = int(img.shape[1] * self.scale_factor)
        scaled_img = resize(img, (scaled_height, scaled_width), anti_aliasing=True, preserve_range=True)
        scaled_mask = resize(mask, (scaled_height, scaled_width), order=0, anti_aliasing=False, preserve_range=True)
        stretched_img = resize(scaled_img, (self.canvas_height, self.canvas_width), anti_aliasing=True, preserve_range=True)
        stretched_mask = resize(scaled_mask, (self.canvas_height, self.canvas_width), order=0, anti_aliasing=False, preserve_range=True)
        return stretched_img.astype(original_dtype), stretched_mask.astype(original_dtype)
    
    def setup_navigation_toolbar(self):
        navigation_toolbar = tk.Frame(self.root)
        navigation_toolbar.pack(side='top', fill='x')
        prev_btn = tk.Button(navigation_toolbar, text="Previous", command=self.previous_image)
        prev_btn.pack(side='left')
        next_btn = tk.Button(navigation_toolbar, text="Next", command=self.next_image)
        next_btn.pack(side='left')
        save_btn = tk.Button(navigation_toolbar, text="Save", command=self.save_mask)
        save_btn.pack(side='left')
        self.intensity_label = tk.Label(navigation_toolbar, text="Image: N/A")
        self.intensity_label.pack(side='right')
        self.mask_value_label = tk.Label(navigation_toolbar, text="Mask: N/A")
        self.mask_value_label.pack(side='right')
        self.pixel_count_label = tk.Label(navigation_toolbar, text="Area: N/A")
        self.pixel_count_label.pack(side='right')

    def setup_mode_toolbar(self):
        self.mode_toolbar = tk.Frame(self.root)
        self.mode_toolbar.pack(side='top', fill='x')
        self.draw_btn = tk.Button(self.mode_toolbar, text="Draw", command=self.toggle_draw_mode)
        self.draw_btn.pack(side='left')
        self.magic_wand_btn = tk.Button(self.mode_toolbar, text="Magic Wand", command=self.toggle_magic_wand_mode)
        self.magic_wand_btn.pack(side='left')
        tk.Label(self.mode_toolbar, text="Tolerance:").pack(side='left')
        self.tolerance_entry = tk.Entry(self.mode_toolbar, textvariable=self.magic_wand_tolerance)
        self.tolerance_entry.pack(side='left')
        tk.Label(self.mode_toolbar, text="Max Pixels:").pack(side='left')
        self.max_pixels_entry = tk.Entry(self.mode_toolbar)
        self.max_pixels_entry.insert(0, "1000")  # Default value
        self.max_pixels_entry.pack(side='left')
        self.erase_btn = tk.Button(self.mode_toolbar, text="Erase", command=self.toggle_erase_mode)
        self.erase_btn.pack(side='left')
    
        self.brush_btn = tk.Button(self.mode_toolbar, text="Brush", command=self.toggle_brush_mode)
        self.brush_btn.pack(side='left')
        self.brush_size_entry = tk.Entry(self.mode_toolbar)
        self.brush_size_entry.insert(0, "10")  # Default brush size
        self.brush_size_entry.pack(side='left')
        tk.Label(self.mode_toolbar, text="Brush Size:").pack(side='left')

    def setup_function_toolbar(self):
        self.function_toolbar = tk.Frame(self.root)
        self.function_toolbar.pack(side='top', fill='x')
        self.fill_btn = tk.Button(self.function_toolbar, text="Fill", command=self.fill_objects)
        self.fill_btn.pack(side='left')
        self.relabel_btn = tk.Button(self.function_toolbar, text="Relabel", command=self.relabel_objects)
        self.relabel_btn.pack(side='left')
        self.clear_btn = tk.Button(self.function_toolbar, text="Clear", command=self.clear_objects)
        self.clear_btn.pack(side='left')
        self.invert_btn = tk.Button(self.function_toolbar, text="Invert", command=self.invert_mask)
        self.invert_btn.pack(side='left')

    def setup_zoom_toolbar(self):
        self.zoom_toolbar = tk.Frame(self.root)
        self.zoom_toolbar.pack(side='top', fill='x')
        self.zoom_btn = tk.Button(self.zoom_toolbar, text="Zoom", command=self.toggle_zoom_mode)
        self.zoom_btn.pack(side='left')
        self.normalize_btn = tk.Button(self.zoom_toolbar, text="Apply Normalization", command=self.apply_normalization)
        self.normalize_btn.pack(side='left')
        self.lower_entry = tk.Entry(self.zoom_toolbar, textvariable=self.lower_quantile)
        self.lower_entry.pack(side='left')
        tk.Label(self.zoom_toolbar, text="Lower Percentile:").pack(side='left')
        self.upper_entry = tk.Entry(self.zoom_toolbar, textvariable=self.upper_quantile)
        self.upper_entry.pack(side='left')
        tk.Label(self.zoom_toolbar, text="Upper Percentile:").pack(side='left')
        
    def load_image_and_mask(self, index):
        image_path = os.path.join(self.folder_path, self.image_filenames[index])
        image = imageio.imread(image_path)        
        #if image.dtype != np.uint8:
        #    image = (image / np.max(image) * 255).astype(np.uint8)
        mask_filename = os.path.splitext(self.image_filenames[index])[0] + '_mask.png'
        mask_path = os.path.join(self.masks_folder, mask_filename)
        if os.path.exists(mask_path):
            mask = imageio.imread(mask_path)
            if mask.dtype != np.uint8:
                mask = (mask / np.max(mask) * 255).astype(np.uint8)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        return image, mask
        
    def display_image(self):
        self.canvas_width = self.canvas.winfo_width()
        self.canvas_height = self.canvas.winfo_height()
        if self.zoom_rectangle_id is not None:
            self.canvas.delete(self.zoom_rectangle_id)
            self.zoom_rectangle_id = None
        lower_quantile = float(self.lower_quantile.get()) if self.lower_quantile.get() else 1.0
        upper_quantile = float(self.upper_quantile.get()) if self.upper_quantile.get() else 99.0
        normalized = self.normalize_image(self.image, lower_quantile, upper_quantile)
        combined = self.overlay_mask_on_image(normalized, self.mask)
        self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(combined))
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)

    def display_zoomed_image(self):
        if self.zoom_rectangle_start and self.zoom_rectangle_end:
            # Convert canvas coordinates to image coordinates
            x0, y0 = self.canvas_to_image(*self.zoom_rectangle_start)
            x1, y1 = self.canvas_to_image(*self.zoom_rectangle_end)
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            self.zoom_x0 = x0
            self.zoom_y0 = y0
            self.zoom_x1 = x1
            self.zoom_y1 = y1
            # Normalize the entire image
            lower_quantile = float(self.lower_quantile.get()) if self.lower_quantile.get() else 1.0
            upper_quantile = float(self.upper_quantile.get()) if self.upper_quantile.get() else 99.0
            normalized_image = self.normalize_image(self.image, lower_quantile, upper_quantile)
            # Extract the zoomed portion of the normalized image and mask
            self.zoom_image = normalized_image[y0:y1, x0:x1]
            self.zoom_image_orig = self.image[y0:y1, x0:x1]
            self.zoom_mask = self.mask[y0:y1, x0:x1]
            original_mask_area = self.mask.shape[0] * self.mask.shape[1]
            zoom_mask_area = self.zoom_mask.shape[0] * self.zoom_mask.shape[1]
            if original_mask_area > 0:
                self.zoom_scale = original_mask_area/zoom_mask_area
            # Resize the zoomed image and mask to fit the canvas
            canvas_height = self.canvas.winfo_height()
            canvas_width = self.canvas.winfo_width()
            
            if self.zoom_image.size > 0 and canvas_height > 0 and canvas_width > 0:
                self.zoom_image = resize(self.zoom_image, (canvas_height, canvas_width), preserve_range=True).astype(self.zoom_image.dtype)
                self.zoom_image_orig = resize(self.zoom_image_orig, (canvas_height, canvas_width), preserve_range=True).astype(self.zoom_image_orig.dtype)
                #self.zoom_mask = resize(self.zoom_mask, (canvas_height, canvas_width), preserve_range=True).astype(np.uint8)
                self.zoom_mask = resize(self.zoom_mask, (canvas_height, canvas_width), order=0, preserve_range=True).astype(np.uint8)
                combined = self.overlay_mask_on_image(self.zoom_image, self.zoom_mask)
                self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(combined))
                self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)

    def overlay_mask_on_image(self, image, mask, alpha=0.5):
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
        mask = mask.astype(np.int32)
        max_label = np.max(mask)
        np.random.seed(0)
        colors = np.random.randint(0, 255, size=(max_label + 1, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # background color
        colored_mask = colors[mask]
        image_8bit = (image / 256).astype(np.uint8)
        # Blend the mask and the image with transparency
        combined_image = np.where(mask[..., None] > 0, 
                                  np.clip(image_8bit * (1 - alpha) + colored_mask * alpha, 0, 255), 
                                  image_8bit)
        # Convert the final image back to uint8
        combined_image = combined_image.astype(np.uint8)
        return combined_image

    def previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.image, self.mask = self.load_image_and_mask(self.current_image_index)
            self.display_image()

    def next_image(self):
        if self.current_image_index < len(self.image_filenames) - 1:
            self.current_image_index += 1
            self.image, self.mask = self.load_image_and_mask(self.current_image_index)
            self.zoom_rectangle_start = None
            self.zoom_rectangle_end = None
            self.zoom_x0 = None
            self.zoom_y0 = None
            self.zoom_x1 = None
            self.zoom_y1 = None
            self.zoom_mask = None
            self.zoom_image = None
            self.zoom_image_orig = None
            self.zoom_scale = 1  # or the appropriate initial value
            self.display_image()

    def save_mask(self):
        try:
            if self.current_image_index < len(self.image_filenames):
                original_size = self.original_size
                resized_mask = resize(self.mask, original_size, preserve_range=True).astype(np.uint16)
                save_folder = os.path.join(self.folder_path, 'new_masks')
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                image_filename = os.path.splitext(self.image_filenames[self.current_image_index])[0] + '.png'
                save_path = os.path.join(save_folder, image_filename)
                
                print(f"Saving mask to: {save_path}")  # Debug print
                imageio.imwrite(save_path, resized_mask)
                print("Save successful!")  # Confirm save
        except Exception as e:
            print(f"Error during saving: {e}")

    def get_scaling_factors(self, img_width, img_height, canvas_width, canvas_height):
        x_scale = img_width / canvas_width
        y_scale = img_height / canvas_height
        return x_scale, y_scale
    
    def canvas_to_image(self, x_canvas, y_canvas):
        x_scale, y_scale = self.get_scaling_factors(
            self.image.shape[1], self.image.shape[0],
            self.canvas_width, self.canvas_height
        )
        x_image = int(x_canvas * x_scale)
        y_image = int(y_canvas * y_scale)
        return x_image, y_image

    def set_zoom_rectangle_start(self, event):
        if self.zoom_active:
            self.zoom_rectangle_start = (event.x, event.y)
    
    def set_zoom_rectangle_end(self, event):
        if self.zoom_active:
            self.zoom_rectangle_end = (event.x, event.y)
    
            # Delete the red zoom rectangle
            if self.zoom_rectangle_id is not None:
                self.canvas.delete(self.zoom_rectangle_id)
                self.zoom_rectangle_id = None
    
            self.display_zoomed_image()  
            self.canvas.unbind("<Motion>")  
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<Button-3>")  
            self.canvas.bind("<Motion>", self.update_mouse_info)
            
    def update_zoom_box(self, event):
        if self.zoom_active and self.zoom_rectangle_start is not None:
            if self.zoom_rectangle_id is not None:
                self.canvas.delete(self.zoom_rectangle_id)
            # Assuming event.x and event.y are already in image coordinates
            self.zoom_rectangle_end = (event.x, event.y)
            x0, y0 = self.zoom_rectangle_start
            x1, y1 = self.zoom_rectangle_end
            self.zoom_rectangle_id = self.canvas.create_rectangle(x0, y0, x1, y1, outline="red", width=2)
            
    def toggle_zoom_mode(self):
        if not self.zoom_active:
            self.zoom_active = True
            self.drawing = False
            self.magic_wand_active = False
            self.erase_active = False
            self.brush_active = False
            self.brush_btn.config(text="Brush")
            self.draw_btn.config(text="Draw")
            self.erase_btn.config(text="Erase")
            self.magic_wand_btn.config(text="Magic Wand")
            self.zoom_btn.config(text="Zoom ON")
            self.canvas.bind("<Button-1>", self.set_zoom_rectangle_start)
            self.canvas.bind("<Button-3>", self.set_zoom_rectangle_end)
            self.canvas.bind("<Motion>", self.update_zoom_box)
        else:
            self.zoom_active = False
            self.zoom_btn.config(text="Zoom")
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<Button-3>")
            self.canvas.unbind("<Motion>")
            self.zoom_rectangle_start = self.zoom_rectangle_end = None
            self.zoom_rectangle_id = None
            self.display_image()
            self.canvas.bind("<Motion>", self.update_mouse_info)
            self.zoom_rectangle_start = None
            self.zoom_rectangle_end = None
            self.zoom_rectangle_id = None
            self.zoom_x0 = None
            self.zoom_y0 = None
            self.zoom_x1 = None
            self.zoom_y1 = None
            self.zoom_mask = None
            self.zoom_image = None
            self.zoom_image_orig = None
            
    def toggle_brush_mode(self):
        self.brush_active = not self.brush_active
        if self.brush_active:
            self.drawing = False
            self.magic_wand_active = False
            self.erase_active = False
            self.brush_btn.config(text="Brush ON")
            self.draw_btn.config(text="Draw")
            self.erase_btn.config(text="Erase")
            self.magic_wand_btn.config(text="Magic Wand")
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<Button-3>")
            self.canvas.unbind("<Motion>")
            self.canvas.bind("<B1-Motion>", self.apply_brush)  # Left click and drag to apply brush
            self.canvas.bind("<B3-Motion>", self.erase_brush)  # Right click and drag to erase with brush
            self.canvas.bind("<ButtonRelease-1>", self.apply_brush_release)  # Left button release
            self.canvas.bind("<ButtonRelease-3>", self.erase_brush_release)  # Right button release
        else:
            self.brush_btn.config(text="Brush")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<B3-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.canvas.unbind("<ButtonRelease-3>")
            
    def toggle_draw_mode(self):
        self.drawing = not self.drawing
        if self.drawing:
            self.magic_wand_active = False
            self.erase_active = False
            self.brush_active = False
            self.draw_btn.config(text="Draw ON")
            self.magic_wand_btn.config(text="Magic Wand")
            self.erase_btn.config(text="Erase")
            self.brush_btn.config(text="Brush")
            
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<Button-3>")
            self.canvas.unbind("<Motion>")
            
            self.canvas.bind("<B1-Motion>", self.draw)
            self.canvas.bind("<ButtonRelease-1>", self.finish_drawing)
        else:
            self.drawing = False
            self.draw_btn.config(text="Draw")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")

    def toggle_magic_wand_mode(self):
        self.magic_wand_active = not self.magic_wand_active
        if self.magic_wand_active:
            self.drawing = False
            self.erase_active = False
            self.brush_active = False
            self.draw_btn.config(text="Draw")
            self.erase_btn.config(text="Erase")
            self.brush_btn.config(text="Brush")
            self.magic_wand_btn.config(text="Magic Wand ON")
            self.canvas.bind("<Button-1>", self.use_magic_wand)
        else:
            self.magic_wand_btn.config(text="Magic Wand")
            self.canvas.unbind("<Button-1>")
            
    def toggle_erase_mode(self):
        self.erase_active = not self.erase_active
        if self.erase_active:
            self.erase_btn.config(text="Erase ON")
            self.canvas.bind("<Button-1>", self.erase_object)
            self.drawing = False
            self.magic_wand_active = False
            self.brush_active = False
            self.draw_btn.config(text="Draw")
            self.magic_wand_btn.config(text="Magic Wand")
            self.brush_btn.config(text="Brush")
        else:
            self.erase_active = False
            self.erase_btn.config(text="Erase")
            self.canvas.unbind("<Button-1>")
            
    def toggle_erase_mode(self):
        self.erase_active = not self.erase_active
        if self.erase_active:
            self.erase_btn.config(text="Erase ON")
            self.brush_active = False
            self.magic_wand_active = False
            self.drawing = False
            self.canvas.bind("<Button-1>", self.erase_object)
            self.draw_btn.config(text="Draw")
            self.brush_btn.config(text="Brush")
            self.magic_wand_btn.config(text="Magic Wand")
        else:
            self.erase_btn.config(text="Erase")
            self.canvas.unbind("<Button-1>")
            
    def apply_brush_release(self, event):
        if hasattr(self, 'brush_path'):
            for x, y, brush_size in self.brush_path:
                img_x, img_y = (x, y) if self.zoom_active else self.canvas_to_image(x, y)
                x0 = max(img_x - brush_size // 2, 0)
                y0 = max(img_y - brush_size // 2, 0)
                x1 = min(img_x + brush_size // 2, self.zoom_mask.shape[1] if self.zoom_active else self.mask.shape[1])
                y1 = min(img_y + brush_size // 2, self.zoom_mask.shape[0] if self.zoom_active else self.mask.shape[0])

                if self.zoom_active:
                    self.zoom_mask[y0:y1, x0:x1] = 255
                    self.update_original_mask_from_zoom()
                else:
                    self.mask[y0:y1, x0:x1] = 255

            del self.brush_path
            self.canvas.delete("temp_line")
            self.update_display()

    def erase_brush_release(self, event):
        if hasattr(self, 'erase_path'):
            for x, y, brush_size in self.erase_path:
                img_x, img_y = (x, y) if self.zoom_active else self.canvas_to_image(x, y)
                x0 = max(img_x - brush_size // 2, 0)
                y0 = max(img_y - brush_size // 2, 0)
                x1 = min(img_x + brush_size // 2, self.zoom_mask.shape[1] if self.zoom_active else self.mask.shape[1])
                y1 = min(img_y + brush_size // 2, self.zoom_mask.shape[0] if self.zoom_active else self.mask.shape[0])

                if self.zoom_active:
                    self.zoom_mask[y0:y1, x0:x1] = 0                    
                    self.update_original_mask_from_zoom()
                else:
                    self.mask[y0:y1, x0:x1] = 0

            del self.erase_path
            self.canvas.delete("temp_line")
            self.update_display()

    def apply_brush(self, event):
        brush_size = int(self.brush_size_entry.get())
        x, y = event.x, event.y

        if not hasattr(self, 'brush_path'):
            self.brush_path = []
            self.last_brush_coord = (x, y)

        self.brush_path.append((x, y, brush_size))
        self.canvas.create_line(self.last_brush_coord[0], self.last_brush_coord[1], x, y, width=brush_size, fill="blue", tag="temp_line")
        self.last_brush_coord = (x, y)

    def erase_brush(self, event):
        brush_size = int(self.brush_size_entry.get())
        x, y = event.x, event.y

        if not hasattr(self, 'erase_path'):
            self.erase_path = []
            self.last_erase_coord = (x, y)

        self.erase_path.append((x, y, brush_size))
        self.canvas.create_line(self.last_erase_coord[0], self.last_erase_coord[1], x, y, width=brush_size, fill="white", tag="temp_line")
        self.last_erase_coord = (x, y)
        
    def update_original_mask_from_zoom(self):
        y0, y1, x0, x1 = self.zoom_y0, self.zoom_y1, self.zoom_x0, self.zoom_x1
        zoomed_mask_resized = resize(self.zoom_mask, (y1 - y0, x1 - x0), order=0, preserve_range=True).astype(np.uint8)
        self.mask[y0:y1, x0:x1] = zoomed_mask_resized

    def erase_object(self, event):
        x, y = event.x, event.y

        if self.zoom_active:
            # Convert canvas coordinates to zoomed image coordinates
            canvas_x, canvas_y = x, y
            zoomed_x = int(canvas_x * (self.zoom_image.shape[1] / self.canvas_width))
            zoomed_y = int(canvas_y * (self.zoom_image.shape[0] / self.canvas_height))

            # Translate zoomed coordinates back to original image coordinates
            orig_x = int(zoomed_x * ((self.zoom_x1 - self.zoom_x0) / self.canvas_width) + self.zoom_x0)
            orig_y = int(zoomed_y * ((self.zoom_y1 - self.zoom_y0) / self.canvas_height) + self.zoom_y0)

            # Check if the point is within the bounds of the original image
            if orig_x < 0 or orig_y < 0 or orig_x >= self.image.shape[1] or orig_y >= self.image.shape[0]:
                print("Point is out of bounds in the original image.")
                return
        else:
            # Use canvas coordinates directly for non-zoom mode
            orig_x, orig_y = x, y

        # Get the label at the click location and erase it from the original mask
        label_to_remove = self.mask[orig_y, orig_x]
        if label_to_remove > 0:
            self.mask[self.mask == label_to_remove] = 0
        self.update_display()
                
    def use_magic_wand(self, event):
        x, y = event.x, event.y
        tolerance = int(self.magic_wand_tolerance.get())
        maximum = int(self.max_pixels_entry.get())  # Retrieve the maximum value

        if self.zoom_active:
            self.magic_wand_zoomed((x, y), tolerance)
        else:
            self.magic_wand_normal((x, y), tolerance)
    
    def apply_magic_wand(self, image, mask, seed_point, tolerance, maximum):
        x, y = seed_point
        initial_value = image[y, x].astype(np.float32)
        visited = np.zeros_like(image, dtype=bool)
        queue = deque([(x, y)])
        added_pixels = 0  # Count of pixels added to the mask
        
        while queue and added_pixels < maximum:
            cx, cy = queue.popleft()
            if visited[cy, cx]:
                continue
            visited[cy, cx] = True
            current_value = image[cy, cx].astype(np.float32)

            if np.linalg.norm(abs(current_value - initial_value)) <= tolerance:
                if mask[cy, cx] == 0:  # Only increment if the pixel was not already in the mask
                    added_pixels += 1
                mask[cy, cx] = 255  # Assign the pixel to the mask regardless

                if added_pixels >= maximum:
                    break  # Stop if the maximum number of pixels for the object is reached

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and not visited[ny, nx]:
                        queue.append((nx, ny))
        return mask
    
    def canvas_to_zoomed_image(self, x_canvas, y_canvas):
        # Calculate the relative position in the zoomed area
        # Assuming self.zoom_x0, self.zoom_y0 are the top-left coordinates of the zoomed area
        rel_x = (x_canvas * self.zoom_scale) + self.zoom_x0
        rel_y = (y_canvas * self.zoom_scale) + self.zoom_y0
        return int(rel_x), int(rel_y)
    
    def magic_wand_normal(self, seed_point, tolerance):
        try:
            maximum = int(self.max_pixels_entry.get())
        except ValueError:
            print("Invalid maximum value; using default of 1000")
            maximum = 1000  
        self.mask = self.apply_magic_wand(self.image, self.mask, seed_point, tolerance, maximum)
        self.display_image()
        
    def magic_wand_zoomed(self, seed_point, tolerance):
        if self.zoom_image_orig is None or self.zoom_mask is None:
            print("Zoomed image or mask not initialized")
            return
        try:
            maximum = int(self.max_pixels_entry.get())
            maximum = maximum * self.zoom_scale
        except ValueError:
            print("Invalid maximum value; using default of 1000")
            maximum = 1000
            
        canvas_x, canvas_y = seed_point
        if canvas_x < 0 or canvas_y < 0 or canvas_x >= self.zoom_image_orig.shape[1] or canvas_y >= self.zoom_image_orig.shape[0]:
            print("Selected point is out of bounds in the zoomed image.")
            return
        self.zoom_mask = self.apply_magic_wand(self.zoom_image_orig, self.zoom_mask, (canvas_x, canvas_y), tolerance, maximum)
        y0, y1, x0, x1 = self.zoom_y0, self.zoom_y1, self.zoom_x0, self.zoom_x1
        zoomed_mask_resized_back = resize(self.zoom_mask, (y1 - y0, x1 - x0), order=0, preserve_range=True).astype(np.uint8)
        self.mask[y0:y1, x0:x1] = np.where(zoomed_mask_resized_back > 0, zoomed_mask_resized_back, self.mask[y0:y1, x0:x1])
        self.display_zoomed_image()
        
    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            if self.draw_coordinates:
                last_x, last_y = self.draw_coordinates[-1]
                self.current_line = self.canvas.create_line(last_x, last_y, x, y, fill="yellow", width=3)
            self.draw_coordinates.append((x, y))
            
    def draw_on_zoomed_mask(self, draw_coordinates):
        canvas_height = self.canvas.winfo_height()
        canvas_width = self.canvas.winfo_width()
        zoomed_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        rr, cc = polygon(np.array(draw_coordinates)[:, 1], np.array(draw_coordinates)[:, 0], shape=zoomed_mask.shape)
        zoomed_mask[rr, cc] = 255
        return zoomed_mask

    def finish_drawing(self, event):
        if len(self.draw_coordinates) > 2:
            self.draw_coordinates.append(self.draw_coordinates[0])
            if self.zoom_active:
                x0, x1, y0, y1 = self.zoom_x0, self.zoom_x1, self.zoom_y0, self.zoom_y1
                zoomed_mask = self.draw_on_zoomed_mask(self.draw_coordinates)
                self.update_original_mask(zoomed_mask, x0, x1, y0, y1)
            else:
                rr, cc = polygon(np.array(self.draw_coordinates)[:, 1], np.array(self.draw_coordinates)[:, 0], shape=self.mask.shape)
                self.mask[rr, cc] = np.maximum(self.mask[rr, cc], 255)
                self.mask = self.mask.copy()
            self.canvas.delete(self.current_line)
            self.draw_coordinates.clear()
            self.update_display()
            
    def finish_drawing_if_active(self, event):
        if self.drawing and len(self.draw_coordinates) > 2:
            self.finish_drawing(event)

    def update_original_mask(self, zoomed_mask, x0, x1, y0, y1):
        actual_mask_region = self.mask[y0:y1, x0:x1]
        target_shape = actual_mask_region.shape
        resized_mask = resize(zoomed_mask, target_shape, order=0, preserve_range=True).astype(np.uint8)
        if resized_mask.shape != actual_mask_region.shape:
            raise ValueError(f"Shape mismatch: resized_mask {resized_mask.shape}, actual_mask_region {actual_mask_region.shape}")
        self.mask[y0:y1, x0:x1] = np.maximum(actual_mask_region, resized_mask)
        self.mask = self.mask.copy()
        self.mask[y0:y1, x0:x1] = np.maximum(self.mask[y0:y1, x0:x1], resized_mask)
        self.mask = self.mask.copy()
            
    def apply_normalization(self):
        self.update_display()
            
    def update_normalized_image_old(self, *args):
        lower_quantile = float(self.lower_quantile.get()) if self.lower_quantile.get() != '' else 1.0
        upper_quantile = float(self.upper_quantile.get()) if self.upper_quantile.get() != '' else 99.0
        self.norm_image = self.normalize_image(self.image, lower_quantile, upper_quantile)  
        self.update_display()

    def fill_objects(self):
        binary_mask = self.mask > 0
        filled_mask = binary_fill_holes(binary_mask)
        self.mask = filled_mask.astype(np.uint8) * 255
        labeled_mask, _ = label(filled_mask)
        self.mask = labeled_mask
        self.update_display()

    def relabel_objects(self):
        mask = self.mask
        labeled_mask, num_labels = label(mask > 0)
        self.mask = labeled_mask
        self.update_display()
        
    def clear_objects(self):
        self.mask = np.zeros_like(self.mask)
        self.update_display()

    def invert_mask(self):
        self.mask = np.where(self.mask > 0, 0, 1)
        self.relabel_objects()
        self.update_display()
