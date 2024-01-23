import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
import imageio.v2 as imageio
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.draw import polygon
from scipy.ndimage import label
from collections import deque
from scipy.ndimage import binary_fill_holes, label
import random

class ImageEditor:
    def __init__(self, root, folder_path, window_width, window_height, scale_factor):
        # Initialize main attributes first
        self.root = root
        self.folder_path = folder_path
        self.scale_factor = scale_factor
        self.current_image_index = 0
        self.images = self.load_images(folder_path)
        self.masks = [np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16) for image in self.images]

        # Initialize Canvas
        self.window_width = window_width
        self.window_height = window_height
        self.canvas = tk.Canvas(root, width=window_width, height=window_height)
        self.canvas.pack()

        # Additional attributes
        self.draw_coordinates = []
        self.redraw_needed = False
        self.eraser_active = False  
        self.current_resized_image = None  
        self.current_resized_mask = None
        self.setup_toolbar()
        
        # Display the first image
        self.display_image()
        
    def load_images(self, folder_path):
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        images = []
        self.image_filenames = []  # Store filenames
        masks = []  # Initialize masks list
        masks_folder = os.path.join(folder_path, 'masks')  # Path to the masks subdirectory

        for file in image_files:
            image_path = os.path.join(folder_path, file)
            mask_path = os.path.join(masks_folder, file)  # Define mask_path here
            self.image_filenames.append(file)  # Add filename
            image = imageio.imread(image_path)
            images.append(image)

            # Check if the mask exists in the 'masks' folder
            if os.path.exists(mask_path):
                mask = imageio.imread(mask_path)
                mask = mask.astype(np.uint8)
                masks.append(mask)
                print(f"Loaded mask for {file} - shape: {mask.shape}, image shape: {image.shape} unique values: {np.unique(mask)}")
            else:
                masks.append(np.zeros(image.shape[:2], dtype=np.uint8))  # Create a zero-filled mask

        self.masks = masks
        return images
        
    def apply_quantiles(self):
        self.display_image()

    def normalize_image(self, image, lower_quantile, upper_quantile):
        lower_bound = np.percentile(image, lower_quantile)
        upper_bound = np.percentile(image, upper_quantile)
        normalized = np.clip(image, lower_bound, upper_bound)
        normalized = (normalized - lower_bound) / (upper_bound - lower_bound)
        max_value = np.iinfo(image.dtype).max
        normalized = (normalized * max_value).astype(image.dtype)
        return normalized
    
    def resize_image(self, image, scale_factor):
        original_dtype = image.dtype
        resized_image = resize(image, 
                               (int(image.shape[0] * scale_factor), int(image.shape[1] * scale_factor)), 
                               anti_aliasing=True, preserve_range=True)
        return resized_image.astype(original_dtype)
    
    def resize_mask(self, mask, image, scale_factor):
        original_dtype = mask.dtype
        resized_mask = resize(mask, 
                               (int(image.shape[0] * scale_factor), int(image.shape[1] * scale_factor)), 
                               anti_aliasing=True, preserve_range=True)
        return resized_mask.astype(original_dtype)
    
    def resize_current_image(self):
        if self.current_image_index < len(self.images):
            original_image = self.images[self.current_image_index]
            self.current_resized_image = self.resize_image(original_image, self.scale_factor)
    
    def resize_current_mask(self, image):
        if self.current_image_index < len(self.images):
            original_mask = self.masks[self.current_image_index]
            self.current_resized_mask = self.resize_mask(original_mask, image, self.scale_factor)
    
    def display_image(self):
        if self.current_image_index < len(self.images):
            self.resize_current_image()  # Resize the current image
            image = self.current_resized_image
            
            self.resize_current_mask(image)  # Resize the current mask
            mask = self.current_resized_mask
            print(mask.shape)
            
            #mask = self.masks[self.current_image_index]
            lower_quantile = float(self.lower_quantile_entry.get())
            upper_quantile = float(self.upper_quantile_entry.get())
            normalized_image = self.normalize_image(image, lower_quantile, upper_quantile)
            scaled_image = resize(normalized_image, (self.window_height, self.window_width), preserve_range=True)

            overlay = self.overlay_mask_on_image(scaled_image, mask)
            self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(overlay))
            self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)

    def redraw_lines(self):
        for i in range(1, len(self.draw_coordinates)):
            x0, y0 = self.draw_coordinates[i - 1]
            x1, y1 = self.draw_coordinates[i]
            self.canvas.create_line(x0, y0, x1, y1, fill="yellow", width=3)

    def setup_toolbar(self):
        toolbar = tk.Frame(self.root)
        toolbar.pack(side='top', fill='x')
        self.draw_btn = tk.Button(toolbar, text="Draw", command=self.freehand_draw)
        self.draw_btn.pack(side='left')
        self.wand_btn = tk.Button(toolbar, text="Magic Wand", command=self.activate_magic_wand)
        self.wand_btn.pack(side='left')
        save_btn = tk.Button(toolbar, text="Save Mask", command=self.save_mask)
        save_btn.pack(side='left')
        next_btn = tk.Button(toolbar, text="Next Image", command=self.next_image)
        next_btn.pack(side='left')
        self.tolerance_entry = tk.Entry(toolbar)
        self.tolerance_entry.insert(0, "1000")
        self.tolerance_entry.pack(side='left')
        self.drawing = False
        self.magic_wand_active = False
        fill_btn = tk.Button(toolbar, text="Fill Objects", command=self.fill_objects)
        fill_btn.pack(side='left')
        relabel_btn = tk.Button(toolbar, text="Relabel", command=self.relabel_objects)
        relabel_btn.pack(side='left')
        erase_btn = tk.Button(toolbar, text="Erase Object", command=self.activate_erase_object_mode)
        erase_btn.pack(side='left')
        
        # Quantile entries
        self.lower_quantile_entry = tk.Entry(toolbar)
        self.lower_quantile_entry.insert(0, "2")  # default lower quantile
        self.lower_quantile_entry.pack(side='left')
        self.upper_quantile_entry = tk.Entry(toolbar)
        self.upper_quantile_entry.insert(0, "99.99")  # default upper quantile
        self.upper_quantile_entry.pack(side='left')
        apply_quantiles_btn = tk.Button(toolbar, text="Apply Quantiles", command=self.apply_quantiles)
        apply_quantiles_btn.pack(side='left')
        clear_btn = tk.Button(toolbar, text="Clear Mask", command=self.clear_mask)
        clear_btn.pack(side='left')
        invert_mask_btn = tk.Button(toolbar, text="Invert Mask", command=self.invert_mask)
        invert_mask_btn.pack(side='left')

    def relabel_objects(self):
        mask = self.masks[self.current_image_index]
        labeled_mask, num_labels = label(mask > 0)
        self.masks[self.current_image_index] = labeled_mask
        self.display_image()

    def flood_fill(self, x, y, label, visited):
        to_fill = deque([(x, y)])
        while to_fill:
            x, y = to_fill.popleft()
            if x < 0 or x >= len(self.masks[0][0]) or y < 0 or y >= len(self.masks[0]) or visited[y][x] or self.masks[self.current_image_index][y][x] == 0:
                continue
            visited[y][x] = True
            self.masks[self.current_image_index][y][x] = label
            to_fill.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
            
    def clear_mask(self):
        self.masks[self.current_image_index] = np.zeros_like(self.masks[self.current_image_index])
        self.display_image()

    def invert_mask(self):
        mask = self.masks[self.current_image_index]
        self.masks[self.current_image_index] = np.where(mask > 0, 0, 1).astype(np.uint16)
        self.display_image()

    def erase_object_mode(self, event):
        x, y = self.canvas_to_image(event.x, event.y)
        if self.masks[self.current_image_index][y][x] > 0:
            self.erase_object(self.masks[self.current_image_index][y][x])
            self.display_image()

    def erase_object(self, label):
        self.masks[self.current_image_index][self.masks[self.current_image_index] == label] = 0

    def fill_objects(self):
        # Fill holes in the binary mask
        binary_mask = self.masks[self.current_image_index] > 0
        filled_mask = binary_fill_holes(binary_mask)

        # Update the original mask
        self.masks[self.current_image_index] = filled_mask.astype(np.uint16) * 65535
        
        # Relabel the mask
        labeled_mask, _ = label(filled_mask)
        self.masks[self.current_image_index] = labeled_mask
        
        # Update the displayed image
        self.display_image()

    def overlay_mask_on_image(self, image, mask):
        if len(image.shape) == 2:  # Grayscale image
            image = np.stack((image,) * 3, axis=-1)  # Convert to RGB

        # Ensure mask is in integer format
        mask = mask.astype(np.int32)
        max_label = np.max(mask)

        np.random.seed(0)  # Optional: for consistent colors across different calls
        colors = np.random.randint(0, 255, size=(max_label + 1, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # Background color

        # Map the labels to colors
        colored_mask = colors[mask]

        image_8bit = (image / 256).astype(np.uint8)
        combined_image = np.where(mask[..., None] > 0, colored_mask, image_8bit)
        return combined_image
    
    def overlay_mask_on_image_broken(self, image, mask):
        if len(image.shape) == 2:  # Grayscale image
            image = np.stack((image,)*3, axis=-1)  # Convert to RGB

        unique_labels = np.unique(mask)
        label_colors = {label: [random.randint(0, 255) for _ in range(3)] for label in unique_labels if label != 0}

        colored_mask = np.zeros_like(image, dtype=np.uint8)
        for label, color in label_colors.items():
            mask_indices = mask == label
            colored_mask[mask_indices] = color

        image_8bit = (image / 256).astype(np.uint8)
        combined_image = np.where(mask[..., None] > 0, colored_mask, image_8bit)
        return combined_image

    def activate_erase_object_mode(self):
        # Bind the canvas click event to the erase_object_mode function
        self.canvas.bind("<Button-1>", self.erase_object_mode)

    def freehand_draw(self):
        self.drawing = not self.drawing
        if self.drawing:
            self.canvas.bind("<B1-Motion>", self.draw)
            self.canvas.bind("<ButtonRelease-1>", self.create_object)
            self.canvas.focus_set()  # Focus on the canvas
            self.draw_btn.config(text="Draw ON")
        else:
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.draw_btn.config(text="Draw")
    
    def draw(self, event):
        x, y = event.x, event.y
        if self.draw_coordinates:
            last_x, last_y = self.draw_coordinates[-1]
            self.canvas.create_line(last_x, last_y, x, y, fill="yellow", width=3)
        self.draw_coordinates.append((x, y))

    def create_object(self, event):
        if len(self.draw_coordinates) > 2:
            # Close the polygon in the mask
            mask = self.masks[self.current_image_index]
            self.draw_coordinates.append(self.draw_coordinates[0])
            draw_polygon = np.array([self.canvas_to_image(*point) for point in self.draw_coordinates])
            rr, cc = polygon(draw_polygon[:, 1], draw_polygon[:, 0], shape=mask.shape)
            mask[rr, cc] = 65535
            self.draw_coordinates.clear()
    
            # Redraw the image to clear the lines and update the mask
            self.display_image()

    def finish_drawing(self, event):
        self.display_image()

    def canvas_to_image(self, x, y):
        # Calculate the scaling factor between the displayed image and the original image
        original_width, original_height = self.images[self.current_image_index].shape[1], self.images[self.current_image_index].shape[0]
        displayed_width, displayed_height = self.window_width, self.window_height
        scale_x = original_width / displayed_width
        scale_y = original_height / displayed_height
        # Adjust coordinates based on the scale factor
        return int(x * scale_x), int(y * scale_y)

    def activate_magic_wand(self):
        self.magic_wand_active = not self.magic_wand_active
        if self.magic_wand_active:
            self.wand_btn.config(text="Magic Wand ON")
            self.canvas.bind("<Button-1>", lambda event: self.on_canvas_click(event, select=True))
            self.canvas.bind("<Button-3>", lambda event: self.on_canvas_click(event, select=False))
        else:
            self.wand_btn.config(text="Magic Wand")
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<Button-3>")
            
    def activate_erase_mode(self):
        self.eraser_active = not self.eraser_active
        if self.eraser_active:
            self.eraser_btn.config(text="Eraser ON")
            self.canvas.bind("<B1-Motion>", self.erase_area)
        else:
            self.eraser_btn.config(text="Eraser")
            self.canvas.unbind("<B1-Motion>")

    def deferred_display_update(self):
        self.display_image()
        self.deferred_update_active = False

    def on_canvas_click(self, event, select=True):
        tolerance = int(self.tolerance_entry.get())
        x, y = self.canvas_to_image(event.x, event.y)
        if select:
            self.magic_wand((x, y), tolerance, select=True)
        else:
            self.magic_wand((x, y), tolerance, select=False)
        
    def magic_wand(self, seed_point, tolerance, select=True):
        x, y = seed_point
        image = self.images[self.current_image_index]
        mask = self.masks[self.current_image_index]
        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            return
        initial_value = image[y, x]
        visited = np.zeros_like(image, dtype=bool)
        queue = deque([(x, y)])
        while queue:
            cx, cy = queue.popleft()
            if visited[cy, cx]:
                continue
            visited[cy, cx] = True
            current_value = image[cy, cx]
            if abs(int(current_value) - int(initial_value)) <= tolerance:
                mask[cy, cx] = 65535 if select else 0
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and not visited[ny, nx]:
                        queue.append((nx, ny))
        self.display_image()

    def save_mask(self):
        mask = self.masks[self.current_image_index]
        original_size = self.images[self.current_image_index].shape[:2]
        resized_mask = resize(mask, original_size, preserve_range=True).astype(np.uint16)
        save_folder = os.path.join(self.folder_path, 'new_masks')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Use the same filename as the original image
        image_filename = self.image_filenames[self.current_image_index]
        save_path = os.path.join(save_folder, image_filename)  # Save with the same filename

        imageio.imwrite(save_path, resized_mask)
        self.next_image()

    def next_image(self):
        if self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            self.display_image()
            self.draw_coordinates.clear()
            
# Folder path, scale factor, and window size as arguments
folder_path = "/mnt/training_data_cellpose/test/"
scale_factor = 1
window_width = 1996
window_height = 1996

root = tk.Tk()
app = ImageEditor(root, folder_path, window_width, window_height, scale_factor)
root.mainloop()
