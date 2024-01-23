import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
import matplotlib.pyplot as plt
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
        #self.masks = [np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16) for image in self.images]

        # Initialize Canvas
        self.window_width = window_width
        self.window_height = window_height
        self.canvas = tk.Canvas(root, width=window_width, height=window_height)
        self.canvas.pack()

        # Additional attributes
        self.draw_coordinates = []
        self.redraw_needed = False
        self.erase_active = False  
        self.current_resized_image = None  
        self.current_resized_mask = None
        self.setup_toolbar()
        
        # For Zoom
        self.zoom_active = False
        self.original_image = None
        self.original_mask = None
        self.zoomed_image = None
        self.zoomed_mask = None
        self.zoom_applied = False
        self.zoom_rectangle_start = None
        
        # Display the first image
        self.display_image()

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
            image_shape = image.shape

            # Check if the mask exists in the 'masks' folder
            if os.path.exists(mask_path):
                mask = imageio.imread(mask_path)
                mask = mask.astype(np.int32)
                masks.append(mask)
            else:
                masks.append(np.zeros(image.shape[:2], dtype=np.uint16))  # Create a zero-filled mask

        self.masks = masks
        return images
    
    def resize_array(self, array, scale_factor):
        original_dtype = array.dtype
        resized_array = resize(array, 
                               (int(array.shape[0] * scale_factor), int(array.shape[1] * scale_factor)), 
                               anti_aliasing=True, preserve_range=True)
        return resized_array.astype(original_dtype)
    
    def resize_current_image_and_mask(self):
        if self.current_image_index < len(self.images):
            
            # Resize the current image
            original_image = self.images[self.current_image_index]
            self.current_resized_image = self.resize_array(original_image, self.scale_factor)

            # Resize the mask using nearest-neighbor interpolation
            original_mask = self.masks[self.current_image_index]            
            resized_mask_dimensions = (self.current_resized_image.shape[0], self.current_resized_image.shape[1])
            self.current_resized_mask = resize(original_mask, resized_mask_dimensions, order=0, anti_aliasing=False, preserve_range=True).astype(original_mask.dtype)
            
    def display_image_no_zoom(self):
        if self.current_image_index < len(self.images):
            
            # Resize image and mask for processing
            self.resize_current_image_and_mask()  
            image = self.current_resized_image
            mask = self.current_resized_mask
            
            # Normalize the image
            lower_quantile = float(self.lower_quantile_entry.get())
            upper_quantile = float(self.upper_quantile_entry.get())
            normalized_image = self.normalize_image(image, lower_quantile, upper_quantile)

            # Resize normalized image for display
            display_size_image = resize(normalized_image, (self.window_height, self.window_width), anti_aliasing=True, preserve_range=True).astype(normalized_image.dtype)

            # Resize mask using nearest-neighbor interpolation for display
            display_size_mask = resize(mask, (self.window_height, self.window_width), order=0, anti_aliasing=False, preserve_range=True).astype(mask.dtype)
            overlay = self.overlay_mask_on_image(display_size_image, display_size_mask)
            self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(overlay))
            self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)
            
    def display_image(self):
        # Ensure there is an image to display
        if self.current_image_index < len(self.images):
            self.resize_current_image_and_mask()

        if self.zoom_active and self.zoomed_image is not None:
            # Display the zoomed image
            image_to_display = self.zoomed_image
            mask_to_display = self.zoomed_mask
        else:
            # Display the regular image
            image_to_display = self.current_resized_image
            mask_to_display = self.current_resized_mask

        # Normalize the image
        lower_quantile = float(self.lower_quantile_entry.get())
        upper_quantile = float(self.upper_quantile_entry.get())
        normalized_image = self.normalize_image(image_to_display, lower_quantile, upper_quantile)

        # Resize normalized image for display
        display_size_image = resize(normalized_image, (self.window_height, self.window_width), anti_aliasing=True, preserve_range=True).astype(normalized_image.dtype)

        # Resize mask using nearest-neighbor interpolation for display
        display_size_mask = resize(mask_to_display, (self.window_height, self.window_width), order=0, anti_aliasing=False, preserve_range=True).astype(mask_to_display.dtype)

        # Overlay mask on image
        overlay = self.overlay_mask_on_image(display_size_image, display_size_mask)

        # Update the canvas
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
        self.erase_btn = tk.Button(toolbar, text="Erase Object", command=self.activate_erase_object_mode)
        self.erase_btn.pack(side='left')
        
        self.zoom_btn = tk.Button(toolbar, text="Zoom", command=self.activate_zoom_mode)
        self.zoom_btn.pack(side='left')
        
        # Quantile entries
        self.lower_quantile_entry = tk.Entry(toolbar)
        self.lower_quantile_entry.insert(0, "2") 
        self.lower_quantile_entry.pack(side='left')
        self.upper_quantile_entry = tk.Entry(toolbar)
        self.upper_quantile_entry.insert(0, "99.99")
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

    def erase_object(self, label):
        self.masks[self.current_image_index][self.masks[self.current_image_index] == label] = 0
        
    def erase_object_mode(self, event):
        x, y = self.canvas_to_image(event.x, event.y)
        if self.masks[self.current_image_index][y][x] > 0:
            self.erase_object(self.masks[self.current_image_index][y][x])
            self.display_image()

    def fill_objects(self):
        # Fill holes in the binary mask
        binary_mask = self.masks[self.current_image_index] > 0
        filled_mask = binary_fill_holes(binary_mask)
        self.masks[self.current_image_index] = filled_mask.astype(np.uint16) * 65535
        labeled_mask, _ = label(filled_mask)
        self.masks[self.current_image_index] = labeled_mask
        self.display_image()

    def overlay_mask_on_image(self, image, mask):
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)  
        mask = mask.astype(np.int32)
        max_label = np.max(mask)
        np.random.seed(0)
        colors = np.random.randint(0, 255, size=(max_label + 1, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]
        colored_mask = colors[mask]
        image_8bit = (image / 256).astype(np.uint8)
        combined_image = np.where(mask[..., None] > 0, colored_mask, image_8bit)
        return combined_image

    def activate_erase_object_mode(self):
        self.erase_active = not self.erase_active
        if self.erase_active:
            self.disable_other_modes('erase')
            self.erase_btn.config(text="Erase Object ON")
            self.canvas.bind("<Button-1>", self.erase_object_mode)
        else:
            self.erase_btn.config(text="Erase Object") 
            
    def activate_magic_wand(self):
        self.magic_wand_active = not self.magic_wand_active
        if self.magic_wand_active:
            self.disable_other_modes('wand')
            self.wand_btn.config(text="Magic Wand ON")
            self.canvas.bind("<Button-1>", lambda event: self.on_canvas_click(event, select=True))
            self.canvas.bind("<Button-3>", lambda event: self.on_canvas_click(event, select=False))
        else:
            self.wand_btn.config(text="Magic Wand")
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<Button-3>")
            
    def freehand_draw(self):
        self.drawing = not self.drawing
        if self.drawing:
            self.disable_other_modes('draw')
            self.canvas.bind("<B1-Motion>", self.draw)
            self.canvas.bind("<ButtonRelease-1>", self.create_object)
            self.canvas.focus_set()  # Focus on the canvas
            self.draw_btn.config(text="Draw ON")
        else:
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.draw_btn.config(text="Draw")

    def disable_other_modes(self, active_mode):
        if active_mode != 'draw':
            self.drawing = False
            self.draw_btn.config(text="Draw")
        if active_mode != 'wand':
            self.magic_wand_active = False
            self.wand_btn.config(text="Magic Wand")
        if active_mode != 'erase':
            self.erase_active = False
            self.erase_btn.config(text="Erase Object")
            
    def activate_zoom_mode(self):
        self.zoom_active = not self.zoom_active
        if self.zoom_active:
            self.disable_other_modes('zoom')
            self.zoom_btn.config(text="Zoom ON")
            self.canvas.bind("<Button-1>", self.set_zoom_rectangle_start)
            if not self.zoom_applied:
                self.canvas.bind("<Motion>", self.update_zoom_rectangle)
        else:
            self.zoom_btn.config(text="Zoom")
            self.return_to_original_image()
            self.canvas.unbind("<Motion>")

    def perform_zoom(self):
        if not (self.zoom_rectangle_start and self.zoom_rectangle_end) or self.zoom_applied:
            print("Zoom rectangle not defined or already zoomed")
            return
        x0, y0 = self.zoom_rectangle_start
        x1, y1 = self.zoom_rectangle_end
        # Ensure coordinates are within the image bounds
        x0 = max(0, min(x0, self.current_resized_image.shape[1] - 1))
        x1 = max(0, min(x1, self.current_resized_image.shape[1] - 1))
        y0 = max(0, min(y0, self.current_resized_image.shape[0] - 1))
        y1 = max(0, min(y1, self.current_resized_image.shape[0] - 1))
        # Validate zoom rectangle dimensions
        if x0 == x1 or y0 == y1:
            print("Invalid zoom rectangle dimensions")
            return
        self.original_image = self.current_resized_image.copy()
        self.original_mask = self.current_resized_mask.copy()
        self.zoomed_image = self.current_resized_image[y0:y1, x0:x1]
        self.zoomed_mask = self.current_resized_mask[y0:y1, x0:x1]
        self.zoom_applied = True
        self.display_image()
        
    def return_to_original_image(self):
        # Return to the original image and mask if they exist
        if self.original_image is not None and self.original_mask is not None:
            self.current_resized_image = self.original_image
            self.current_resized_mask = self.original_mask
            self.display_image()

    def draw(self, event):
        x, y = event.x, event.y
        if self.draw_coordinates:
            last_x, last_y = self.draw_coordinates[-1]
            self.canvas.create_line(last_x, last_y, x, y, fill="yellow", width=3)
        self.draw_coordinates.append((x, y))

    def finish_drawing(self, event):
        self.display_image()
        
    def update_original_mask_from_zoomed(self):
        if self.zoomed_mask is not None and self.original_mask is not None:
            # Get zoomed area dimensions
            x0, y0 = self.zoom_rectangle_start
            x1, y1 = self.zoom_rectangle_end

            # Calculate the original dimensions of the zoomed area
            original_zoomed_area_width = x1 - x0
            original_zoomed_area_height = y1 - y0

            # Resize zoomed mask to the size of the zoomed area in the original mask
            resized_zoomed_mask = resize(self.zoomed_mask, 
                                         (original_zoomed_area_height, original_zoomed_area_width),
                                         order=0, anti_aliasing=False, preserve_range=True).astype(np.uint8)

            # Update the relevant section of the original mask
            self.masks[self.current_image_index][y0:y1, x0:x1] = resized_zoomed_mask
            
    def set_zoom_rectangle_start(self, event):
        if self.zoom_active and not self.zoom_applied:
            self.zoom_rectangle_start = self.canvas_to_image(event.x, event.y)
            self.canvas.bind("<Motion>", self.update_zoom_rectangle)

    def update_zoom_rectangle(self, event):
        if self.zoom_rectangle_start is not None and not self.zoom_applied:
            # Dynamically calculate the end point to maintain aspect ratio
            start_x, start_y = self.zoom_rectangle_start
            aspect_ratio = self.current_resized_image.shape[1] / self.current_resized_image.shape[0]

            # Calculate width and height based on aspect ratio
            current_width = abs(event.x - start_x)
            current_height = current_width / aspect_ratio
            if event.y < start_y:
                end_y = start_y - current_height
            else:
                end_y = start_y + current_height

            # Update canvas with red rectangle
            self.canvas.delete("zoom_rect")  # Remove previous rectangle
            self.canvas.create_rectangle(start_x, start_y, event.x, end_y, outline="red", tag="zoom_rect")

    def set_zoom_rectangle_end(self, event):
        if self.zoom_active and not self.zoom_applied:
            # Use the dynamically calculated end point
            start_x, start_y = self.zoom_rectangle_start
            aspect_ratio = self.current_resized_image.shape[1] / self.current_resized_image.shape[0]
            current_width = abs(event.x - start_x)
            current_height = current_width / aspect_ratio
            if event.y < start_y:
                end_y = start_y - current_height
            else:
                end_y = start_y + current_height

            self.zoom_rectangle_end = self.canvas_to_image(event.x, end_y)
            self.perform_zoom()
            self.canvas.unbind("<Motion>")
            self.zoom_applied = True

    def zoom_canvas_to_image(self, x, y):
        x0, y0 = self.zoom_rectangle_start
        x1, y1 = self.zoom_rectangle_end
        zoom_width_original, zoom_height_original = x1 - x0, y1 - y0
        display_ratio = min(self.window_width / zoom_width_original, self.window_height / zoom_height_original)
        zoom_width_display = zoom_width_original * display_ratio
        zoom_height_display = zoom_height_original * display_ratio
        offset_x = (self.window_width - zoom_width_display) / 2 if self.window_width > zoom_width_display else 0
        offset_y = (self.window_height - zoom_height_display) / 2 if self.window_height > zoom_height_display else 0
        adjusted_x = x - offset_x
        adjusted_y = y - offset_y
        scale_x = zoom_width_original / zoom_width_display
        scale_y = zoom_height_original / zoom_height_display
        translated_x = x0 + (adjusted_x * scale_x)
        translated_y = y0 + (adjusted_y * scale_y)

        return int(translated_x), int(translated_y)
            
    def create_object(self, event):
        print("Zoom Mode Active:", self.zoom_active)
        if len(self.draw_coordinates) > 2:
            self.draw_coordinates.append(self.draw_coordinates[0])
            mask = self.current_resized_mask if self.zoom_active else self.masks[self.current_image_index]

            if self.zoom_active:
                draw_polygon = np.array([self.zoom_canvas_to_image(x, y) for x, y in self.draw_coordinates])
            else:
                draw_polygon = np.array([self.canvas_to_image(x, y) for x, y in self.draw_coordinates])
                rr, cc = polygon(draw_polygon[:, 1], draw_polygon[:, 0], shape=mask.shape)
            
            mask[rr, cc] = 65535
            self.draw_coordinates.clear()
            if self.zoom_active:
                self.update_original_mask_from_zoomed()
            self.display_image()
            
    def canvas_to_image(self, x, y):
        # Regular canvas to image coordinates
        scale_x = self.current_resized_image.shape[1] / self.window_width
        scale_y = self.current_resized_image.shape[0] / self.window_height
        return int(x * scale_x), int(y * scale_y)

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
window_width = 2000
window_height = 2000

root = tk.Tk()
app = ImageEditor(root, folder_path, window_width, window_height, scale_factor)
root.mainloop()
