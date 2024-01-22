import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
import imageio.v2 as imageio
from skimage.transform import resize
from skimage.draw import polygon
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

         # Attributes for zooming
        self.zoom_mode = False
        self.zoom_box_start = None
        self.zoom_box_end = None
        self.original_image = None  # To store the full-size image

        # Bind additional mouse events for zooming
        self.canvas.bind("<ButtonPress-3>", self.start_zoom)  # Right-click to start zoom box
        self.canvas.bind("<B3-Motion>", self.draw_zoom_box)   # Drag to draw the zoom box
        self.canvas.bind("<ButtonRelease-3>", self.perform_zoom)  # Release to zoom

        # Initialize Drawing Attributes
        self.draw_coordinates = []
        self.redraw_needed = False

        # Set up the toolbar after initializing the necessary attributes
        self.setup_toolbar()
        
        # Display the first image
        self.display_image()

    # Start drawing the zoom box
    def start_zoom(self, event):
        self.zoom_mode = True
        self.zoom_box_start = (event.x, event.y)

    # Draw the zoom box
    def draw_zoom_box(self, event):
        if self.zoom_mode:
            self.zoom_box_end = (event.x, event.y)
            self.canvas.delete("zoom_box")  # Remove previous zoom box
            self.canvas.create_rectangle(self.zoom_box_start[0], self.zoom_box_start[1],
                                         self.zoom_box_end[0], self.zoom_box_end[1],
                                         outline="red", tag="zoom_box")

    # Perform the zoom action
    def perform_zoom(self, event):
        if self.zoom_mode and self.original_image is not None:
            x1, y1 = self.zoom_box_start
            x2, y2 = self.zoom_box_end
            # Coordinates adjustment based on the original image
            zoomed_image = self.original_image.crop((x1, y1, x2, y2))
            self.display_zoomed_image(zoomed_image)
            self.zoom_mode = False

    # Display the zoomed image
    def display_zoomed_image(self, zoomed_image):
        self.tk_zoomed_image = ImageTk.PhotoImage(image=zoomed_image)
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_zoomed_image)

    def load_images(self, folder_path):
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        images = [imageio.imread(os.path.join(folder_path, file)) for file in image_files]
        return images

    def display_image(self):
        if self.current_image_index < len(self.images):
            image = self.images[self.current_image_index]
            mask = self.masks[self.current_image_index]
            # Resize image and mask
            scaled_image = resize(image, (self.window_height, self.window_width), preserve_range=True).astype(np.uint16)
            scaled_mask = resize(mask, (self.window_height, self.window_width), preserve_range=True).astype(np.uint16)
            # Overlay mask on image
            overlay = self.overlay_mask_on_image(scaled_image, scaled_mask)
            # Convert to Tkinter format and display
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

    def relabel_objects(self):
        label = 1
        visited = np.zeros_like(self.masks[self.current_image_index], dtype=bool)
        for y in range(len(self.masks[0])):
            for x in range(len(self.masks[0][0])):
                if self.masks[self.current_image_index][y][x] > 0 and not visited[y][x]:
                    self.flood_fill(x, y, label, visited)
                    label += 1
        self.display_image()

    def flood_fill(self, x, y, label, visited):
        # Iterative flood fill
        to_fill = deque([(x, y)])
        while to_fill:
            x, y = to_fill.popleft()
            if x < 0 or x >= len(self.masks[0][0]) or y < 0 or y >= len(self.masks[0]) or visited[y][x] or self.masks[self.current_image_index][y][x] == 0:
                continue
            visited[y][x] = True
            self.masks[self.current_image_index][y][x] = label
            to_fill.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

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
        # Updated method to handle colored overlay for different labels
        if len(image.shape) == 2:  # Grayscale image
            image = np.stack((image,)*3, axis=-1)  # Convert to RGB

        unique_labels = np.unique(mask)
        colored_mask = np.zeros_like(image, dtype=np.uint8)

        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            mask_indices = mask == label
            random_color = [random.randint(0, 255) for _ in range(3)]
            colored_mask[mask_indices] = random_color

        image_8bit = (image / 256).astype(np.uint8)
        combined_image = np.where(mask[..., None], colored_mask, image_8bit)
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
        to_check = deque([(x, y)])
        checked = set()
        while to_check:
            x, y = to_check.popleft()
            if (x, y) in checked or not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
                continue
            checked.add((x, y))
            current_value = image[y, x]
            if abs(int(current_value) - int(initial_value)) <= tolerance:
                mask[y, x] = 65535 if select else 0  # Set mask to 0 for deselecting
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                        to_check.append((nx, ny))
        self.display_image()

    def save_mask(self):
        mask = self.masks[self.current_image_index]
        original_size = self.images[self.current_image_index].shape[:2]
        resized_mask = resize(mask, original_size, preserve_range=True).astype(np.uint16)
        # Create 'new_masks' folder if it doesn't exist
        save_folder = os.path.join(self.folder_path, 'new_masks')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, f"mask_{self.current_image_index}.png")
        imageio.imwrite(save_path, resized_mask)
        self.next_image()  # Automatically load the next image after saving

    def next_image(self):
        if self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            self.display_image()
            self.draw_coordinates.clear()
