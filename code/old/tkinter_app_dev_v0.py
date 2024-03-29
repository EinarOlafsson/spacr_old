{\rtf1\ansi\ansicpg1252\cocoartf2708
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import os\
import numpy as np\
import tkinter as tk\
import imageio.v2 as imageio\
from collections import deque\
from tkinter import filedialog\
from PIL import Image, ImageTk\
from skimage.draw import polygon\
from skimage.transform import resize\
from skimage.measure import label as sk_label\
from scipy.ndimage import binary_fill_holes, label\
\
\
class modify_masks:\
    def __init__(self, root, folder_path, scale_factor, width, height):\
        self.root = root\
        self.folder_path = folder_path\
        self.scale_factor = scale_factor\
        self.image_filenames = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])\
        self.masks_folder = os.path.join(folder_path, 'masks')\
        self.current_image_index = 0\
        self.canvas_width = width\
        self.canvas_height = height\
\
        # Initialize these before the first call to load_image_and_mask\
        self.zoom_rectangle_start = None\
        self.zoom_rectangle_end = None\
        self.zoom_rectangle_id = None\
        self.zoom_x0 = None\
        self.zoom_y0 = None\
        self.zoom_x1 = None\
        self.zoom_y1 = None\
        self.root.bind('<Return>', self.apply_zoom_on_enter)\
        \
        # Initialize mode flags\
        self.drawing = False\
        self.zoom_active = False\
        self.magic_wand_active = False\
        \
        # Initialize percentile values\
        self.lower_quantile = tk.StringVar(value="1.0")\
        self.upper_quantile = tk.StringVar(value="99.0")\
        \
        self.image, self.mask = self.load_image_and_mask(self.current_image_index)\
        self.original_size = self.image.shape\
        \
        self.image, self.mask = self.resize_arrays(self.image,self.mask)\
        \
        self.setup_navigation_toolbar()\
        \
        # Initialize the canvas\
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height)\
        self.canvas.pack()\
        \
        self.magic_wand_tolerance = tk.StringVar(value="10")\
        \
        self.setup_mode_toolbar()\
        self.setup_function_toolbar()\
        self.setup_zoom_toolbar()\
        \
        self.draw_coordinates = []\
        self.canvas.bind("<B1-Motion>", self.draw)\
        self.canvas.bind("<ButtonRelease-1>", self.finish_drawing_if_active)\
        self.display_image()\
\
    def apply_zoom_on_enter(self, event):\
        if self.zoom_active and self.zoom_rectangle_start is not None:\
            self.set_zoom_rectangle_end(event)\
        \
    def normalize_image(self, image, lower_quantile, upper_quantile):\
        lower_bound = np.percentile(image, lower_quantile)\
        upper_bound = np.percentile(image, upper_quantile)\
        normalized = np.clip(image, lower_bound, upper_bound)\
        normalized = (normalized - lower_bound) / (upper_bound - lower_bound)\
        max_value = np.iinfo(image.dtype).max\
        normalized = (normalized * max_value).astype(image.dtype)\
        return normalized\
    \
    def resize_arrays(self, img, mask):\
        original_dtype = img.dtype\
        scaled_height = int(img.shape[0] * self.scale_factor)\
        scaled_width = int(img.shape[1] * self.scale_factor)\
        scaled_img = resize(img, (scaled_height, scaled_width), anti_aliasing=True, preserve_range=True)\
        scaled_mask = resize(mask, (scaled_height, scaled_width), order=0, anti_aliasing=False, preserve_range=True)\
        stretched_img = resize(scaled_img, (self.canvas_height, self.canvas_width), anti_aliasing=True, preserve_range=True)\
        stretched_mask = resize(scaled_mask, (self.canvas_height, self.canvas_width), order=0, anti_aliasing=False, preserve_range=True)\
        return stretched_img.astype(original_dtype), stretched_mask.astype(original_dtype)\
\
    def setup_mode_toolbar(self):\
        self.mode_toolbar = tk.Frame(self.root)\
        self.mode_toolbar.pack(side='top', fill='x')\
        self.draw_btn = tk.Button(self.mode_toolbar, text="Draw", command=self.toggle_draw_mode)\
        self.draw_btn.pack(side='left')\
        self.magic_wand_btn = tk.Button(self.mode_toolbar, text="Magic Wand", command=self.toggle_magic_wand_mode)\
        self.magic_wand_btn.pack(side='left')\
        self.tolerance_entry = tk.Entry(self.mode_toolbar, textvariable=self.magic_wand_tolerance)\
        self.tolerance_entry.pack(side='left')\
        tk.Label(self.mode_toolbar, text="Tolerance:").pack(side='left')\
        self.erase_btn = tk.Button(self.mode_toolbar, text="Erase", command=self.toggle_erase_mode)\
        self.erase_btn.pack(side='left')\
        \
    def setup_function_toolbar(self):\
        self.function_toolbar = tk.Frame(self.root)\
        self.function_toolbar.pack(side='top', fill='x')\
        self.fill_btn = tk.Button(self.function_toolbar, text="Fill", command=self.fill_objects)\
        self.fill_btn.pack(side='left')\
        self.relabel_btn = tk.Button(self.function_toolbar, text="Relabel", command=self.relabel_objects)\
        self.relabel_btn.pack(side='left')\
        self.clear_btn = tk.Button(self.function_toolbar, text="Clear", command=self.clear_objects)\
        self.clear_btn.pack(side='left')\
        self.invert_btn = tk.Button(self.function_toolbar, text="Invert", command=self.invert_mask)\
        self.invert_btn.pack(side='left')\
\
    def setup_zoom_toolbar(self):\
        self.zoom_toolbar = tk.Frame(self.root)\
        self.zoom_toolbar.pack(side='top', fill='x')\
        self.zoom_btn = tk.Button(self.zoom_toolbar, text="Zoom", command=self.toggle_zoom_mode)\
        self.zoom_btn.pack(side='left')\
        self.normalize_btn = tk.Button(self.zoom_toolbar, text="Apply Normalization", command=self.apply_normalization)\
        self.normalize_btn.pack(side='left')\
        self.lower_entry = tk.Entry(self.zoom_toolbar, textvariable=self.lower_quantile)\
        self.lower_entry.pack(side='left')\
        tk.Label(self.zoom_toolbar, text="Lower Percentile:").pack(side='left')\
        self.upper_entry = tk.Entry(self.zoom_toolbar, textvariable=self.upper_quantile)\
        self.upper_entry.pack(side='left')\
        tk.Label(self.zoom_toolbar, text="Upper Percentile:").pack(side='left')\
        \
    def load_image_and_mask(self, index):\
        image_path = os.path.join(self.folder_path, self.image_filenames[index])\
        image = imageio.imread(image_path)\
        #if image.dtype != np.uint8:\
        #    image = (image / np.max(image) * 255).astype(np.uint8)\
        mask_filename = os.path.splitext(self.image_filenames[index])[0] + '_mask.png'\
        mask_path = os.path.join(self.masks_folder, mask_filename)\
        if os.path.exists(mask_path):\
            mask = imageio.imread(mask_path)\
            if mask.dtype != np.uint8:\
                mask = (mask / np.max(mask) * 255).astype(np.uint8)\
        else:\
            mask = np.zeros(image.shape[:2], dtype=np.uint8)\
\
        return image, mask\
        \
    def display_image_old(self):\
        if self.zoom_rectangle_id is not None:\
            self.canvas.delete(self.zoom_rectangle_id)\
            self.zoom_rectangle_id = None\
\
        combined = self.overlay_mask_on_image(self.image, self.mask)\
        self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(combined))\
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)\
        \
    def display_image(self):\
        if self.zoom_rectangle_id is not None:\
            self.canvas.delete(self.zoom_rectangle_id)\
            self.zoom_rectangle_id = None\
            \
        lower_quantile = float(self.lower_quantile.get()) if self.lower_quantile.get() else 1.0\
        upper_quantile = float(self.upper_quantile.get()) if self.upper_quantile.get() else 99.0\
        normalized = self.normalize_image(self.image, lower_quantile, upper_quantile)\
        combined = self.overlay_mask_on_image(normalized, self.mask)\
        self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(combined))\
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)\
\
    def display_zoomed_image(self):\
        if self.zoom_rectangle_start and self.zoom_rectangle_end:\
            # Convert canvas coordinates to image coordinates\
            x0, y0 = self.canvas_to_image(*self.zoom_rectangle_start)\
            x1, y1 = self.canvas_to_image(*self.zoom_rectangle_end)\
    \
            #x0, y0 = self.zoom_rectangle_start\
            #x1, y1 = self.zoom_rectangle_end\
            x0, x1 = min(x0, x1), max(x0, x1)\
            y0, y1 = min(y0, y1), max(y0, y1)\
            self.zoom_x0 = x0\
            self.zoom_y0 = y0\
            self.zoom_x1 = x1\
            self.zoom_y1 = y1\
\
            # Normalize the entire image\
            lower_quantile = float(self.lower_quantile.get()) if self.lower_quantile.get() else 1.0\
            upper_quantile = float(self.upper_quantile.get()) if self.upper_quantile.get() else 99.0\
            normalized_image = self.normalize_image(self.image, lower_quantile, upper_quantile)\
\
            # Extract the zoomed portion of the normalized image and mask\
            zoomed_image = normalized_image[y0:y1, x0:x1]\
            zoomed_mask = self.mask[y0:y1, x0:x1]\
\
            # Resize the zoomed image and mask to fit the canvas\
            canvas_height = self.canvas.winfo_height()\
            canvas_width = self.canvas.winfo_width()\
            if zoomed_image.size > 0 and canvas_height > 0 and canvas_width > 0:\
                zoomed_image_resized = resize(zoomed_image, (canvas_height, canvas_width), preserve_range=True).astype(zoomed_image.dtype)\
                zoomed_mask_resized = resize(zoomed_mask, (canvas_height, canvas_width), preserve_range=True, order=0).astype(zoomed_mask.dtype)\
                combined = self.overlay_mask_on_image(zoomed_image_resized, zoomed_mask_resized)\
                self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(combined))\
                self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)\
\
    def overlay_mask_on_image(self, image, mask, alpha=0.5):\
        if len(image.shape) == 2:\
            image = np.stack((image,) * 3, axis=-1)\
        mask = mask.astype(np.int32)\
        max_label = np.max(mask)\
        np.random.seed(0)\
        colors = np.random.randint(0, 255, size=(max_label + 1, 3), dtype=np.uint8)\
        colors[0] = [0, 0, 0]  # background color\
        colored_mask = colors[mask]\
        image_8bit = (image / 256).astype(np.uint8)\
\
        # Blend the mask and the image with transparency\
        combined_image = np.where(mask[..., None] > 0, \
                                  np.clip(image_8bit * (1 - alpha) + colored_mask * alpha, 0, 255), \
                                  image_8bit)\
        # Convert the final image back to uint8\
        combined_image = combined_image.astype(np.uint8)\
\
        return combined_image\
\
    def setup_navigation_toolbar(self):\
        navigation_toolbar = tk.Frame(self.root)\
        navigation_toolbar.pack(side='top', fill='x')\
        prev_btn = tk.Button(navigation_toolbar, text="Previous", command=self.previous_image)\
        prev_btn.pack(side='left')\
        next_btn = tk.Button(navigation_toolbar, text="Next", command=self.next_image)\
        next_btn.pack(side='left')\
        save_btn = tk.Button(navigation_toolbar, text="Save", command=self.save_mask)\
        save_btn.pack(side='left')\
\
    def previous_image(self):\
        if self.current_image_index > 0:\
            self.current_image_index -= 1\
            self.image, self.mask = self.load_image_and_mask(self.current_image_index)\
            self.display_image()\
\
    def next_image(self):\
        if self.current_image_index < len(self.image_filenames) - 1:\
            self.current_image_index += 1\
            self.image, self.mask = self.load_image_and_mask(self.current_image_index)\
            self.display_image()\
\
    def save_mask(self):\
        try:\
            if self.current_image_index < len(self.image_filenames):\
                original_size = self.original_size\
                resized_mask = resize(self.mask, original_size, preserve_range=True).astype(np.uint16)\
                save_folder = os.path.join(self.folder_path, 'new_masks')\
                if not os.path.exists(save_folder):\
                    os.makedirs(save_folder)\
                image_filename = os.path.splitext(self.image_filenames[self.current_image_index])[0] + '.png'\
                save_path = os.path.join(save_folder, image_filename)\
                \
                print(f"Saving mask to: \{save_path\}")  # Debug print\
                imageio.imwrite(save_path, resized_mask)\
                print("Save successful!")  # Confirm save\
        except Exception as e:\
            print(f"Error during saving: \{e\}")\
\
    def get_scaling_factors(self, img_width, img_height, canvas_width, canvas_height):\
        x_scale = img_width / canvas_width\
        y_scale = img_height / canvas_height\
        return x_scale, y_scale\
\
    def canvas_to_image(self, x_canvas, y_canvas):\
        x_scale, y_scale = self.get_scaling_factors(\
            self.image.shape[1], self.image.shape[0], \
            self.canvas_width, self.canvas_height\
        )\
        x_image = int(x_canvas * x_scale)\
        y_image = int(y_canvas * y_scale)\
        return x_image, y_image\
    \
    def canvas_to_image(self, x_canvas, y_canvas):\
        x_scale, y_scale = self.get_scaling_factors(\
            self.image.shape[1], self.image.shape[0],\
            self.canvas_width, self.canvas_height\
        )\
        x_image = int(x_canvas * x_scale)\
        y_image = int(y_canvas * y_scale)\
        return x_image, y_image\
\
    def set_zoom_rectangle_start(self, event):\
        if self.zoom_active:\
            self.zoom_rectangle_start = (event.x, event.y)\
    \
    def set_zoom_rectangle_end(self, event):\
        if self.zoom_active:\
            self.zoom_rectangle_end = (event.x, event.y)\
    \
            # Delete the red zoom rectangle\
            if self.zoom_rectangle_id is not None:\
                self.canvas.delete(self.zoom_rectangle_id)\
                self.zoom_rectangle_id = None  # Reset the ID after deletion\
    \
            self.display_zoomed_image()  # Trigger zoom display\
            self.canvas.unbind("<Motion>")  # Unbind the motion event\
\
    def update_zoom_box(self, event):\
        if self.zoom_active and self.zoom_rectangle_start is not None:\
            if self.zoom_rectangle_id is not None:\
                self.canvas.delete(self.zoom_rectangle_id)\
            # Assuming event.x and event.y are already in image coordinates\
            self.zoom_rectangle_end = (event.x, event.y)\
            x0, y0 = self.zoom_rectangle_start\
            x1, y1 = self.zoom_rectangle_end\
            self.zoom_rectangle_id = self.canvas.create_rectangle(x0, y0, x1, y1, outline="red", width=2)\
\
    def toggle_draw_mode(self):\
        self.drawing = not self.drawing\
        if self.drawing:\
            self.magic_wand_active = False\
            self.erase_active = False\
            self.draw_btn.config(text="Draw ON")\
            self.canvas.bind("<B1-Motion>", self.draw)\
            self.canvas.bind("<ButtonRelease-1>", self.finish_drawing)\
        else:\
            self.draw_btn.config(text="Draw")\
            self.canvas.unbind("<B1-Motion>")\
            self.canvas.unbind("<ButtonRelease-1>")\
\
    def toggle_magic_wand_mode(self):\
        self.magic_wand_active = not self.magic_wand_active\
        if self.magic_wand_active:\
            self.drawing = False\
            self.draw_btn.config(text="Draw")\
            self.erase_active = False\
            self.erase_btn.config(text="Erase")\
            self.magic_wand_btn.config(text="Magic Wand ON")\
            self.canvas.bind("<Button-1>", self.use_magic_wand)\
        else:\
            self.magic_wand_btn.config(text="Magic Wand")\
            self.canvas.unbind("<Button-1>")\
  \
    def toggle_zoom_mode(self):\
        if not self.zoom_active:\
            self.zoom_active = True\
            self.zoom_btn.config(text="Zoom ON")\
            self.canvas.bind("<Button-1>", self.set_zoom_rectangle_start)\
            self.canvas.bind("<Button-3>", self.set_zoom_rectangle_end)\
            self.canvas.bind("<Motion>", self.update_zoom_box)\
        else:\
            self.zoom_active = False\
            self.zoom_btn.config(text="Zoom")\
            self.canvas.unbind("<Button-1>")\
            self.canvas.unbind("<Button-3>")\
            self.canvas.unbind("<Motion>")\
            self.zoom_rectangle_start = self.zoom_rectangle_end = None\
            self.zoom_rectangle_id = None\
            self.display_image()\
            \
    def toggle_erase_mode(self):\
        self.erase_active = not self.erase_active\
        if self.erase_active:\
            self.erase_btn.config(text="Erase ON")\
            self.canvas.bind("<Button-1>", self.erase_object)\
            # Deactivate other modes\
            self.drawing = False\
            self.draw_btn.config(text="Draw")\
            self.magic_wand_active = False\
            self.magic_wand_btn.config(text="Magic Wand")\
        else:\
            self.erase_btn.config(text="Erase")\
            self.canvas.unbind("<Button-1>")\
            \
    def erase_object(self, event):\
        x, y = event.x, event.y\
        label_to_remove = self.mask[y, x]\
        if label_to_remove > 0:\
            self.mask[self.mask == label_to_remove] = 0\
            self.display_image()\
            \
    def use_magic_wand(self, event):\
        if self.magic_wand_active:\
            x, y = event.x, event.y\
            tolerance = int(self.magic_wand_tolerance.get())\
            if self.zoom_active:\
                self.magic_wand_zoomed((x, y), tolerance)\
            else:\
                self.magic_wand_normal((x, y), tolerance)\
\
    def magic_wand_normal(self, seed_point, tolerance):\
        x, y = seed_point\
        initial_value = self.image[y, x]\
        visited = np.zeros_like(self.image, dtype=bool)\
        queue = deque([(x, y)])\
        while queue:\
            cx, cy = queue.popleft()\
            if visited[cy, cx]:\
                continue\
            visited[cy, cx] = True\
            current_value = self.image[cy, cx]\
            if np.linalg.norm(current_value - initial_value) <= tolerance:\
                self.mask[cy, cx] = 255\
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:\
                    nx, ny = cx + dx, cy + dy\
                    if 0 <= nx < self.image.shape[1] and 0 <= ny < self.image.shape[0] and not visited[ny, nx]:\
                        queue.append((nx, ny))\
        self.display_image()\
\
    def magic_wand_zoomed_old(self, seed_point, tolerance):\
        x0, y0, x1, y1 = self.zoom_x0, self.zoom_y0, self.zoom_x1, self.zoom_y1\
        zoomed_image = self.image[y0:y1, x0:x1]\
        x, y = seed_point\
        initial_value = zoomed_image[y, x]\
        visited = np.zeros_like(zoomed_image, dtype=bool)\
        queue = deque([(x, y)])\
        zoomed_mask = np.zeros_like(zoomed_image, dtype=np.uint8)\
        while queue:\
            cx, cy = queue.popleft()\
            if visited[cy, cx]:\
                continue\
            visited[cy, cx] = True\
            current_value = zoomed_image[cy, cx]\
            if np.linalg.norm(current_value - initial_value) <= tolerance:\
                zoomed_mask[cy, cx] = 255\
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:\
                    nx, ny = cx + dx, cy + dy\
                    if 0 <= nx < zoomed_image.shape[1] and 0 <= ny < zoomed_image.shape[0] and not visited[ny, nx]:\
                        queue.append((nx, ny))\
        resized_mask = resize(zoomed_mask, (y1 - y0, x1 - x0), order=0, preserve_range=True).astype(np.uint8)\
        self.mask[y0:y1, x0:x1] = resized_mask\
        self.display_image()\
\
    def magic_wand_zoomed(self, seed_point, tolerance):\
        x0, y0, x1, y1 = self.zoom_x0, self.zoom_y0, self.zoom_x1, self.zoom_y1\
        zoomed_image = self.image[y0:y1, x0:x1]\
    \
        # Assuming seed_point is already in coordinates relative to zoomed_image\
        x, y = seed_point\
        initial_value = zoomed_image[y, x]\
    \
        visited = np.zeros_like(zoomed_image, dtype=bool)\
        queue = deque([(x, y)])\
        zoomed_mask = np.zeros_like(zoomed_image, dtype=np.uint8)\
    \
        while queue:\
            cx, cy = queue.popleft()\
            if visited[cy, cx]:\
                continue\
            visited[cy, cx] = True\
            current_value = zoomed_image[cy, cx]\
    \
            if np.linalg.norm(current_value - initial_value) <= tolerance:\
                zoomed_mask[cy, cx] = 255\
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:\
                    nx, ny = cx + dx, cy + dy\
                    if 0 <= nx < zoomed_image.shape[1] and 0 <= ny < zoomed_image.shape[0] and not visited[ny, nx]:\
                        queue.append((nx, ny))\
\
    def draw(self, event):\
        if self.drawing:\
            x, y = event.x, event.y\
            if self.draw_coordinates:\
                last_x, last_y = self.draw_coordinates[-1]\
                self.current_line = self.canvas.create_line(last_x, last_y, x, y, fill="yellow", width=3)\
            self.draw_coordinates.append((x, y))\
            \
    def draw_on_zoomed_mask(self, draw_coordinates):\
        canvas_height = self.canvas.winfo_height()\
        canvas_width = self.canvas.winfo_width()\
        zoomed_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)\
        rr, cc = polygon(np.array(draw_coordinates)[:, 1], np.array(draw_coordinates)[:, 0], shape=zoomed_mask.shape)\
        zoomed_mask[rr, cc] = 255\
        return zoomed_mask\
\
    def finish_drawing(self, event):\
        if len(self.draw_coordinates) > 2:\
            self.draw_coordinates.append(self.draw_coordinates[0])\
            if self.zoom_active:\
                x0, x1, y0, y1 = self.zoom_x0, self.zoom_x1, self.zoom_y0, self.zoom_y1\
                zoomed_mask = self.draw_on_zoomed_mask(self.draw_coordinates)\
                self.update_original_mask(zoomed_mask, x0, x1, y0, y1)\
            else:\
                rr, cc = polygon(np.array(self.draw_coordinates)[:, 1], np.array(self.draw_coordinates)[:, 0], shape=self.mask.shape)\
                self.mask[rr, cc] = np.maximum(self.mask[rr, cc], 255)\
                self.mask = self.mask.copy()\
            self.canvas.delete(self.current_line)\
            self.draw_coordinates.clear()\
            self.display_image()\
            \
    def finish_drawing_if_active(self, event):\
        if self.drawing and len(self.draw_coordinates) > 2:\
            self.finish_drawing(event)\
\
    def update_original_mask(self, zoomed_mask, x0, x1, y0, y1):\
        actual_mask_region = self.mask[y0:y1, x0:x1]\
        target_shape = actual_mask_region.shape\
        resized_mask = resize(zoomed_mask, target_shape, order=0, preserve_range=True).astype(np.uint8)\
        if resized_mask.shape != actual_mask_region.shape:\
            raise ValueError(f"Shape mismatch: resized_mask \{resized_mask.shape\}, actual_mask_region \{actual_mask_region.shape\}")\
        self.mask[y0:y1, x0:x1] = np.maximum(actual_mask_region, resized_mask)\
        self.mask = self.mask.copy()\
        self.mask[y0:y1, x0:x1] = np.maximum(self.mask[y0:y1, x0:x1], resized_mask)\
        self.mask = self.mask.copy()\
            \
    def apply_normalization(self):\
        self.image, self.mask = self.load_image_and_mask(self.current_image_index)\
        self.display_image()\
        \
    def update_normalized_image(self, *args):\
        lower_quantile = float(self.lower_quantile.get()) if self.lower_quantile.get() != '' else 1.0\
        upper_quantile = float(self.upper_quantile.get()) if self.upper_quantile.get() != '' else 99.0\
        self.norm_image = self.normalize_image(self.image, lower_quantile, upper_quantile)  \
        self.display_image()\
\
    def fill_objects(self):\
        binary_mask = self.mask > 0\
        filled_mask = binary_fill_holes(binary_mask)\
        self.mask = filled_mask.astype(np.uint8) * 255\
        labeled_mask, _ = label(filled_mask)\
        self.mask = labeled_mask\
        self.display_image()\
\
    def relabel_objects(self):\
        mask = self.mask\
        labeled_mask, num_labels = label(mask > 0)\
        self.mask = labeled_mask\
        self.display_image()\
        \
    def clear_objects(self):\
        self.mask = np.zeros_like(self.mask)\
        self.display_image()\
\
    def invert_mask(self):\
        self.mask = np.where(self.mask > 0, 0, 1)\
        self.relabel_objects()\
        self.display_image()\
    \
# Folder path, scale factor, and window size as arguments\
root = tk.Tk()\
folder_path = "/Users/olafsson/Desktop/test"\
scale_factor = 1\
width, height = 600,600\
\
if folder_path:\
    app = modify_masks(root, folder_path, scale_factor, width, height)\
    root.mainloop()}
