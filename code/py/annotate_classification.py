import tkinter as tk
from tkinter import Label
from PIL import Image, ImageOps, ImageTk
import numpy as np
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import os
import threading
from queue import Queue
import time

class ImageApp:
    def __init__(self, root, db_path, image_type=None, channels=None, grid_rows=None, grid_cols=None, image_size=(200, 200), annotation_column='annotate'):
        self.root = root
        self.db_path = db_path
        self.index = 0
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.image_size = image_size
        self.annotation_column = annotation_column
        self.image_type = image_type
        self.channels = channels
        self.images = {}
        self.pending_updates = {}
        self.labels = []
        #self.updating_db = True
        self.terminate = False
        self.update_queue = Queue()
        self.status_label = Label(self.root, text="", font=("Arial", 12))
        self.status_label.grid(row=self.grid_rows + 1, column=0, columnspan=self.grid_cols)

        self.db_update_thread = threading.Thread(target=self.update_database_worker)
        self.db_update_thread.start()
        
        for i in range(grid_rows * grid_cols):
            label = tk.Label(root)
            label.grid(row=i // grid_cols, column=i % grid_cols)
            self.labels.append(label)
        
    @staticmethod
    def normalize_image(img):
        img_array = np.array(img)
        img_array = ((img_array - img_array.min()) * (1/(img_array.max() - img_array.min()) * 255)).astype('uint8')
        return Image.fromarray(img_array)

    def add_colored_border(self, img, border_width, border_color):
        top_border = Image.new('RGB', (img.width, border_width), color=border_color)
        bottom_border = Image.new('RGB', (img.width, border_width), color=border_color)
        left_border = Image.new('RGB', (border_width, img.height), color=border_color)
        right_border = Image.new('RGB', (border_width, img.height), color=border_color)

        bordered_img = Image.new('RGB', (img.width + 2 * border_width, img.height + 2 * border_width), color='white')
        bordered_img.paste(top_border, (border_width, 0))
        bordered_img.paste(bottom_border, (border_width, img.height + border_width))
        bordered_img.paste(left_border, (0, border_width))
        bordered_img.paste(right_border, (img.width + border_width, border_width))
        bordered_img.paste(img, (border_width, border_width))

        return bordered_img
    
    def filter_channels(self, img):
        r, g, b = img.split()
        if self.channels:
            if 'r' not in self.channels:
                r = r.point(lambda _: 0)
            if 'g' not in self.channels:
                g = g.point(lambda _: 0)
            if 'b' not in self.channels:
                b = b.point(lambda _: 0)

            if len(self.channels) == 1:
                channel_img = r if 'r' in self.channels else (g if 'g' in self.channels else b)
                return ImageOps.grayscale(channel_img)

        return Image.merge("RGB", (r, g, b))

    def load_images(self):
        for label in self.labels:
            label.config(image='')

        self.images = {}

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        if self.image_type:
            c.execute(f"SELECT png_path, {self.annotation_column} FROM png_list WHERE png_path LIKE ? LIMIT ?, ?", (f"%{self.image_type}%", self.index, self.grid_rows * self.grid_cols))
        else:
            c.execute(f"SELECT png_path, {self.annotation_column} FROM png_list LIMIT ?, ?", (self.index, self.grid_rows * self.grid_cols))
        
        paths = c.fetchall()
        conn.close()

        with ThreadPoolExecutor() as executor:
            loaded_images = list(executor.map(self.load_single_image, paths))

        for i, (img, annotation) in enumerate(loaded_images):
            if annotation:
                border_color = 'teal' if annotation == 1 else 'red'
                img = self.add_colored_border(img, border_width=5, border_color=border_color)
            
            photo = ImageTk.PhotoImage(img)
            label = self.labels[i]
            self.images[label] = photo
            label.config(image=photo)
            
            path = paths[i][0]
            label.bind('<Button-1>', self.get_on_image_click(path, label, img))
            label.bind('<Button-3>', self.get_on_image_click(path, label, img))

        self.root.update()

    def load_single_image(self, path_annotation_tuple):
        path, annotation = path_annotation_tuple
        img = Image.open(path)
        if img.mode == "I":
            img = self.normalize_image(img)
        img = img.convert('RGB')
        img = self.filter_channels(img)
        img = img.resize(self.image_size)
        return img, annotation
        
    def get_on_image_click(self, path, label, img):
        def on_image_click(event):
            
            new_annotation = 1 if event.num == 1 else (2 if event.num == 3 else None)
            
            if path in self.pending_updates and self.pending_updates[path] == new_annotation:
                self.pending_updates[path] = None
                new_annotation = None
            else:
                self.pending_updates[path] = new_annotation
            
            print(f"Image {os.path.split(path)[1]} annotated: {new_annotation}")
            
            img_ = img.crop((5, 5, img.width-5, img.height-5))
            border_fill = 'teal' if new_annotation == 1 else ('red' if new_annotation == 2 else None)
            img_ = ImageOps.expand(img_, border=5, fill=border_fill) if border_fill else img_

            photo = ImageTk.PhotoImage(img_)
            self.images[label] = photo
            label.config(image=photo)
            self.root.update()

        return on_image_click
     
    @staticmethod
    def update_html(text):
        display(HTML(f"""
        <script>
        document.getElementById('unique_id').innerHTML = '{text}';
        </script>
        """))

    def update_database_worker(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        display(HTML("<div id='unique_id'>Initial Text</div>"))

        while True:
            if self.terminate:
                conn.close()
                break

            if not self.update_queue.empty():
                ImageApp.update_html("Do not exit, Updating database...")
                self.status_label.config(text='Do not exit, Updating database...')

                pending_updates = self.update_queue.get()
                for path, new_annotation in pending_updates.items():
                    if new_annotation is None:
                        c.execute(f'UPDATE png_list SET {self.annotation_column} = NULL WHERE png_path = ?', (path,))
                    else:
                        c.execute(f'UPDATE png_list SET {self.annotation_column} = ? WHERE png_path = ?', (new_annotation, path))
                conn.commit()

                # Reset the text
                ImageApp.update_html('')
                self.status_label.config(text='')
                self.root.update()
            time.sleep(0.1)

    def update_gui_text(self, text):
        self.status_label.config(text=text)
        self.root.update()

    def next_page(self):
        if self.pending_updates:  # Check if the dictionary is not empty
            self.update_queue.put(self.pending_updates.copy())
        self.pending_updates.clear()
        self.index += self.grid_rows * self.grid_cols
        self.load_images()

    def previous_page(self):
        if self.pending_updates:  # Check if the dictionary is not empty
            self.update_queue.put(self.pending_updates.copy())
        self.pending_updates.clear()
        self.index -= self.grid_rows * self.grid_cols
        if self.index < 0:
            self.index = 0
        self.load_images()

    def shutdown(self):
        self.terminate = True  # Set terminate first
        self.update_queue.put(self.pending_updates.copy())
        self.pending_updates.clear()
        self.db_update_thread.join()  # Join the thread to make sure database is updated
        self.root.quit()
        self.root.destroy()
        print(f'Quit application')

def annotate(db, image_type=None, channels=None, geom="1000x1100", img_size=(200, 200), rows=5, columns=5, annotation_column='annotate'):
    #display(HTML("<div id='unique_id'>Initial Text</div>"))
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('PRAGMA table_info(png_list)')
    cols = c.fetchall()
    if annotation_column not in [col[1] for col in cols]:
        c.execute(f'ALTER TABLE png_list ADD COLUMN {annotation_column} integer')
    conn.commit()
    conn.close()

    root = tk.Tk()
    root.geometry(geom)
    app = ImageApp(root, db, image_type=image_type, channels=channels, image_size=img_size, grid_rows=rows, grid_cols=columns, annotation_column=annotation_column)
    
    next_button = tk.Button(root, text="Next", command=app.next_page)
    next_button.grid(row=app.grid_rows, column=app.grid_cols - 1)
    back_button = tk.Button(root, text="Back", command=app.previous_page)
    back_button.grid(row=app.grid_rows, column=app.grid_cols - 2)
    exit_button = tk.Button(root, text="Exit", command=app.shutdown)
    exit_button.grid(row=app.grid_rows, column=app.grid_cols - 3)
    
    app.load_images()
    root.mainloop()
    
def check_for_duplicates(db):
    db_path = db
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT file_name, COUNT(file_name) FROM png_list GROUP BY file_name HAVING COUNT(file_name) > 1')
    duplicates = c.fetchall()
    for duplicate in duplicates:
        file_name = duplicate[0]
        count = duplicate[1]
        #print(f"Found {count} duplicates for file_name {file_name}. Deleting {count-1} of them.")
        c.execute('SELECT rowid FROM png_list WHERE file_name = ?', (file_name,))
        rowids = c.fetchall()
        for rowid in rowids[:-1]:
            c.execute('DELETE FROM png_list WHERE rowid = ?', (rowid[0],))
    conn.commit()
    conn.close()
