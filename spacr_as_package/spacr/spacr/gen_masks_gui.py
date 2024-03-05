import tkinter as tk
from tkinter import ttk, messagebox
import spacr
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')
import sys

from .logger import log_function_call

class StdoutRedirector(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

def setup_console_capture(frame):
    text = tk.Text(frame)
    text.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    sys.stdout = StdoutRedirector(text)

def display_plot(frame, data_x, data_y):
    figure = Figure(figsize=(5, 4), dpi=100)
    plot = figure.add_subplot(1, 1, 1)
    
    # Plot directly on the 'plot' (Axes object) instead of using plt.plot()
    plot.plot(data_x, data_y)  # Use your actual data here
    
    # Embed the figure in the Tkinter window
    canvas = FigureCanvasTkAgg(figure, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

@log_function_call
def on_run_clicked():
    global vars_dict, root  # Ensure root is accessible

    new_window = tk.Toplevel(root)
    new_window.title("Output Window")
    new_window.geometry("800x600")

    bottom_frame = tk.Frame(new_window, bg='#333333')
    bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    setup_console_capture(bottom_frame)  # Set up console capture in the bottom frame

    settings = {}
    for key, var in vars_dict.items():
        value = var.get()
        
        # Handle conversion for specific types if necessary
        if key in ['channels', 'timelapse_frame_limits']:  # Assuming these should be lists
            try:
                # Convert string representation of a list into an actual list
                settings[key] = eval(value)
            except:
                messagebox.showerror("Error", f"Invalid format for {key}. Please enter a valid list.")
                return
        elif key in ['nucleus_channel', 'cell_channel', 'pathogen_channel', 'examples_to_plot', 'batch_size', 'timelapse_memory', 'workers', 'fps', 'magnification']:  # Assuming these should be integers
            try:
                settings[key] = int(value) if value else None
            except ValueError:
                messagebox.showerror("Error", f"Invalid number for {key}.")
                return
        elif key in ['nucleus_background', 'cell_background', 'pathogen_background', 'nucleus_Signal_to_noise', 'cell_Signal_to_noise', 'pathogen_Signal_to_noise', 'nucleus_CP_prob', 'cell_CP_prob', 'pathogen_CP_prob', 'lower_quantile']:  # Assuming these should be floats
            try:
                settings[key] = float(value) if value else None
            except ValueError:
                messagebox.showerror("Error", f"Invalid number for {key}.")
                return
        else:
            # Directly assign the value for other types (strings and booleans)
            settings[key] = value
    
    # After collecting and converting settings, call your processing function
    
    settings['remove_background'] = True
    settings['fps'] = 2
    settings['all_to_mip'] = False
    settings['pick_slice'] = False
    settings['merge'] = False
    settings['skip_mode'] = ''
    settings['verbose'] = False
    settings['normalize_plots'] = True
    settings['randomize'] = True
    settings['preprocess'] = True
    settings['masks'] = True
    settings['examples_to_plot'] = 1
    
    
    
    
    
    try:
        print("Settings:", settings)  # For demonstration, replace with your actual processing call
        spacr.core.preprocess_generate_masks(settings['src'], settings=settings, advanced_settings={})
        #data_x, data_y = get_plot_data()
        #display_plot(top_frame, data_x, data_y)
    except Exception as e:
        messagebox.showerror("Error", str(e))
        print("Error during processing:", e)

# Function to create input fields dynamically
def create_input_field(frame, label_text, row, var_type='entry', options=None, default_value=None):
    label = ttk.Label(frame, text=label_text, style='TLabel')  # Assuming you have a dark mode style for labels too
    label.grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
    
    if var_type == 'entry':
        var = tk.StringVar(value=default_value)  # Set default value
        entry = ttk.Entry(frame, textvariable=var, style='TEntry')  # Assuming you have a dark mode style for entries
        entry.grid(column=1, row=row, sticky=tk.EW, padx=5)
    elif var_type == 'check':
        var = tk.BooleanVar(value=default_value)  # Set default value (True/False)
        # Use the custom style for Checkbutton
        check = ttk.Checkbutton(frame, variable=var, style='Dark.TCheckbutton')
        check.grid(column=1, row=row, sticky=tk.W, padx=5)
    elif var_type == 'combo':
        var = tk.StringVar(value=default_value)  # Set default value
        combo = ttk.Combobox(frame, textvariable=var, values=options, style='TCombobox')  # Assuming you have a dark mode style for comboboxes
        combo.grid(column=1, row=row, sticky=tk.EW, padx=5)
        if default_value:
            combo.set(default_value)
    else:
        var = None  # Placeholder in case of an undefined var_type
    
    return var

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, bg='#333333', **kwargs):
        super().__init__(container, *args, **kwargs)
        self.configure(style='TFrame')  # Ensure this uses the styled frame from dark mode
        
        canvas = tk.Canvas(self, bg=bg)  # Set canvas background to match dark mode
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        
        self.scrollable_frame = ttk.Frame(canvas, style='TFrame')  # Ensure it uses the styled frame
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

def create_dark_mode(root):
    dark_bg = '#333333'
    light_text = 'white'
    
    style = ttk.Style(root)
    style.theme_use('clam')

    # Configure styles for all widget types you're using
    style.configure('TFrame', background=dark_bg)
    style.configure('TLabel', background=dark_bg, foreground=light_text, font=("Arial", 12))
    style.configure('TEntry', fieldbackground=dark_bg, foreground=light_text, background=dark_bg, font=("Arial", 12))
    style.configure('TButton', background=dark_bg, foreground=light_text, font=("Arial", 14))
    style.map('TButton', background=[('active', dark_bg)], foreground=[('active', light_text)])
    style.configure('Dark.TCheckbutton', background='#333333', foreground='white')
    style.map('Dark.TCheckbutton',
              background=[('active', '#333333'), ('selected', '#333333')],
              foreground=[('active', 'white'), ('selected', 'white')])

    # Set root background explicitly
    root.configure(bg=dark_bg)
    
def mask_gui():
    global vars_dict, root
    # Create the main window
    root = tk.Tk()
    root.geometry("500x1000")
    
    
    root.configure(bg='#333333')
    root.title("Configuration GUI")
    create_dark_mode(root)
    
    # Using ScrollableFrame
    scrollable_frame = ScrollableFrame(root)
    scrollable_frame.pack(fill="both", expand=True)

    # Variables
    variables = {
        'src': ('entry', None, '/mnt/data/CellVoyager/40x/einar/mitotrackerHeLaToxoDsRed_20240224_123156/test_gui'),
        'metadata_type': ('combo', ['cellvoyager', 'cq1', 'nikon', 'zeis', 'custom'], 'cellvoyager'),
        'custom_regex': ('entry', None, None),
        'experiment': ('entry', None, 'exp'),
        'channels': ('combo', ['[0,1,2,3]','[0,1,2]','[0,1]','[0]'], '[0,1,2,3]'),
        'magnification': ('combo', [20, 40, 60,], 40),
        'nucleus_channel': ('combo', [0,1,2,3, None], 0),
        'nucleus_background': ('entry', None, 100),
        'nucleus_Signal_to_noise': ('entry', None, 5),
        'nucleus_CP_prob': ('entry', None, 0),
        'cell_channel': ('combo', [0,1,2,3, None], 3),
        'cell_background': ('entry', None, 100),
        'cell_Signal_to_noise': ('entry', None, 5),
        'cell_CP_prob': ('entry', None, 0),
        'pathogen_channel': ('combo', [0,1,2,3, None], 2),
        'pathogen_background': ('entry', None, 100),
        'pathogen_Signal_to_noise': ('entry', None, 3),
        'pathogen_CP_prob': ('entry', None, 0),
        #'preprocess': ('check', None, True),
        #'masks': ('check', None, True),
        #'examples_to_plot': ('entry', None, 1),
        #'randomize': ('check', None, True),
        'batch_size': ('entry', None, 50),
        'timelapse': ('check', None, False),
        'timelapse_displacement': ('entry', None, None),
        'timelapse_memory': ('entry', None, 0),
        'timelapse_frame_limits': ('entry', None, '[0,1000]'),
        'timelapse_remove_transient': ('check', None, True),
        'timelapse_mode': ('combo',  ['trackpy', 'btrack'], 'trackpy'),
        'timelapse_objects': ('combo', ['cell','nucleus','pathogen','cytoplasm', None], None),
        #'fps': ('entry', None, 2),
        #'remove_background': ('check', None, True),
        'lower_quantile': ('entry', None, 0.01),
        #'merge': ('check', None, False),
        #'normalize_plots': ('check', None, True),
        #'all_to_mip': ('check', None, False),
        #'pick_slice': ('check', None, False),
        #'skip_mode': ('entry', None, None),
        'save': ('check', None, True),
        'plot': ('check', None, True),
        'workers': ('entry', None, 30),
        'verbose': ('check', None, True),
    }

    vars_dict = {}

    # Create input fields within scrollable_frame.scrollable_frame instead of frame
    row = 0
    for key, (var_type, options, default_value) in variables.items():
        vars_dict[key] = create_input_field(scrollable_frame.scrollable_frame, key, row, var_type, options, default_value)
        row += 1

    # Run button, adjusted to use 'on_run_clicked'
    run_button = ttk.Button(scrollable_frame.scrollable_frame, text="Run", command=on_run_clicked)
    run_button.grid(column=0, row=len(variables), columnspan=2, pady=10)

    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    mask_gui()