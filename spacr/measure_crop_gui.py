import os, spacr, sys, threading, queue
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from tkinter.font import nametofont
from ttkthemes import ThemedTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import traceback
from threading import Thread
import ctypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except AttributeError:
    pass

from .logger import log_function_call
from .gui_utils import ScrollableFrame, StdoutRedirector, set_dark_style, set_default_font, measure_variables, generate_fields, create_dark_mode, check_measure_gui_settings, add_measure_gui_defaults
thread_control = {"run_thread": None, "stop_requested": False}


def clear_canvas():
    global canvas
    # Clear each plot (axes) in the figure
    for ax in canvas.figure.get_axes():
        ax.clear()  # Clears the axes, but keeps them visible for new plots

    # Redraw the now empty canvas without changing its size
    canvas.draw_idle()  # Using draw_idle for efficiency in redrawing
        
def initiate_abort():
    global thread_control, q
    thread_control["stop_requested"] = True
    if thread_control["run_thread"] is not None:
        thread_control["run_thread"].join(timeout=1)  # Timeout after 1 second
        if thread_control["run_thread"].is_alive():
            q.put("Thread didn't terminate within timeout.")
        thread_control["run_thread"] = None

def start_thread(q, fig_queue):
    global thread_control
    thread_control["stop_requested"] = False  # Reset the stop signal
    thread_control["run_thread"] = Thread(target=run_measure_gui, args=(q, fig_queue))
    thread_control["run_thread"].start()

@log_function_call
def measure_crop_wrapper(*args, **kwargs):
    global fig_queue
    
    def my_show():
        fig = plt.gcf()
        fig_queue.put(fig)  # Put the figure into the queue
        plt.close(fig)  # Close the figure to prevent it from being shown by plt.show()

    original_show = plt.show
    plt.show = my_show

    try:
        spacr.measure.measure_crop(*args, **kwargs)
    except Exception as e:
        q.put(f"Error during processing: {e}")
        traceback.print_exc()
        #pass
    finally:
        plt.show = original_show

@log_function_call
def run_measure_gui_v1(q, fig_queue):
    global vars_dict, thread_control
    try:
        while not thread_control["stop_requested"]:
            #for key in vars_dict:
            #    value = vars_dict[key].get()
            #    print(key, value, type(value))
            settings = check_measure_gui_settings(vars_dict)
            settings = add_measure_gui_defaults(settings)
            for key in settings:
                value = settings[key]
                print(key, value, type(value))
            measure_crop_wrapper(settings=settings, annotation_settings={}, advanced_settings={})
            thread_control["stop_requested"] = True
    except Exception as e:
        #pass
        q.put(f"Error during processing: {e}")
        traceback.print_exc()
    finally:
        # Ensure the thread is marked as not running anymore
        thread_control["run_thread"] = None
        # Reset the stop_requested flag for future operations
        thread_control["stop_requested"] = False
        
@log_function_call        
def run_measure_gui(q, fig_queue):
    global vars_dict, thread_control
    try:
        while not thread_control["stop_requested"]:
            settings = check_measure_gui_settings(vars_dict)
            settings = add_measure_gui_defaults(settings)
            measure_crop_wrapper(settings=settings, annotation_settings={}, advanced_settings={})
            thread_control["stop_requested"] = True
    except Exception as e:
        q.put(f"Error during processing: {e}")  # Use the queue to report errors
        traceback.print_exc()
    finally:
        thread_control["run_thread"] = None
        thread_control["stop_requested"] = False

def main_thread_update_function(root, q, fig_queue):
    """
    This function is to be called periodically by root.after() in the main GUI thread.
    It checks the queues for new messages or figures and updates the GUI accordingly.
    """
    try:
        # Process any new messages in q
        while not q.empty():
            message = q.get_nowait()
            # Update GUI with message

        # Process any new figures in fig_queue
        while not fig_queue.empty():
            fig = fig_queue.get_nowait()
            # Update GUI to display fig
    except Exception as e:
        # Handle any exceptions, likely during GUI update attempts
        print(f"Error updating GUI: {e}")
    finally:
        root.after(100, lambda: main_thread_update_function(root, q, fig_queue))

@log_function_call
def initiate_measure_root(width, height):
    global root, vars_dict, q, canvas, fig_queue, canvas_widget, thread_control
    
    theme = 'breeze'
    
    if theme in ['clam']:
        root = tk.Tk()
        style = ttk.Style(root)
        style.theme_use(theme) #plastik, clearlooks, elegance, default was clam #alt, breeze, arc
        set_dark_style(style)

    elif theme in ['breeze']:
        root = ThemedTk(theme="breeze")
        style = ttk.Style(root)
        set_dark_style(style)
    
    set_default_font(root, font_name="Arial", size=10)
    #root.state('zoomed')  # For Windows to maximize the window
    root.attributes('-fullscreen', True)
    root.geometry(f"{width}x{height}")
    root.configure(bg='#333333')
    root.title("SpaCer: generate masks")
    fig_queue = queue.Queue()
    
    def _process_fig_queue():
        global canvas
        try:
            while not fig_queue.empty():
                clear_canvas()
                fig = fig_queue.get_nowait()
                #set_fig_text_properties(fig, font_size=8)
                for ax in fig.get_axes():
                    ax.set_xticks([])  # Remove x-axis ticks
                    ax.set_yticks([])  # Remove y-axis ticks
                    ax.xaxis.set_visible(False)  # Hide the x-axis
                    ax.yaxis.set_visible(False)  # Hide the y-axis
                    #ax.title.set_fontsize(14) 
                #disable_interactivity(fig)
                fig.tight_layout()
                fig.set_facecolor('#333333')
                canvas.figure = fig
                fig_width, fig_height = canvas_widget.winfo_width(), canvas_widget.winfo_height()
                fig.set_size_inches(fig_width / fig.dpi, fig_height / fig.dpi, forward=True)
                canvas.draw_idle() 
        except queue.Empty:
            pass
        except Exception as e:
            traceback.print_exc()
            #pass
        finally:
            canvas_widget.after(100, _process_fig_queue)
    
    # Process queue for console output
    def _process_console_queue():
        while not q.empty():
            message = q.get_nowait()
            console_output.insert(tk.END, message)
            console_output.see(tk.END)
        console_output.after(100, _process_console_queue)
        
    # Vertical container for settings and console
    vertical_container = tk.PanedWindow(root, orient=tk.HORIZONTAL) #VERTICAL
    vertical_container.pack(fill=tk.BOTH, expand=True)

    # Scrollable Frame for user settings
    scrollable_frame = ScrollableFrame(vertical_container)
    vertical_container.add(scrollable_frame, stretch="always")

    # Setup for user input fields (variables)
    variables = measure_variables()
    vars_dict = generate_fields(variables, scrollable_frame)
    
    # Horizontal container for Matplotlib figure and the vertical pane (for settings and console)
    horizontal_container = tk.PanedWindow(vertical_container, orient=tk.VERTICAL) #HORIZONTAL
    vertical_container.add(horizontal_container, stretch="always")

    # Matplotlib figure setup
    figure = Figure(figsize=(30, 4), dpi=100, facecolor='#333333')
    plot = figure.add_subplot(111)
    plot.plot([], [])  # This creates an empty plot.
    plot.axis('off')

    # Embedding the Matplotlib figure in the Tkinter window
    canvas = FigureCanvasTkAgg(figure, master=horizontal_container)
    canvas.get_tk_widget().configure(cursor='arrow', background='#333333', highlightthickness=0)
    #canvas.get_tk_widget().configure(cursor='arrow')
    canvas_widget = canvas.get_tk_widget()
    horizontal_container.add(canvas_widget, stretch="always")
    canvas.draw()

    # Console output setup below the settings
    console_output = scrolledtext.ScrolledText(vertical_container, height=10)
    vertical_container.add(console_output, stretch="always")

    # Queue and redirection setup for updating console output safely
    q = queue.Queue()
    sys.stdout = StdoutRedirector(console_output)
    sys.stderr = StdoutRedirector(console_output)
    
    # This is your GUI setup where you create the Run button
    #run_button = ttk.Button(scrollable_frame.scrollable_frame, text="Run", command=lambda: threading.Thread(target=run_mask_gui, args=(q, fig_queue)).start())
    #run_button.grid(row=40, column=0, pady=10)
    run_button = ttk.Button(scrollable_frame.scrollable_frame, text="Run",command=lambda: start_thread(q, fig_queue))
    run_button.grid(row=40, column=0, pady=10)
    
    abort_button = ttk.Button(scrollable_frame.scrollable_frame, text="Abort", command=initiate_abort)
    abort_button.grid(row=40, column=1, pady=10)
    
    _process_console_queue()
    _process_fig_queue()
    create_dark_mode(root, style, console_output)
    
    return root, vars_dict

def measure_gui():
    global vars_dict, root
    root, vars_dict = initiate_measure_root(1000, 1500)
    root.mainloop()

if __name__ == "__main__":
    measure_gui()