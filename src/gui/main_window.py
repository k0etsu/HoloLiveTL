"""
Main GUI window for Live Translator
"""
import time
import threading
from queue import Queue, Empty
import tkinter as tk
from tkinter import messagebox, Toplevel, Label, colorchooser
from datetime import datetime
import json
import os
import traceback
import sys
import io
from collections import deque

# Import our modular components
from modules.config import Config
from modules.stats import TranslatorStats
from modules.audio_utils import find_audio_device
from modules.recorder import recorder_thread
from modules.processor import processor_thread
from modules.model_utils import ensure_model_downloaded
from modules.config import MODEL_ID

# Check dependencies
try:
    import torchaudio
except ImportError:
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Dependency Error", "torchaudio not found. Please run 'pip install torchaudio' in your terminal.")
    root.destroy()
    raise ImportError("torchaudio module not found")

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    hf_hub_download = None
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Dependency Error", "huggingface_hub not found. Please run 'pip install huggingface_hub' in your terminal.")
    root.destroy()
    raise ImportError("huggingface_hub module not found")

try:
    import soundcard as sc
except ImportError:
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Dependency Error", "soundcard not found. Please run 'pip install soundcard' in your terminal.")
    root.destroy()
    raise ImportError("soundcard module not found")

class ControlGUI:
    def __init__(self, root, config, stats, gui_queue):
        self.root = root
        self.config = config
        self.stats = stats
        self.gui_queue = gui_queue
        
        self.root.title("Live Audio Translator - Organized")
        self.root.geometry("600x800")
        self.root.resizable(True, True)

        self.worker_threads = []
        self.stop_event = None
        self.subtitle_window = None
        self.subtitle_label = None
        self.subtitle_shadow_label = None
        self.background_canvas = None
        self.background_rect = None
        self.last_subtitle = ""
        self.subtitle_history = []
        self._drag_data = {"x": 0, "y": 0}
        self.device_list = []

        self.log_window = None
        self.log_text_widget = None

        self.log_queue = Queue()
        self.log_buffer = deque(maxlen=1000)
        self.log_file = None
        self._patch_stdout()
        
        os.makedirs("presets", exist_ok=True)
        
        self.setup_ui()
        self._start_log_processor()

    def _patch_stdout(self):
        self.log_file = open("translator_app.log", "a", encoding='utf-8', buffering=1)

        class StdoutRedirector(io.TextIOBase):
            def __init__(self, outer):
                self.outer = outer

            def write(self, s):
                if sys.__stdout__ is not None:
                    sys.__stdout__.write(s)
                if self.outer.log_file and not self.outer.log_file.closed:
                    self.outer.log_file.write(s)
                self.outer.log_buffer.append(s)
                self.outer.log_queue.put(s)
            
            def flush(self):
                if sys.__stdout__ is not None:
                    sys.__stdout__.flush()
                if self.outer.log_file and not self.outer.log_file.closed:
                    self.outer.log_file.flush()

        sys.stdout = StdoutRedirector(self)
        sys.stderr = sys.stdout
        print(f"\n--- Application session started at {datetime.now()} ---")

    def _start_log_processor(self):
        self.root.after(100, self._process_log_queue)

    def _process_log_queue(self):
        messages_to_process = 100
        batch = []
        for _ in range(messages_to_process):
            try:
                message = self.log_queue.get_nowait()
                batch.append(message)
            except Empty:
                break
        
        if batch and self.log_text_widget and self.log_text_widget.winfo_exists():
            self.log_text_widget.config(state='normal')
            self.log_text_widget.insert('end', "".join(batch))
            self.log_text_widget.see('end')
            self.log_text_widget.config(state='disabled')
            
        self.root.after(100, self._process_log_queue)

    def setup_ui(self):
        title_label = tk.Label(self.root, text="Live Audio Translator (Organized)", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=5)
        info_label = tk.Label(self.root, text="Organized modular version for better performance.", font=("Helvetica", 8), fg="grey")
        info_label.pack(pady=(0, 5))

        top_controls_frame = tk.Frame(self.root)
        top_controls_frame.pack(pady=(0, 5), padx=20, fill='x')
        tk.Label(top_controls_frame, text="Audio Device:").pack(side="left")
        self.device_var = tk.StringVar()
        self.device_menu = tk.OptionMenu(top_controls_frame, self.device_var, "Loading...")
        self.device_menu.pack(side="left", padx=5, expand=True, fill='x')
        self.refresh_devices()
        self.device_var.trace_add('write', self.on_device_select)

        log_button = tk.Button(self.root, text="Show Log", command=self.open_log_window, font=("Helvetica", 10))
        log_button.pack(pady=(0, 5))

        self.status_label = tk.Label(self.root, text="Status: Ready", font=("Helvetica", 10), fg="green")
        self.status_label.pack(pady=(0, 5))

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)

        self.download_button = tk.Button(button_frame, text="Download Model", command=self.download_model, bg="#007bff",
                                        fg="white", font=("Helvetica", 12), width=15, height=2)
        self.download_button.pack(side="left", padx=10)

        self.start_button = tk.Button(button_frame, text="Start", command=self.start_translator, bg="#28a745", fg="white", font=("Helvetica", 12), width=10, height=2)
        self.start_button.pack(side="left", padx=10)

        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop_translator, bg="#dc3545", fg="white", font=("Helvetica", 12), width=10, height=2, state="disabled")
        self.stop_button.pack(side="left", padx=10)

        settings_container = tk.Frame(self.root)
        settings_container.pack(pady=5, padx=20, fill='x', expand=True)

        dynamic_frame = tk.LabelFrame(settings_container, text="Dynamic Chunking Settings", padx=10, pady=10)
        dynamic_frame.pack(pady=5, fill="x")
        self.dynamic_chunk_var = tk.BooleanVar(value=self.config.use_dynamic_chunking)
        self.dynamic_chunk_check = tk.Checkbutton(dynamic_frame, text="Enable Dynamic Chunks (Recommended)", variable=self.dynamic_chunk_var)
        self.dynamic_chunk_check.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 5))
        tk.Label(dynamic_frame, text="Silence Timeout (s):").grid(row=1, column=0, sticky="w", pady=2)
        self.dyn_silence_var = tk.StringVar(value=str(self.config.dynamic_silence_timeout))
        self.dyn_silence_entry = tk.Entry(dynamic_frame, textvariable=self.dyn_silence_var, width=8)
        self.dyn_silence_entry.grid(row=1, column=1, padx=5, sticky="w")
        tk.Label(dynamic_frame, text="Max Record Time (s):").grid(row=1, column=2, sticky="w", pady=2, padx=(10,0))
        self.dyn_max_dur_var = tk.StringVar(value=str(self.config.dynamic_max_chunk_duration))
        self.dyn_max_dur_entry = tk.Entry(dynamic_frame, textvariable=self.dyn_max_dur_var, width=8)
        self.dyn_max_dur_entry.grid(row=1, column=3, padx=5, sticky="w")
        tk.Label(dynamic_frame, text="Min Speech Time (s):").grid(row=2, column=0, sticky="w", pady=2)
        self.dyn_min_speech_var = tk.StringVar(value=str(self.config.dynamic_min_speech_duration))
        self.dyn_min_speech_entry = tk.Entry(dynamic_frame, textvariable=self.dyn_min_speech_var, width=8)
        self.dyn_min_speech_entry.grid(row=2, column=1, padx=5, sticky="w")

        basic_frame = tk.LabelFrame(settings_container, text="Audio Filter Settings", padx=10, pady=10)
        basic_frame.pack(pady=5, fill="x")
        tk.Label(basic_frame, text="Volume Threshold:").grid(row=0, column=0, sticky="w", pady=2)
        self.volume_var = tk.StringVar(value=str(self.config.volume_threshold))
        self.volume_entry = tk.Entry(basic_frame, textvariable=self.volume_var, width=8)
        self.volume_entry.grid(row=0, column=1, padx=5, sticky="w")
        self.vad_var = tk.BooleanVar(value=self.config.use_vad_filter)
        self.vad_check = tk.Checkbutton(basic_frame, text="Enable VAD Filter (for both modes)", variable=self.vad_var)
        self.vad_check.grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))
        tk.Label(basic_frame, text="VAD Threshold (%):").grid(row=2, column=0, sticky="w", pady=2)
        self.vad_threshold_var = tk.StringVar(value=str(int(self.config.vad_threshold * 100)))
        self.vad_threshold_entry = tk.Entry(basic_frame, textvariable=self.vad_threshold_var, width=8)
        self.vad_threshold_entry.grid(row=2, column=1, padx=5, sticky="w")

        appearance_frame = tk.LabelFrame(settings_container, text="Subtitle Appearance", padx=10, pady=10)
        appearance_frame.pack(pady=5, fill="x")
        tk.Label(appearance_frame, text="Font Size:").grid(row=0, column=0, sticky="w", pady=2)
        self.font_var = tk.StringVar(value=str(self.config.font_size))
        self.font_entry = tk.Entry(appearance_frame, textvariable=self.font_var, width=8)
        self.font_entry.grid(row=0, column=1, padx=5, sticky="w")
        self.font_entry.bind('<KeyRelease>', self.update_subtitle_style)
        tk.Label(appearance_frame, text="Font Weight:").grid(row=0, column=2, sticky="w", pady=2, padx=(10, 0))
        self.font_weight_var = tk.StringVar(value=self.config.font_weight)
        self.font_weight_menu = tk.OptionMenu(appearance_frame, self.font_weight_var, 'normal', 'bold', command=self.on_font_weight_change)
        self.font_weight_menu.grid(row=0, column=3, padx=5, sticky="w")
        tk.Label(appearance_frame, text="Opacity (%):").grid(row=1, column=0, sticky="w", pady=2)
        self.opacity_var = tk.StringVar(value=str(int(self.config.window_opacity * 100)))
        self.opacity_entry = tk.Entry(appearance_frame, textvariable=self.opacity_var, width=8)
        self.opacity_entry.grid(row=1, column=1, padx=5, sticky="w")
        self.opacity_entry.bind('<KeyRelease>', self.on_opacity_change)
        tk.Label(appearance_frame, text="BG Mode:").grid(row=1, column=2, sticky="w", pady=2, padx=(10,0))
        self.bg_mode_var = tk.StringVar(value=self.config.subtitle_bg_mode)
        self.bg_mode_menu = tk.OptionMenu(appearance_frame, self.bg_mode_var, 'transparent', 'solid', command=self.set_bg_mode)
        self.bg_mode_menu.grid(row=1, column=3, padx=5, sticky="w")
        tk.Label(appearance_frame, text="BG Color:").grid(row=2, column=0, sticky="w", pady=2)
        self.bg_color_var = tk.StringVar(value=self.config.subtitle_bg_color)
        self.bg_color_btn = tk.Button(appearance_frame, text="Pick", command=self.pick_bg_color)
        self.bg_color_btn.grid(row=2, column=1, padx=5, sticky="w")
        self.bg_color_display = tk.Label(appearance_frame, text='  ', bg=self.config.subtitle_bg_color, relief="solid", borderwidth=1)
        self.bg_color_display.grid(row=2, column=2, padx=5, sticky="w")
        tk.Label(appearance_frame, text="Font Color:").grid(row=3, column=0, sticky="w", pady=2)
        self.font_color_var = tk.StringVar(value=self.config.subtitle_font_color)
        self.font_color_btn = tk.Button(appearance_frame, text="Pick", command=self.pick_font_color)
        self.font_color_btn.grid(row=3, column=1, padx=5, sticky="w")
        self.font_color_display = tk.Label(appearance_frame, text='  ', bg=self.config.subtitle_font_color, relief="solid", borderwidth=1)
        self.font_color_display.grid(row=3, column=2, padx=5, sticky="w")
        
        self.text_shadow_var = tk.BooleanVar(value=getattr(self.config, 'text_shadow', True))
        self.text_shadow_check = tk.Checkbutton(appearance_frame, text="Text Shadow", variable=self.text_shadow_var, command=self.on_text_shadow_change)
        self.text_shadow_check.grid(row=4, column=0, columnspan=2, sticky="w", pady=2)
        
        presets_frame = tk.LabelFrame(settings_container, text="Presets", padx=10, pady=10)
        presets_frame.pack(pady=5, fill="x")

        tk.Label(presets_frame, text="Load Preset:").grid(row=0, column=0, sticky="w", pady=2)
        self.preset_var = tk.StringVar()
        self.preset_menu = tk.OptionMenu(presets_frame, self.preset_var, "No presets found")
        self.preset_menu.grid(row=0, column=1, padx=5, sticky="ew")
        self.load_preset_button = tk.Button(presets_frame, text="Load", command=self.load_preset)
        self.load_preset_button.grid(row=0, column=2, padx=5)

        tk.Label(presets_frame, text="Save Preset As:").grid(row=1, column=0, sticky="w", pady=2)
        self.save_preset_name_var = tk.StringVar()
        self.save_preset_entry = tk.Entry(presets_frame, textvariable=self.save_preset_name_var, width=15)
        self.save_preset_entry.grid(row=1, column=1, padx=5, sticky="ew")
        self.save_preset_button = tk.Button(presets_frame, text="Save", command=self.save_preset)
        self.save_preset_button.grid(row=1, column=2, padx=5)
        
        presets_frame.columnconfigure(1, weight=1)
        self.refresh_preset_list()

        info_text = "Subtitle window: Ctrl+C: Copy | Ctrl+S: Save | Esc: Stop"
        tk.Label(self.root, text=info_text, font=("Helvetica", 8), justify="center").pack(pady=(10, 5), side="bottom")

    def refresh_devices(self):
        self.device_list = sc.all_microphones(include_loopback=True)
        device_names = [mic.name for mic in self.device_list]
        menu = self.device_menu["menu"]
        menu.delete(0, "end")
        if not device_names:
            menu.add_command(label="No devices found", state="disabled")
            self.device_var.set("No devices found")
        else:
            for name in device_names:
                menu.add_command(label=name, command=lambda v=name: self.device_var.set(v))
            
            if self.config.selected_audio_device and self.config.selected_audio_device in device_names:
                self.device_var.set(self.config.selected_audio_device)
            else:
                preferred_device = find_audio_device()
                if preferred_device:
                    self.device_var.set(preferred_device.name)
                elif device_names:
                    self.device_var.set(device_names[0])
                else: 
                     self.device_var.set("No devices found")

    def get_selected_device_name(self):
        selected_name = self.device_var.get()
        return selected_name if selected_name != "No devices found" else None

    def stop_translator(self, event=None):
        if self.worker_threads:
            print("Stopping translator...")
            if self.stop_event: self.stop_event.set()
            for t in self.worker_threads: 
                t.join(timeout=1.0)
        self.worker_threads = []
        self.destroy_subtitle_window()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Status: Stopped", fg="red")
        print("Translator stopped.")

    def check_gui_queue(self):
        try:
            while True:
                msg_type, data = self.gui_queue.get_nowait()
                if msg_type == "subtitle": self.update_subtitle_text(data)
                elif msg_type == "model_loaded":
                    self.status_label.config(text="Status: Running", fg="green")
                    self.stop_button.config(state="normal")
                elif msg_type == "error":
                    self.status_label.config(text="Status: Error!", fg="red")
                    if self.subtitle_label: self.update_subtitle_text(f"FATAL ERROR: {data}")
                    self.stop_translator()
                    return
        except Empty: pass
        finally:
            if self.worker_threads: self.root.after(100, self.check_gui_queue)

    def create_subtitle_window(self):
        if self.subtitle_window: return
        self.subtitle_window = tk.Toplevel(self.root)
        self.subtitle_window.overrideredirect(True)
        self.subtitle_window.geometry(f"1000x300+{self.root.winfo_screenwidth() // 2 - 500}+{self.root.winfo_screenheight() // 2 - 150}")
        self.subtitle_window.wm_attributes("-topmost", True)
        self.subtitle_window.config(bg='green')
        self.subtitle_window.wm_attributes("-transparentcolor", "green")
        self.background_canvas = tk.Canvas(self.subtitle_window, bg='green', highlightthickness=0)
        self.background_canvas.pack(pady=40, padx=40, expand=True, fill="both")
        self.background_rect = self.background_canvas.create_rectangle(0, 0, 0, 0, outline="", width=0)
        self.subtitle_shadow_label = tk.Label(self.background_canvas, text="", wraplength=900, justify="center")
        self.subtitle_label = tk.Label(self.background_canvas, text="...", wraplength=900, justify="center")
        self.update_subtitle_style()
        self.subtitle_window.bind("<Escape>", self.stop_translator)
        for widget in [self.subtitle_label, self.subtitle_shadow_label, self.background_canvas]:
            widget.bind("<ButtonPress-1>", self.start_drag)
            widget.bind("<ButtonRelease-1>", self.stop_drag)
            widget.bind("<B1-Motion>", self.do_drag)
            widget.bind("<Control-c>", self.copy_subtitle)
            widget.bind("<Control-s>", self.save_subtitle_history)

    def destroy_subtitle_window(self):
        if self.subtitle_window:
            try: self.subtitle_window.destroy()
            except tk.TclError: pass
            self.subtitle_window = None

    def update_subtitle_text(self, text):
        if not self.subtitle_label or not self.subtitle_label.winfo_exists(): return
        if text != self.last_subtitle:
            self.last_subtitle = text
            display_text = text or "..."
            try:
                self.subtitle_label.config(text=display_text)
                if self.subtitle_shadow_label: self.subtitle_shadow_label.config(text=display_text)
            except tk.TclError: return
            if text.strip() and "FATAL ERROR" not in text:
                self.subtitle_history.append(f"[{datetime.now():%H:%M:%S}] {text}")
            
            self._update_background_size()
            self._resize_window_if_needed()

    def _update_background_size(self):
        if not self.subtitle_window or not self.background_canvas.winfo_exists(): return
        try:
            self.subtitle_window.update_idletasks()
            
            label_width = self.subtitle_label.winfo_reqwidth()
            label_height = self.subtitle_label.winfo_reqheight()
            
            min_width = 200
            min_height = 60
            label_width = max(label_width, min_width)
            label_height = max(label_height, min_height)
            
            canvas_width = self.background_canvas.winfo_width()
            canvas_height = self.background_canvas.winfo_height()
            
            padding_x = max(30, min(50, label_width * 0.1))
            padding_y = max(20, min(30, label_height * 0.15))
            
            x0 = (canvas_width - label_width) / 2 - padding_x
            y0 = (canvas_height - label_height) / 2 - padding_y
            x1 = (canvas_width + label_width) / 2 + padding_x
            y1 = (canvas_height + label_height) / 2 + padding_y
            
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(canvas_width, x1)
            y1 = min(canvas_height, y1)
            
            self.background_canvas.coords(self.background_rect, x0, y0, x1, y1)
            
            self.subtitle_label.place(relx=0.5, rely=0.5, anchor="center")
            
            if self.config.text_shadow and self.subtitle_shadow_label.winfo_exists():
                shadow_offset = 2
                self.subtitle_shadow_label.place(
                    x=self.subtitle_label.winfo_x() + shadow_offset, 
                    y=self.subtitle_label.winfo_y() + shadow_offset
                )
                self.background_canvas.tag_lower(self.background_rect)
                self.subtitle_shadow_label.lift()
                self.subtitle_label.lift()
            elif self.subtitle_shadow_label.winfo_exists():
                self.subtitle_shadow_label.place_forget()
                
        except tk.TclError as e:
            print(f"Error updating background size: {e}")
            pass

    def _resize_window_if_needed(self):
        if not self.subtitle_window or not self.subtitle_label.winfo_exists(): return
        
        try:
            label_width = self.subtitle_label.winfo_reqwidth()
            label_height = self.subtitle_label.winfo_reqheight()
            
            padding_x = 80
            padding_y = 80
            
            required_width = max(1000, label_width + padding_x)
            required_height = max(300, label_height + padding_y)
            
            current_width = self.subtitle_window.winfo_width()
            current_height = self.subtitle_window.winfo_height()
            
            width_diff = abs(required_width - current_width)
            height_diff = abs(required_height - current_height)
            
            if width_diff > 50 or height_diff > 50:
                x = self.subtitle_window.winfo_x()
                y = self.subtitle_window.winfo_y()
                
                self.subtitle_window.geometry(f"{required_width}x{required_height}+{x}+{y}")
                
                self.background_canvas.configure(width=required_width-80, height=required_height-80)
                
                new_wraplength = max(400, required_width - 100)
                self.subtitle_label.configure(wraplength=new_wraplength)
                if self.subtitle_shadow_label:
                    self.subtitle_shadow_label.configure(wraplength=new_wraplength)
                
                self.subtitle_window.update_idletasks()
                self._update_background_size()
                
        except tk.TclError as e:
            print(f"Error resizing window: {e}")
            pass

    def start_drag(self, event): self._drag_data["x"], self._drag_data["y"] = event.x, event.y
    def stop_drag(self, event): self._drag_data["x"], self._drag_data["y"] = 0, 0
    def do_drag(self, event):
        if self.subtitle_window:
            x = self.subtitle_window.winfo_pointerx() - self._drag_data["x"]
            y = self.subtitle_window.winfo_pointery() - self._drag_data["y"]
            self.subtitle_window.geometry(f"+{x}+{y}")

    def copy_subtitle(self, event=None):
        if self.last_subtitle:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.last_subtitle)
            print(f"📋 Copied: {self.last_subtitle}")

    def save_subtitle_history(self, event=None):
        if self.subtitle_history:
            filename = f"subtitles_{datetime.now():%Y%m%d_%H%M%S}.txt"
            try:
                with open(filename, 'w', encoding='utf-8') as f: f.write("\n".join(self.subtitle_history))
                print(f"💾 Saved history to: {filename}")
            except Exception as e: print(f"Error saving history: {e}")

    def on_close(self):
        print("Closing application...")
        
        self.stop_event.set() if self.stop_event else None

        self.apply_and_save_settings()

        print("--- Application session ended ---")
        if hasattr(sys, '__stdout__'):
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

        if self.log_file and not self.log_file.closed:
            self.log_file.close()
            self.log_file = None

        self.root.destroy()
        
    def open_log_window(self):
        if self.log_window and tk.Toplevel.winfo_exists(self.log_window):
            self.log_window.lift()
            return
        self.log_window = tk.Toplevel(self.root)
        self.log_window.title("Application Log")
        self.log_window.geometry("700x400")
        self.log_text_widget = tk.Text(self.log_window, wrap='word', font=("Consolas", 10), state='disabled')
        self.log_text_widget.pack(expand=True, fill='both', padx=5, pady=5)
        
        if self.log_buffer:
            self.log_text_widget.config(state='normal')
            self.log_text_widget.insert('1.0', "".join(self.log_buffer))
            self.log_text_widget.see('end')
            self.log_text_widget.config(state='disabled')
            
        self.log_window.protocol("WM_DELETE_WINDOW", self._on_log_close)

    def _on_log_close(self):
        if self.log_window:
            self.log_window.destroy()
            self.log_window = None
            self.log_text_widget = None

    def pick_bg_color(self):
        color = colorchooser.askcolor(title="Pick Background Color", initialcolor=self.config.subtitle_bg_color)
        if color and color[1]:
            self.config.subtitle_bg_color = color[1]
            self.bg_color_display.config(bg=color[1])
            self.update_subtitle_style()

    def pick_font_color(self):
        color = colorchooser.askcolor(title="Pick Font Color", initialcolor=self.config.subtitle_font_color)
        if color and color[1]:
            self.config.subtitle_font_color = color[1]
            self.font_color_display.config(bg=color[1])
            self.update_subtitle_style()

    def apply_and_save_settings(self, save_to_disk=True):
        try:
            self.config.volume_threshold = max(0.0, float(self.volume_var.get()))
            self.config.use_vad_filter = self.vad_var.get()
            self.config.vad_threshold = max(0.0, min(1.0, float(self.vad_threshold_var.get()) / 100.0))
            
            self.config.use_dynamic_chunking = self.dynamic_chunk_var.get()
            self.config.dynamic_silence_timeout = max(0.1, float(self.dyn_silence_var.get()))
            self.config.dynamic_max_chunk_duration = max(1.0, float(self.dyn_max_dur_var.get()))
            self.config.dynamic_min_speech_duration = max(0.1, float(self.dyn_min_speech_var.get()))

            self.config.font_size = int(self.font_var.get())
            self.config.window_opacity = max(0.0, min(1.0, float(self.opacity_var.get()) / 100.0))
            self.config.font_weight = self.font_weight_var.get()
            self.config.text_shadow = self.text_shadow_var.get()
            self.config.subtitle_bg_mode = self.bg_mode_var.get()
            
            self.config.selected_audio_device = self.device_var.get()
            if save_to_disk:
                self.config.save_config()
                print("Settings applied and saved.")
            else:
                print("Settings applied to current session.")
            return True
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Invalid Input", f"Please ensure all numeric fields are valid numbers.\nError: {e}")
            return False

    def start_translator(self):
        if self.worker_threads: return
        if not self.apply_and_save_settings(): return
        self.stats.reset()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Status: Loading model(s)...", fg="orange")
        self.root.update_idletasks()

        self.create_subtitle_window()
        self.stop_event = threading.Event()
        audio_queue = Queue(maxsize=20) 
        selected_device_name = self.get_selected_device_name()
        if selected_device_name is None:
            messagebox.showerror("Audio Error", "Could not find a valid audio device.")
            self.status_label.config(text="Status: Error!", fg="red")
            self.start_button.config(state="normal")
            return
            
        recorder = threading.Thread(target=recorder_thread, args=(self.stop_event, audio_queue, self.config, self.gui_queue, selected_device_name), daemon=True)
        processor = threading.Thread(target=processor_thread, args=(self.stop_event, audio_queue, self.config, self.stats, self.gui_queue), daemon=True)
        self.worker_threads = [recorder, processor]
        for t in self.worker_threads: t.start()
        self.check_gui_queue()

    def on_opacity_change(self, event=None):
        try: self.config.window_opacity = max(0.0, min(1.0, float(self.opacity_var.get()) / 100.0))
        except (ValueError, tk.TclError): pass
        self.update_subtitle_style()

    def on_font_weight_change(self, value=None):
        self.config.font_weight = self.font_weight_var.get()
        self.update_subtitle_style()

    def on_text_shadow_change(self):
        self.config.text_shadow = self.text_shadow_var.get()
        self.update_subtitle_style()

    def set_bg_mode(self, value=None):
        self.config.subtitle_bg_mode = self.bg_mode_var.get()
        self.update_subtitle_style()

    def update_subtitle_style(self, event=None):
        if not self.subtitle_window or not self.subtitle_label.winfo_exists(): return
        try:
            font_size = int(self.font_var.get())
            font_weight = self.font_weight_var.get()
            font_tuple = ("Helvetica", font_size, font_weight)
            self.subtitle_label.config(font=font_tuple, fg=self.config.subtitle_font_color, bg=self.config.subtitle_bg_color)
            if self.subtitle_shadow_label and self.subtitle_shadow_label.winfo_exists():
                self.subtitle_shadow_label.config(font=font_tuple, fg='#1c1c1c', bg=self.config.subtitle_bg_color)
            if self.background_canvas and self.background_rect:
                self.background_canvas.itemconfig(self.background_rect, fill=self.config.subtitle_bg_color, outline=self.config.border_color, width=self.config.border_width)
            if self.config.subtitle_bg_mode == 'transparent':
                self.subtitle_window.wm_attributes("-alpha", self.config.window_opacity)
            else:
                self.subtitle_window.wm_attributes("-alpha", 1.0)
            self._update_background_size()
        except (ValueError, tk.TclError): pass

    def on_device_select(self, *args):
        self.config.selected_audio_device = self.device_var.get()

    def download_model(self):
        self.status_label.config(text="Status: Downloading model...", fg="blue")
        self.download_button.config(state="disabled")
        self.start_button.config(state="disabled")
        self.stop_button.config(state="disabled")
        self.root.update_idletasks()
        
        def do_download():
            try:
                print(f"Starting model download...")
                ensure_model_downloaded(MODEL_ID, self.config.model_cache_dir)
                print("Model downloads/verifications complete.")
                self.gui_queue.put(("status_update", ("Status: All models are ready!", "green")))
            except Exception as e:
                error_msg = f"Download failed: {e}"
                print(error_msg)
                traceback.print_exc()
                self.gui_queue.put(("status_update", (f"Status: {error_msg}", "red")))
            finally:
                self.gui_queue.put(("download_finished", None))

        def process_download_queue():
            try:
                msg_type, data = self.gui_queue.get_nowait()
                if msg_type == "status_update":
                    text, color = data
                    self.status_label.config(text=text, fg=color)
                elif msg_type == "download_finished":
                    self.download_button.config(state="normal")
                    self.start_button.config(state="normal")
                    if not self.worker_threads: 
                        self.stop_button.config(state="disabled")
                    return
            except Empty:
                pass
            self.root.after(100, process_download_queue)

        threading.Thread(target=do_download, daemon=True).start()
        process_download_queue()

    def refresh_preset_list(self):
        preset_dir = "presets"
        if not os.path.exists(preset_dir):
            os.makedirs(preset_dir, exist_ok=True)
        
        preset_files = [f for f in os.listdir(preset_dir) if f.endswith('.json')]
        presets = [os.path.splitext(f)[0] for f in preset_files]
        
        menu = self.preset_menu["menu"]
        menu.delete(0, "end")
        
        if not presets:
            menu.add_command(label="No presets found", state="disabled")
            self.preset_var.set("No presets found")
        else:
            for preset_name in sorted(presets):
                menu.add_command(label=preset_name, command=lambda v=preset_name: self.preset_var.set(v))
            self.preset_var.set(presets[0])

    def save_preset(self):
        preset_name = self.save_preset_name_var.get().strip()
        if not preset_name:
            messagebox.showwarning("Warning", "Please enter a name for the preset.")
            return

        if not self.apply_and_save_settings(save_to_disk=False):
             messagebox.showerror("Error", "Could not save preset due to invalid settings.")
             return

        preset_data = {
            "volume_threshold": self.config.volume_threshold,
            "chunk_duration": self.config.chunk_duration,
            "language_code": self.config.language_code,
            "window_opacity": self.config.window_opacity,
            "font_size": self.config.font_size,
            "use_vad_filter": self.config.use_vad_filter,
            "vad_threshold": self.config.vad_threshold,
            "subtitle_bg_color": self.config.subtitle_bg_color,
            "subtitle_font_color": self.config.subtitle_font_color,
            "subtitle_bg_mode": self.config.subtitle_bg_mode,
            "font_weight": self.config.font_weight,
            "text_shadow": self.config.text_shadow,
            "border_width": self.config.border_width,
            "border_color": self.config.border_color,
            "output_mode": self.config.output_mode,
            "use_dynamic_chunking": self.config.use_dynamic_chunking,
            "dynamic_max_chunk_duration": self.config.dynamic_max_chunk_duration,
            "dynamic_silence_timeout": self.config.dynamic_silence_timeout,
            "dynamic_min_speech_duration": self.config.dynamic_min_speech_duration
        }
        
        preset_dir = "presets"
        os.makedirs(preset_dir, exist_ok=True)
        
        file_path = os.path.join(preset_dir, f"{preset_name}.json")
        try:
            with open(file_path, 'w') as f:
                json.dump(preset_data, f, indent=4)
            messagebox.showinfo("Success", f"Preset '{preset_name}' saved successfully.")
            self.refresh_preset_list()
            self.save_preset_name_var.set("")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save preset: {e}")

    def load_preset(self):
        preset_name = self.preset_var.get()
        if not preset_name or preset_name == "No presets found":
            messagebox.showwarning("Warning", "No preset selected.")
            return

        file_path = os.path.join("presets", f"{preset_name}.json")
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"Preset file not found: {file_path}")
            self.refresh_preset_list()
            return
            
        try:
            with open(file_path, 'r') as f:
                preset_data = json.load(f)

            for key, value in preset_data.items():
                setattr(self.config, key, value)

            self.volume_var.set(str(self.config.volume_threshold))
            self.opacity_var.set(str(int(self.config.window_opacity * 100)))
            self.font_var.set(str(self.config.font_size))
            self.font_weight_var.set(self.config.font_weight)
            self.vad_var.set(self.config.use_vad_filter)
            self.vad_threshold_var.set(str(int(self.config.vad_threshold * 100)))
            self.bg_mode_var.set(self.config.subtitle_bg_mode)
            self.bg_color_display.config(bg=self.config.subtitle_bg_color)
            self.font_color_display.config(bg=self.config.subtitle_font_color)
            self.text_shadow_var.set(self.config.text_shadow)
            
            if 'use_dynamic_chunking' in preset_data:
                self.dynamic_chunk_var.set(self.config.use_dynamic_chunking)
                self.dyn_silence_var.set(str(self.config.dynamic_silence_timeout))
                self.dyn_max_dur_var.set(str(self.config.dynamic_max_chunk_duration))
                self.dyn_min_speech_var.set(str(self.config.dynamic_min_speech_duration))
            
            self.update_subtitle_style()
            
            messagebox.showinfo("Success", f"Preset '{preset_name}' loaded.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load preset: {e}")