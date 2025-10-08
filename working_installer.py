#!/usr/bin/env python3
"""
Working LiveTranslator Installer
Fixed UI and Google Drive download
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys
import requests
import threading
import subprocess
import tempfile
from urllib.parse import urlparse

class WorkingInstaller:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LiveTranslator Installer")
        self.root.geometry("600x500")
        self.root.configure(bg='#f0f0f0')
        
        # Direct Google Drive download URL
        self.download_url = "https://drive.usercontent.google.com/download?id=1PYqcCvf9X9DPEuLyUweMFwIY5tbaGDhO&export=download&authuser=1&confirm=t&uuid=1762bec3-99fc-4543-9978-b72eeb774000&at=AKSUxGMnkxgxcMl51uI9PfIUIk9T:1759941564117"
        self.file_name = "LiveTranslator.exe"
        self.file_size_gb = 2.35
        
        self.install_path = os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), 'LiveTranslator')
        self.downloading = False
        
        self.create_ui()
        
    def create_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title = tk.Label(main_frame, text="LiveTranslator Installer", 
                        font=("Arial", 18, "bold"), bg='#f0f0f0', fg='#333')
        title.pack(pady=(0, 20))
        
        # Description
        desc = tk.Label(main_frame, 
                       text=f"This installer will download LiveTranslator ({self.file_size_gb:.1f}GB)\nfrom Google Drive and install it on your computer.\n\nFeatures:\n• Real-time speech translation\n• CUDA GPU acceleration\n• Multiple language support", 
                       font=("Arial", 11), bg='#f0f0f0', justify=tk.LEFT)
        desc.pack(pady=(0, 20))
        
        # Install path section
        path_frame = tk.LabelFrame(main_frame, text="Installation Directory", 
                                  font=("Arial", 10, "bold"), bg='#f0f0f0')
        path_frame.pack(fill=tk.X, pady=(0, 20))
        
        path_inner = tk.Frame(path_frame, bg='#f0f0f0')
        path_inner.pack(fill=tk.X, padx=10, pady=10)
        
        self.path_var = tk.StringVar(value=self.install_path)
        path_entry = tk.Entry(path_inner, textvariable=self.path_var, 
                             font=("Arial", 10), width=50)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_btn = tk.Button(path_inner, text="Browse...", 
                              command=self.browse_path, font=("Arial", 10))
        browse_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Progress section
        progress_frame = tk.LabelFrame(main_frame, text="Progress", 
                                      font=("Arial", 10, "bold"), bg='#f0f0f0')
        progress_frame.pack(fill=tk.X, pady=(0, 20))
        
        progress_inner = tk.Frame(progress_frame, bg='#f0f0f0')
        progress_inner.pack(fill=tk.X, padx=10, pady=10)
        
        self.status_label = tk.Label(progress_inner, text="Ready to install", 
                                    font=("Arial", 10), bg='#f0f0f0')
        self.status_label.pack(anchor=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_inner, mode='determinate', length=400)
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        
        self.progress_text = tk.Label(progress_inner, text="", 
                                     font=("Arial", 9), bg='#f0f0f0', fg='#666')
        self.progress_text.pack(anchor=tk.W, pady=(5, 0))
        
        # Button section - Fixed at bottom
        button_frame = tk.Frame(main_frame, bg='#f0f0f0', height=60)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(20, 0))
        button_frame.pack_propagate(False)
        
        # Create buttons with proper spacing
        self.install_btn = tk.Button(button_frame, text="Install", 
                                    command=self.start_install,
                                    bg="#4CAF50", fg="white", 
                                    font=("Arial", 12, "bold"),
                                    width=12, height=2)
        self.install_btn.pack(side=tk.LEFT, padx=(50, 10))
        
        self.cancel_btn = tk.Button(button_frame, text="Cancel", 
                                   command=self.cancel_install,
                                   bg="#f44336", fg="white",
                                   font=("Arial", 12, "bold"),
                                   width=12, height=2)
        self.cancel_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Test button to verify UI works
        test_btn = tk.Button(button_frame, text="Test", 
                            command=self.test_click,
                            bg="#2196F3", fg="white",
                            font=("Arial", 10),
                            width=8, height=2)
        test_btn.pack(side=tk.RIGHT, padx=(0, 50))
        
    def test_click(self):
        messagebox.showinfo("Test", "UI is working! Buttons are clickable.")
        
    def browse_path(self):
        folder = filedialog.askdirectory(initialdir=os.path.dirname(self.install_path))
        if folder:
            self.install_path = os.path.join(folder, 'LiveTranslator')
            self.path_var.set(self.install_path)
    
    def start_install(self):
        if self.downloading:
            return
            
        # Confirm installation
        result = messagebox.askyesno("Confirm Installation", 
                                   f"This will download {self.file_size_gb:.1f}GB and install LiveTranslator to:\n{self.install_path}\n\nThis may take a while depending on your internet speed.\n\nContinue?")
        if not result:
            return
        
        self.downloading = True
        self.install_btn.config(state="disabled", text="Installing...")
        self.cancel_btn.config(text="Cancel Download")
        
        # Start download in thread
        thread = threading.Thread(target=self.download_file, daemon=True)
        thread.start()
    
    def download_file(self):
        try:
            # Create install directory
            os.makedirs(self.install_path, exist_ok=True)
            file_path = os.path.join(self.install_path, self.file_name)
            
            self.update_status("Connecting to Google Drive...")
            
            # Use requests session with proper headers
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            # Direct download using the working URL
            self.update_status("Starting download...")
            response = session.get(self.download_url, stream=True, timeout=30)
            
            if response.status_code != 200:
                raise Exception(f"Download failed: HTTP {response.status_code}")
            
            # Get file size from headers
            total_size = int(response.headers.get('content-length', 0))
            if total_size == 0:
                total_size = int(self.file_size_gb * 1024 * 1024 * 1024)  # Use expected size as fallback
                self.update_status("Warning: Could not determine file size, using estimated size...")
            
            # Verify we're getting a binary file, not HTML
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                raise Exception("Received HTML instead of file. The download link may have expired.")
            
            self.update_status(f"Downloading {total_size / (1024*1024*1024):.1f}GB file...")
            
            # Download with progress
            downloaded = 0
            chunk_size = 8192
            
            self.update_status("Downloading LiveTranslator...")
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not self.downloading:  # Check for cancellation
                        f.close()
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        return
                    
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            self.update_progress(progress, downloaded, total_size)
            
            # Create shortcuts
            self.update_status("Creating shortcuts...")
            self.create_shortcut(file_path)
            
            # Complete
            self.update_status("Installation completed successfully!")
            self.progress_bar.config(value=100)
            
            messagebox.showinfo("Success", 
                              f"LiveTranslator installed successfully!\n\nLocation: {file_path}\n\nA desktop shortcut has been created.")
            
            # Ask to launch
            if messagebox.askyesno("Launch", "Launch LiveTranslator now?"):
                subprocess.Popen([file_path])
            
            self.root.quit()
            
        except Exception as e:
            self.update_status(f"Installation failed: {str(e)}")
            messagebox.showerror("Error", f"Installation failed:\n{str(e)}")
            self.reset_ui()
    
    def create_shortcut(self, exe_path):
        try:
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            shortcut_path = os.path.join(desktop, "LiveTranslator.lnk")
            
            # PowerShell script to create shortcut
            ps_script = f'''
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
$Shortcut.TargetPath = "{exe_path}"
$Shortcut.WorkingDirectory = "{os.path.dirname(exe_path)}"
$Shortcut.Description = "LiveTranslator - Real-time Speech Translation"
$Shortcut.Save()
'''
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ps1', delete=False) as f:
                f.write(ps_script)
                ps_file = f.name
            
            subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File', ps_file], 
                          capture_output=True, check=False)
            os.unlink(ps_file)
            
        except Exception as e:
            print(f"Shortcut creation failed: {e}")
    
    def update_status(self, message):
        self.root.after(0, lambda: self.status_label.config(text=message))
    
    def update_progress(self, percentage, downloaded, total):
        def update():
            self.progress_bar.config(value=percentage)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            self.progress_text.config(text=f"{downloaded_mb:.1f} MB / {total_mb:.1f} MB ({percentage:.1f}%)")
        self.root.after(0, update)
    
    def cancel_install(self):
        if self.downloading:
            if messagebox.askyesno("Cancel", "Cancel the download?"):
                self.downloading = False
                self.reset_ui()
        else:
            self.root.quit()
    
    def reset_ui(self):
        self.downloading = False
        self.install_btn.config(state="normal", text="Install")
        self.cancel_btn.config(text="Cancel")
        self.progress_bar.config(value=0)
        self.progress_text.config(text="")
        self.status_label.config(text="Ready to install")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    try:
        app = WorkingInstaller()
        app.run()
    except Exception as e:
        messagebox.showerror("Error", f"Installer error: {str(e)}")
        sys.exit(1)