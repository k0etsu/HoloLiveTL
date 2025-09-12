

# LiveTranslate.py: Real-Time Japanese to English Subtitle Overlay

LiveTranslate.py is a real-time, on-screen translator that captures your computer's audio, translates spoken Japanese into English text, and displays it as a customizable subtitle overlay. It's perfect for watching Japanese livestreams, videos, or online meetings without native English subtitles.

The application uses the powerful `kotoba-tech/kotoba-whisper-bilingual-v1.0` model for high-quality, context-aware translation and features an intelligent, VAD-based (Voice Activity Detection) audio chunking system for efficient and natural-looking subtitles.

## ✨ Features

  * **Real-Time Translation**: Translates Japanese speech to English text with minimal delay.
  * **On-Screen Overlay**: Displays subtitles in a clean, movable, and customizable window that can be placed over any application.
  * **Dynamic Audio Chunking**: Uses Silero VAD to intelligently detect speech, creating perfectly timed subtitle chunks based on natural pauses. This is more efficient and accurate than fixed-time slicing.
  * **Highly Customizable Subtitles**:
      * Adjust font size, weight (bold/normal), and color.
      * Change the background color and opacity.
      * Choose between a solid background or a transparent window.
      * Enable a text shadow for better readability.
  * **Audio Device Selection**: Automatically detects loopback audio devices (like "Stereo Mix" or "VB-CABLE") but allows manual selection from all available sound inputs.
  * **Hallucination Filtering**: Includes filters to remove common ASR model hallucinations (e.g., "Thanks for watching\!", "Please subscribe").
  * **Easy-to-Use GUI**: A simple control panel built with Tkinter to manage all settings and start/stop the translator.
  * **Settings & Presets**: All your settings are automatically saved. You can also save and load different configurations as presets for various use cases.
  * **Performance**: Optimized for real-time use, with GPU (CUDA) acceleration supported for the best performance.

## ⚙️ How It Works

The application operates through a multi-threaded pipeline:

1.  **Audio Capture**: A **Recorder Thread** listens to the selected audio device (e.g., your system's audio output).
2.  **Voice Activity Detection (VAD)**: In the recommended **Dynamic Mode**, this thread uses a VAD model to listen for speech. It records as long as speech is detected, then sends the complete voice segment for processing once a pause is heard. This ensures that full sentences are captured.
3.  **Processing**: A **Processor Thread** receives the audio chunk. It runs the powerful Whisper ASR model to transcribe and translate the Japanese speech into English text.
4.  **Filtering & Display**: The resulting text is filtered for common hallucinations and then sent to the **GUI Thread**, which updates the on-screen subtitle overlay instantly.

## 🚀 Installation

### Prerequisites

  * **Python 3.8+**
  * **Git** (for cloning the repository)
  * **An NVIDIA GPU with CUDA** is *highly recommended* for smooth real-time performance. The application will run on a CPU, but it may be significantly slower.
  * **(Optional but Recommended) A virtual audio device** like [VB-CABLE](https://vb-audio.com/Cable/) to easily route desktop audio.

### Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Shemo37/HoloLiveTL.git
    cd LiveTranslate.py
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install PyTorch with CUDA support:**
    Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to get the correct installation command for your specific CUDA version. An example command is:

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

4.  **Install the remaining dependencies:**
    A `requirements.txt` is provided for convenience.

    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Usage

1.  **Configure Your Audio:**

      * **Method A (Recommended - VB-CABLE):** Install VB-CABLE. Set `CABLE Input` as your default Windows audio playback device. This will route all desktop sound through the virtual cable.
      * **Method B (Stereo Mix):** Enable the "Stereo Mix" or "What U Hear" recording device in your Windows sound settings and set it as the default recording device.

2.  **Run the application:**

    ```bash
    python LiveTranslate.py
    ```

3.  **Using the Control Panel:**

      * On the first run, it's recommended to click the **Download Model** button. This will pre-download all necessary AI models and avoid a long delay when you first click "Start". Watch the console for progress.
      * Select your audio loopback device (e.g., `CABLE Output (VB-Audio Virtual Cable)` or `Stereo Mix`) from the dropdown menu.
      * Adjust any settings as needed. The defaults are a great starting point.
      * Click **Start**.
      * A semi-transparent subtitle window will appear. You can click and drag this window to position it anywhere on your screen.
      * Play any Japanese audio on your computer. The English translation will appear in the overlay window.
      * Click **Stop** to end the translation.

### Subtitle Window Hotkeys

  * `Ctrl + C`: Copy the last subtitle to the clipboard.
  * `Ctrl + S`: Save the entire transcript history of the current session to a `.txt` file.
  * `Esc`: Immediately stop the translator.

## 🔧 Configuration Tips

  * **Dynamic Chunking Settings**: This is the most important section for performance.
      * **Enable Dynamic Chunks**: Keep this checked for the best results.
      * **Silence Timeout (s)**: How long of a pause the app should wait for before processing a sentence. `0.8` - `1.5` seconds is usually good.
      * **Max Record Time (s)**: A safeguard to process a chunk even if there are no pauses. Prevents infinitely long chunks.
  * **Audio Filter Settings**:
      * **Volume Threshold**: Adjust this if the app is picking up background noise or missing quiet speech. Use the console output (which shows RMS volume) to find a good value.
      * **Enable VAD Filter**: Acts as a secondary check to ensure a chunk contains speech. It's useful in both dynamic and fixed modes.
  * **Appearance**:
      * **BG Mode**: `transparent` makes the entire window see-through (controlled by Opacity), while `solid` makes only the text background see-through.
      * **Text Shadow**: Highly recommended for readability on any background.

## ⚠️ Troubleshooting

  * **No Audio Devices Found**: Ensure you have enabled "Stereo Mix" or installed a virtual audio cable.
  * **Error During Startup / Model Download**: Your internet connection may be blocked by a firewall, or a dependency might be missing. Check the `translator_app.log` file for detailed error messages.
  * **Laggy / Slow Translation**:
      * Confirm that PyTorch was installed correctly with CUDA support. The console will print `Using device: CUDA:0` if it's working.
      * Close other resource-intensive applications.
      * If you are on a CPU, translation will inherently be slower.
  * **Poor Translation Quality**: The quality depends entirely on the underlying Whisper model. Clear, non-overlapping speech with low background noise will produce the best results.

## 📄 License

This project is licensed under the MIT License

-----


