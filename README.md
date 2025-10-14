#  HoloLiveTL - Real-Time Japanese Translation for VTuber Streams

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)
![AI](https://img.shields.io/badge/AI-Whisper%20%2B%20VAD-orange.svg)

*Real-time Japanese to English subtitle overlay for VTuber livestreams and Japanese content*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [VTuber Setup](#-vtuber-setup-guide) • [Building](#-building)

</div>

## 🌟 Overview

HoloLiveTL is a powerful real-time translation application that captures your computer's audio, translates spoken Japanese into English text, and displays it as a customizable subtitle overlay. Perfect for watching Japanese VTuber livestreams, anime, or any Japanese content without native English subtitles.


## ✨ Features

### 🎯 Core Translation Features
- **Real-Time Translation**: Minimal delay Japanese-to-English translation
- **Advanced AI Models**: Uses Kotoba Whisper for high-quality, context-aware translation
- **Dynamic Audio Chunking**: Intelligent VAD-based speech detection for natural subtitle timing
- **Hallucination Filtering**: Removes common ASR artifacts and false positives
- **VTuber Optimized**: Special filters and optimizations for VTuber content

### 🎨 Customizable Overlay
- **Movable Subtitle Window**: Drag and position anywhere on screen
- **Full Appearance Control**: Font size, weight, color, and shadow options
- **Background Modes**: Transparent or solid background with opacity control
- **Real-time Updates**: Instant subtitle appearance changes

### 🔧 Advanced Audio Processing
- **Smart Device Detection**: Auto-detects loopback devices (Stereo Mix, VB-Cable)
- **Multiple Audio Sources**: Support for virtual audio cables and system audio
- **Noise Filtering**: Advanced audio preprocessing for better recognition
- **Volume Threshold Control**: Adjustable sensitivity for different audio levels

### 💾 Convenience Features
- **Settings Persistence**: All configurations automatically saved
- **Preset System**: Save and load different configurations for various use cases
- **Subtitle History**: Copy last subtitle or save entire session transcript
- **Performance Monitoring**: Built-in statistics and logging
- **GPU Acceleration**: CUDA support for optimal performance

## 🚀 Installation

### Prerequisites
- **Python 3.8+**
- **NVIDIA GPU with CUDA** (highly recommended for real-time performance)
- **Virtual Audio Device** (VB-Cable recommended) or Stereo Mix enabled

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shemo37/HoloLiveTL.git
   cd HoloLiveTL
   ```

2. **Install PyTorch with CUDA:**
   ```bash
   # Visit https://pytorch.org/get-started/locally/ for your specific CUDA version
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python LiveTranslate.py
   ```

### First Run
- Click **"Download Model"** to pre-download AI models (~1-2GB)
- Configure your audio device (see setup guides below)
- Click **"Start"** to begin translation

## 🎮 Usage

### Basic Operation
1. **Configure Audio Source**: Select your loopback device from the dropdown
2. **Adjust Settings**: Customize translation and appearance settings
3. **Start Translation**: Click "Start" - a subtitle overlay will appear
4. **Position Overlay**: Drag the subtitle window to your preferred location
5. **Enjoy**: Play Japanese content and watch real-time translations appear

### Subtitle Window Controls
- **Drag**: Click and drag to reposition
- **Ctrl+C**: Copy current subtitle to clipboard
- **Ctrl+S**: Save entire session transcript
- **Esc**: Stop translation immediately


### Optimal Settings for VTuber Streams

```
✅ Dynamic Chunking: Enabled
   - Silence Timeout: 0.8-1.2s
   - Max Record Time: 12-15s
   - Min Speech Duration: 0.3s

✅ VAD Filter: Enabled (50-60% threshold)
✅ Volume Threshold: 0.003-0.005
✅ Hallucination Filtering: Enabled
```

### Stream-Specific Presets
- **Chatting Streams**: Higher confidence (75-80%)
- **Gaming Streams**: Enable noise filtering
- **Singing Streams**: Lower confidence during songs
- **ASMR Streams**: Increased sensitivity settings

## 🔧 Configuration

### Dynamic Chunking (Recommended)
- **Enable Dynamic Chunks**: Uses VAD for natural speech boundaries
- **Silence Timeout**: Pause duration before processing (0.8-1.5s)
- **Max Record Time**: Safety limit for long recordings (10-15s)
- **Min Speech Duration**: Filters very short sounds (0.3s)

### Audio Filters
- **Volume Threshold**: Minimum audio level to trigger processing
- **VAD Filter**: Secondary speech detection for accuracy
- **VAD Threshold**: Sensitivity of voice activity detection

### Appearance
- **Font Settings**: Size, weight (normal/bold), color
- **Background**: Transparent or solid with opacity control
- **Text Shadow**: Improves readability over any background
- **Window Opacity**: Overall transparency level


## 🏗️ Building

### Quick Build
```bash
python advanced_build.py    # Complete distribution package
python build_exe.py         # Basic executable
quick_build.bat             # Windows batch build
```

### Manual Build
```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name "LiveTranslate" LiveTranslate.py
```

See [BUILD_README.md](BUILD_README.md) for detailed building instructions.



### Getting Help
- Check the console output for detailed error messages
- Review `translator_app.log` for debugging information
- Ensure all dependencies are properly installed

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kotoba Technologies** for the excellent bilingual Whisper model
- **Silero Team** for the VAD model
- **Hugging Face** for the transformers library
- **VTuber Community** for inspiration and feedback
- **OpenAI** for the original Whisper architecture

## 🌟 Star History

If you find this project helpful, please consider giving it a star! ⭐

---
