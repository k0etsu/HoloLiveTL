import time
import numpy as np
import soundcard as sc
import threading
from queue import Queue, Empty
import torch
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, Toplevel, Label, colorchooser
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import string
import json
import os
import traceback
import sys
import io
from collections import deque
import requests
from tqdm import tqdm
import hashlib
import gc
import shutil
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")

# Environment optimizations
os.environ['TRANSFORMERS_VERBOSITY'] = 'warning'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

try:
    import torchaudio
except ImportError:
    messagebox.showerror("Dependency Error", "torchaudio not found. Please run 'pip install torchaudio' in your terminal.")
    exit()

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    hf_hub_download = None
    messagebox.showerror("Dependency Error", "huggingface_hub not found. Please run 'pip install huggingface_hub' in your terminal.")
    exit()

from scipy.signal import butter, lfilter

# Constants
MODEL_ID = "kotoba-tech/kotoba-whisper-bilingual-v1.0"
SAMPLE_RATE = 16000
CHUNK_DURATION = 5 
LANGUAGE_CODE = "en"
VOLUME_THRESHOLD = 0.003
USE_VAD_FILTER = True
VAD_THRESHOLD = 0.25
DEFAULT_BG_COLOR = '#282828'
DEFAULT_FONT_COLOR = '#FFFFFF'
DEFAULT_BG_MODE = 'transparent'
DEFAULT_WINDOW_OPACITY = 0.85

# Model cache directory
MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "translator_models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Hallucination filters
SUBSTRING_HALLUCINATION_FILTER = [
    "thank you for watching", "thanks for watching", "don't forget",
    "to subscribe", "subscribe", "bell icon", "see you next time",
    "in the next video", "like and subscribe", "hit the bell",
    "comment below", "let me know", "see you later", "as a language model", 
    "provide more context", "i'm an ai", "i cannot", "i don't have access",
    "please provide", "more information", "context is needed"
]

EXACT_MATCH_HALLUCINATION_FILTER = {
    "i see", "i understand", "i know", "i'm sorry", "thank you", "thanks",
    "you're welcome", "okay", "ok", "all right", "alright", "got it", "right",
    "of course", "excuse me", "please", "the end", "hello", "hi", "hey",
    "um", "uh", "hmm", "well", "so", "like", "you know", "that's right", 
    "such as", "i see, i see", "alright i see", "ah i see", "sound good", 
    "oh i see", "heav-ho", "mm-hmm", "uh-huh", "yeah", "yep", "nope", "nah"
}

PRESERVE_SOUNDS = {
    "ah", "oh", "wow", "no", "yes", "stop", "help", "wait", "go", "come",
    "aah", "ooh", "eeh", "kyaa", "waa", "haa", "yaa", "noo", "ahh", "ohh",
    "nya", "uwu", "owo", "ara", "ehe", "ehehe", "hehe", "hihi", "hoho",
    "yay", "yey", "yup", "nope", "mhm", "mmm", "hmm", "huh", "eh",
    "gg", "nice", "good", "bad", "fail", "win", "lose", "dead", "alive",
    "hai", "iie", "sou", "nani", "mou", "demo", "kedo", "desu", "masu"
}

QUALITY_INDICATORS = {
    "repetitive_patterns": [r"(.{1,10})\1{3,}", r"(\w+\s+)\1{2,}"],
    "nonsense_patterns": [r"[a-z]{15,}", r"\b\w{1}\s+\w{1}\s+\w{1}\b"],
    "filler_heavy": [r"\b(um|uh|ah|eh|mm)\b.*\b(um|uh|ah|eh|mm)\b.*\b(um|uh|ah|eh|mm)\b"]
}

gui_queue = Queue()
config = None
stats = None

def download_with_progress(url, destination, description="Downloading"):
    """Download file with progress bar"""
    if os.path.exists(destination):
        print(f"✅ {description} already exists at {destination}")
        return destination
    
    print(f"📥 {description} from {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=description,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        
        print(f"✅ {description} completed")
        return destination
    except Exception as e:
        print(f"❌ Error downloading {description}: {e}")
        if os.path.exists(destination):
            os.remove(destination)
        raise

def ensure_model_downloaded():
    """Download models if not present to avoid bundling them in the exe"""
    model_dir = os.path.join(MODEL_CACHE_DIR, "whisper_model")
    vad_dir = os.path.join(MODEL_CACHE_DIR, "vad_model")
    
    # Check if model directory exists and has required files
    model_files_exist = False
    if os.path.exists(model_dir):
        required_files = ["model.safetensors", "config.json", "preprocessor_config.json"]
        model_files_exist = all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)
    
    # Download Whisper model if needed
    if not model_files_exist:
        print("📥 Downloading Whisper model...")
        
        # Remove existing directory if incomplete
        if os.path.exists(model_dir):
            try:
                shutil.rmtree(model_dir)
                print(f"🗑️ Removed incomplete model directory: {model_dir}")
            except Exception as e:
                print(f"⚠️ Could not remove existing model directory: {e}")
        
        try:
            # Try using snapshot_download for a complete model download
            print("📥 Using Hugging Face snapshot download...")
            model_dir = snapshot_download(
                repo_id=MODEL_ID,
                cache_dir=MODEL_CACHE_DIR,
                local_dir=model_dir,
                resume_download=True
            )
            print(f"✅ Model downloaded to: {model_dir}")
        except Exception as e:
            print(f"⚠️ Snapshot download failed: {e}")
            print("📥 Falling back to manual download...")
            
            # Fallback to manual download
            os.makedirs(model_dir, exist_ok=True)
            
            # Download essential files
            base_url = f"https://huggingface.co/{MODEL_ID}/resolve/main"
            
            files_to_download = [
                ("model.safetensors", "model.safetensors"),
                ("config.json", "config.json"),
                ("preprocessor_config.json", "preprocessor_config.json"),
                ("tokenizer.json", "tokenizer.json"),
                ("tokenizer_config.json", "tokenizer_config.json"),
                ("generation_config.json", "generation_config.json"),
                ("special_tokens_map.json", "special_tokens_map.json"),
                ("vocab.json", "vocab.json")
            ]
            
            for filename, local_name in files_to_download:
                url = f"{base_url}/{filename}"
                destination = os.path.join(model_dir, local_name)
                download_with_progress(url, destination, f"Downloading {filename}")
    
    # Download VAD model
    vad_path = os.path.join(vad_dir, "silero_vad.jit")
    if not os.path.exists(vad_path):
        print("📥 Downloading VAD model...")
        os.makedirs(vad_dir, exist_ok=True)
        
        # Try multiple URLs for the VAD model
        vad_urls = [
            "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.jit",
            "https://github.com/snakers4/silero-vad/raw/main/src/silero_vad/data/silero_vad.jit",
            "https://models.silero.ai/models/vad/silero_vad.jit",
            "https://github.com/snakers4/silero-vad/releases/download/v3.1/silero_vad.jit"
        ]
        
        vad_downloaded = False
        for vad_url in vad_urls:
            try:
                print(f"📥 Trying VAD model URL: {vad_url}")
                download_with_progress(vad_url, vad_path, "Downloading VAD model")
                vad_downloaded = True
                break
            except Exception as e:
                print(f"⚠️ Failed to download from {vad_url}: {e}")
                if os.path.exists(vad_path):
                    os.remove(vad_path)
                continue
        
        if not vad_downloaded:
            print("⚠️ Could not download VAD model from any source. VAD filtering will be disabled.")
            # Create a dummy file to prevent repeated download attempts
            with open(vad_path + ".failed", "w") as f:
                f.write("VAD model download failed")
    
    return model_dir, vad_dir

def get_file_hash(filepath):
    """Get SHA256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

class Config:
    def __init__(self):
        self.config_file = "translator_config.json"
        self.load_config()

    def load_config(self):
        default_config = {
            "volume_threshold": VOLUME_THRESHOLD,
            "chunk_duration": CHUNK_DURATION,
            "language_code": LANGUAGE_CODE,
            "window_opacity": DEFAULT_WINDOW_OPACITY,
            "font_size": 24,
            "use_vad_filter": USE_VAD_FILTER,
            "vad_threshold": VAD_THRESHOLD,
            "subtitle_bg_color": DEFAULT_BG_COLOR,
            "subtitle_font_color": DEFAULT_FONT_COLOR,
            "subtitle_bg_mode": DEFAULT_BG_MODE,
            "font_weight": "bold",
            "text_shadow": True,
            "border_width": 2,
            "border_color": "#000000",
            "output_mode": "translate",
            "selected_audio_device": None,
            "use_dynamic_chunking": True,
            "dynamic_max_chunk_duration": 15.0,
            "dynamic_silence_timeout": 1.2,
            "dynamic_min_speech_duration": 0.3,
            "model_cache_dir": MODEL_CACHE_DIR
        }
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                print(f"Error loading config: {e}")
        self.__dict__.update(default_config)

    def save_config(self):
        config_data = {k: v for k, v in self.__dict__.items() if k != 'config_file'}
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

class TranslatorStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.chunks_processed = 0
        self.translations_made = 0
        self.hallucinations_filtered = 0
        self.total_processing_time = 0

    def add_chunk(self, processing_time, had_translation, was_hallucination):
        self.chunks_processed += 1
        self.total_processing_time += processing_time
        if had_translation:
            if was_hallucination:
                self.hallucinations_filtered += 1
            else:
                self.translations_made += 1

def find_audio_device(selected_device_name=None):
    print("🔍 Searching for audio capture devices...")
    all_mics = sc.all_microphones(include_loopback=True)
    if not all_mics:
        print("❌ No audio devices found at all.")
        return None

    if selected_device_name:
        for mic in all_mics:
            if mic.name == selected_device_name:
                print(f"🎚️ Using selected device: '{mic.name}'")
                return mic
        print(f"⚠️ Could not find previously selected device '{selected_device_name}'. Searching for alternatives.")

    preferred_names = ["cable", "stereo mix", "what u hear", "loopback", "virtual"]
    for name in preferred_names:
        for mic in all_mics:
            if name in mic.name.lower():
                print(f"✅ Found preferred capture device: '{mic.name}'")
                return mic
    try:
        default_mic = sc.default_microphone(include_loopback=True)
        print(f"⚠️ Using default system loopback device: '{default_mic.name}'")
        return default_mic
    except Exception:
        print(f"⚠️ No default loopback found. Falling back to first available device: '{all_mics[0].name}'")
        return all_mics[0]

def recorder_thread(stop_event, audio_queue, selected_device_name=None):
    if config.use_dynamic_chunking:
        print("🎙️ Recorder thread started (Dynamic Chunking Mode).")
        dynamic_recorder_thread(stop_event, audio_queue, selected_device_name)
    else:
        print("🎙️ Recorder thread started (Fixed Chunk Mode).")
        fixed_recorder_thread(stop_event, audio_queue, selected_device_name)

def fixed_recorder_thread(stop_event, audio_queue, selected_device_name):
    try:
        target_mic = find_audio_device(selected_device_name)
        if target_mic is None:
            raise RuntimeError("No audio devices found. Cannot start recording.")
        with target_mic.recorder(samplerate=SAMPLE_RATE, channels=1) as mic:
            while not stop_event.is_set():
                data = mic.record(numframes=int(SAMPLE_RATE * config.chunk_duration))
                if not stop_event.is_set():
                    audio_queue.put(data)
    except Exception as e:
        print(f"🔴 Recorder Thread Error (Fixed): {e}")
        traceback.print_exc()
        gui_queue.put(("error", "Audio device error! Check console."))
    finally:
        print("🎙️ Recorder thread stopped (Fixed).")

def dynamic_recorder_thread(stop_event, audio_queue, selected_device_name):
    try:
        target_mic = find_audio_device(selected_device_name)
        if target_mic is None:
            raise RuntimeError("No audio devices found. Cannot start recording.")
            
        print("🎙️ [Dynamic] Loading Silero VAD model for recorder...")
        torch.set_num_threads(1)
        
        # Load VAD from cache
        vad_path = os.path.join(config.model_cache_dir, "vad_model", "silero_vad.jit")
        vad_failed_path = vad_path + ".failed"
        
        if not os.path.exists(vad_path) and not os.path.exists(vad_failed_path):
            print("🎙️ [Dynamic] VAD model not found, downloading...")
            ensure_model_downloaded()
        
        if os.path.exists(vad_failed_path):
            print("⚠️ [Dynamic] VAD model download previously failed. Using volume-based detection only.")
            vad_model = None
        elif not os.path.exists(vad_path):
            print("⚠️ [Dynamic] VAD model file not found. Using volume-based detection only.")
            vad_model = None
        else:
            try:
                vad_model = torch.jit.load(vad_path, map_location='cpu')
                print("🎙️ [Dynamic] VAD model loaded.")
            except Exception as e:
                print(f"⚠️ [Dynamic] Failed to load VAD model: {e}. Using volume-based detection only.")
                vad_model = None

        VAD_FRAME_DURATION_MS = 30
        VAD_FRAME_SIZE = int(SAMPLE_RATE * VAD_FRAME_DURATION_MS / 1000)

        is_speaking = False
        speech_buffer = []
        silence_frames_after_speech = 0
        
        silence_timeout_frames = int(config.dynamic_silence_timeout * 1000 / VAD_FRAME_DURATION_MS)
        max_chunk_frames = int(config.dynamic_max_chunk_duration * 1000 / VAD_FRAME_DURATION_MS)

        with target_mic.recorder(samplerate=SAMPLE_RATE, channels=1) as mic:
            print("🎙️ [Dynamic] Now listening...")
            while not stop_event.is_set():
                frame_data = mic.record(numframes=VAD_FRAME_SIZE)
                
                rms = np.sqrt(np.mean(frame_data ** 2))
                peak_level = np.max(np.abs(frame_data))
                is_loud_sound = peak_level > 0.1
                
                if rms < config.volume_threshold and not is_loud_sound:
                    is_speech = False
                else:
                    if vad_model is not None:
                        # Use VAD model if available
                        audio_tensor = torch.from_numpy(frame_data.flatten()).float()
                        if len(audio_tensor) < 512:
                            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, 512 - len(audio_tensor)))
                        speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()
                        
                        vad_threshold = config.vad_threshold * 0.5 if is_loud_sound else config.vad_threshold
                        is_speech = speech_prob > vad_threshold or is_loud_sound
                        
                        if is_loud_sound:
                            print(f"🔊 Loud sound detected! Peak: {peak_level:.3f}, RMS: {rms:.3f}, VAD: {speech_prob:.3f}")
                    else:
                        # Fallback to volume-based detection only
                        is_speech = rms > config.volume_threshold or is_loud_sound
                        
                        if is_loud_sound:
                            print(f"🔊 Loud sound detected! Peak: {peak_level:.3f}, RMS: {rms:.3f} (VAD disabled)")

                if is_speaking:
                    speech_buffer.append(frame_data)
                    if is_speech:
                        silence_frames_after_speech = 0
                    else:
                        silence_frames_after_speech += 1
                    
                    chunk_ended = (silence_frames_after_speech > silence_timeout_frames) or \
                                  (len(speech_buffer) > max_chunk_frames)

                    if chunk_ended:
                        audio_chunk = np.concatenate(speech_buffer)
                        chunk_duration_s = len(audio_chunk) / SAMPLE_RATE
                        chunk_peak = np.max(np.abs(audio_chunk))
                        is_loud_chunk = chunk_peak > 0.1
                        
                        min_duration = config.dynamic_min_speech_duration * 0.5 if is_loud_chunk else config.dynamic_min_speech_duration
                        min_samples = int(SAMPLE_RATE * 0.5) if is_loud_chunk else int(SAMPLE_RATE * 1.0)
                        
                        if chunk_duration_s > min_duration and len(audio_chunk) >= min_samples:
                            chunk_type = "LOUD" if is_loud_chunk else "speech"
                            print(f"🎤 Detected {chunk_type} chunk of {chunk_duration_s:.2f}s (peak: {chunk_peak:.3f}). Sending for processing.")
                            audio_queue.put(audio_chunk)
                        else:
                            print(f"⏩ Skipped short chunk: {chunk_duration_s:.2f}s (peak: {chunk_peak:.3f})")
                        
                        is_speaking = False
                        speech_buffer = []
                        silence_frames_after_speech = 0

                elif is_speech:
                    is_speaking = True
                    speech_buffer.append(frame_data)
                    silence_frames_after_speech = 0

    except Exception as e:
        print(f"🔴 Recorder Thread Error (Dynamic): {e}")
        traceback.print_exc()
        gui_queue.put(("error", "Audio device error! Check console."))
    finally:
        print("🎙️ Recorder thread stopped (Dynamic).")

def highpass_filter(data, cutoff=100, fs=16000, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y

def normalize_audio(audio_data):
    """Normalize audio to improve recognition accuracy while preserving loud sounds"""
    audio_data = audio_data - np.mean(audio_data)
    
    rms = np.sqrt(np.mean(audio_data ** 2))
    if rms > 0:
        target_rms = 0.3
        audio_data = audio_data * (target_rms / rms)
    
    audio_data = np.where(np.abs(audio_data) > 0.95, 
                         np.sign(audio_data) * (0.95 + 0.05 * np.tanh((np.abs(audio_data) - 0.95) * 10)), 
                         audio_data)
    
    return audio_data

def enhance_audio_quality(audio_data, sample_rate=16000):
    """Apply audio enhancements for better speech recognition including loud sounds"""
    audio_data = highpass_filter(audio_data, cutoff=60, fs=sample_rate, order=3)
    audio_data = normalize_audio(audio_data)
    
    threshold = 0.005
    audio_data = np.where(np.abs(audio_data) < threshold, 
                         audio_data * 0.3, audio_data)
    
    return audio_data

def get_kotoba_generate_kwargs(task="translate", target_language="en"):
    """Get appropriate generate_kwargs for kotoba-whisper-bilingual"""
    return {
        "language": target_language,
        "task": task,
        "temperature": 0.05,
        "max_new_tokens": 224,
        "no_repeat_ngram_size": 2,
        "suppress_tokens": [-1],
    }

def get_kotoba_pipeline_kwargs():
    """Pipeline-level configuration for kotoba-whisper"""
    return {
        "chunk_length_s": 15,
        "batch_size": 16,
        "return_timestamps": True,
    }

def optimize_for_vtuber_content(generate_kwargs):
    """Apply VTuber-specific optimizations to generate_kwargs"""
    generate_kwargs["temperature"] = 0.1
    generate_kwargs["no_repeat_ngram_size"] = 1
    return generate_kwargs

def post_process_translation(text):
    """Clean up and improve translation text"""
    import re
    
    text = ' '.join(text.split())
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)
    text = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    text = re.sub(r'([.!?]){2,}', r'\1', text)
    text = re.sub(r'\bi\b', 'I', text)
    text = re.sub(r'\bim\b', "I'm", text)
    text = re.sub(r'\bdont\b', "don't", text)
    text = re.sub(r'\bcant\b', "can't", text)
    text = re.sub(r'\bwont\b', "won't", text)
    
    if len(text.split()) == 1:
        text = text.rstrip('.')
    
    return text.strip()

def processor_thread(stop_event, audio_queue):
    print("⚙️ Processor thread started.")
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {device.upper()}")

        # Ensure models are downloaded
        model_dir, vad_dir = ensure_model_downloaded()
        
        vad_model = None
        
        if not config.use_dynamic_chunking and config.use_vad_filter:
            try:
                print("📥 Loading Silero VAD model...")
                torch.set_num_threads(1)
                vad_path = os.path.join(vad_dir, "silero_vad.jit")
                vad_model = torch.jit.load(vad_path, map_location='cpu')
                print("✅ VAD model loaded successfully.")
            except Exception as e:
                print(f"⚠️ Could not load VAD model: {e}. Disabling VAD filter.")
                config.use_vad_filter = False

        print(f"📥 Loading ASR model from cache...")
        
        # Determine the appropriate dtype
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # Try to load the model directly first
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_dir,
                dtype=model_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            processor = AutoProcessor.from_pretrained(model_dir)
            
            # Create pipeline with loaded model and processor
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                dtype=model_dtype,
                device=device,
                **get_kotoba_pipeline_kwargs()
            )
            print("✅ Model loaded successfully from cache.")
        except Exception as e:
            print(f"⚠️ Failed to load model from cache: {e}")
            print("📥 Attempting to download model directly from Hugging Face...")
            
            # Fallback to downloading directly from Hugging Face
            try:
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=MODEL_ID,
                    dtype=model_dtype,
                    device=device,
                    model_kwargs={"attn_implementation": "sdpa"} if torch.cuda.is_available() else {},
                    **get_kotoba_pipeline_kwargs()
                )
                print("✅ Model loaded successfully from Hugging Face.")
            except Exception as e2:
                print(f"🔴 Failed to load model from Hugging Face: {e2}")
                raise Exception(f"Could not load model: {e}, {e2}")

        task = "translate" if config.output_mode == "translate" else "transcribe"
        target_lang = "en" if task == "translate" else config.language_code
        print(f"✅ Setting model task to: '{task}' targeting '{target_lang}' for Japanese audio.")

        generate_kwargs = get_kotoba_generate_kwargs(task, target_lang)
        generate_kwargs = optimize_for_vtuber_content(generate_kwargs)
        print("✅ ASR Model loaded successfully.")

        gui_queue.put(("model_loaded", None))

        translator = str.maketrans('', '', string.punctuation)
        last_valid_translation = ""
        translation_history = []

        while not stop_event.is_set():
            start_time = time.time()
            had_translation, was_hallucination = False, False
            try:
                audio_chunk_np = audio_queue.get(timeout=1)

                audio_chunk_np = enhance_audio_quality(audio_chunk_np.flatten(), sample_rate=SAMPLE_RATE)
                
                if not config.use_dynamic_chunking:
                    rms = np.sqrt(np.mean(audio_chunk_np ** 2))
                    if rms < config.volume_threshold:
                        stats.add_chunk(time.time() - start_time, False, False)
                        continue

                    if config.use_vad_filter and vad_model is not None:
                        audio_tensor = torch.from_numpy(audio_chunk_np.flatten()).float()
                        if len(audio_tensor) < 512:
                            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, 512 - len(audio_tensor)))
                        speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()

                        speech_windows = [prob > config.vad_threshold for prob in [speech_prob]]
                        if sum(speech_windows) < 3:
                            stats.add_chunk(time.time() - start_time, False, False)
                            continue

                audio_data = audio_chunk_np.flatten().astype(np.float32)
                min_samples = int(SAMPLE_RATE * 1.0)
                if len(audio_data) < min_samples:
                    print(f"⏩ Skipped chunk: too short ({len(audio_data)/SAMPLE_RATE:.2f}s)")
                    stats.add_chunk(time.time() - start_time, False, False)
                    continue

                result = pipe({"sampling_rate": SAMPLE_RATE, "raw": audio_data}, 
                            generate_kwargs=generate_kwargs)
                processed_text = result["text"].strip()
                
                confidence_score = 1.0
                if "chunks" in result and result["chunks"]:
                    chunk_confidences = []
                    for chunk in result["chunks"]:
                        if "timestamp" in chunk and chunk["timestamp"]:
                            timestamp = chunk["timestamp"]
                            if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                                duration = timestamp[1] - timestamp[0] if timestamp[1] else 1.0
                                chunk_conf = min(0.9, 0.6 + (duration * 0.1))
                                chunk_confidences.append(chunk_conf)
                            else:
                                chunk_confidences.append(0.8)
                        else:
                            chunk_confidences.append(0.7)
                    
                    if chunk_confidences:
                        confidence_score = sum(chunk_confidences) / len(chunk_confidences)
                        print(f"📊 Chunk analysis: {len(chunk_confidences)} chunks, avg confidence: {confidence_score:.2f}")
                else:
                    word_count = len(processed_text.split())
                    if word_count >= 3:
                        confidence_score = 0.85
                    elif word_count >= 1:
                        confidence_score = 0.75
                    else:
                        confidence_score = 0.6

                had_translation = bool(processed_text)
                is_hallucination = False
                quality_score = 1.0
                
                if had_translation:
                    lower_text = processed_text.lower()
                    
                    if any(phrase in lower_text for phrase in SUBSTRING_HALLUCINATION_FILTER):
                        is_hallucination = True
                    
                    cleaned = lower_text.translate(translator).strip()
                    if cleaned in EXACT_MATCH_HALLUCINATION_FILTER and cleaned not in PRESERVE_SOUNDS:
                        is_hallucination = True
                    
                    import re
                    
                    for pattern in QUALITY_INDICATORS["repetitive_patterns"]:
                        if re.search(pattern, processed_text, re.IGNORECASE):
                            quality_score *= 0.3
                    
                    for pattern in QUALITY_INDICATORS["nonsense_patterns"]:
                        if re.search(pattern, processed_text, re.IGNORECASE):
                            quality_score *= 0.4
                    
                    for pattern in QUALITY_INDICATORS["filler_heavy"]:
                        if re.search(pattern, processed_text, re.IGNORECASE):
                            quality_score *= 0.5
                    
                    word_count = len(processed_text.split())
                    if word_count == 1:
                        if cleaned in PRESERVE_SOUNDS or len(processed_text) <= 5:
                            quality_score *= 0.9
                        else:
                            quality_score *= 0.6
                    elif word_count > 50:
                        quality_score *= 0.7
                    
                    if quality_score < 0.4:
                        is_hallucination = True
                        print(f"👻 Low quality filtered (score: {quality_score:.2f}): '{processed_text}'")

                was_hallucination = is_hallucination
                if had_translation and not is_hallucination:
                    cleaned_text = post_process_translation(processed_text)
                    
                    if confidence_score > 0.6 and quality_score > 0.4:
                        is_contextually_valid = True
                        if translation_history:
                            if cleaned_text in translation_history[-3:]:
                                is_contextually_valid = False
                                print(f"🔄 Duplicate translation filtered: '{cleaned_text}'")
                        
                        if is_contextually_valid:
                            last_valid_translation = cleaned_text
                            translation_history.append(cleaned_text)
                            
                            if len(translation_history) > 10:
                                translation_history.pop(0)
                            
                            gui_queue.put(("subtitle", cleaned_text))
                            print(f"✅ Translation (conf: {confidence_score:.2f}, qual: {quality_score:.2f}): '{cleaned_text}'")
                    else:
                        print(f"⚠️ Low confidence translation filtered (conf: {confidence_score:.2f}, qual: {quality_score:.2f}): '{processed_text}'")
                else:
                    if had_translation:
                        print(f"👻 Hallucination filtered: '{processed_text}'")
                    
                    if last_valid_translation:
                        gui_queue.put(("subtitle", last_valid_translation))
            except Empty:
                continue
            finally:
                if 'start_time' in locals():
                    stats.add_chunk(time.time() - start_time, had_translation, was_hallucination)
                    
                # Clear GPU memory periodically
                if torch.cuda.is_available() and stats.chunks_processed % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
    except Exception as e:
        traceback.print_exc()
        print(f"🔴 Processor Thread Error: {e}")
        gui_queue.put(("error", "Model/Processing error! Check console."))
    finally:
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()
        print("⚙️ Processor thread stopped.")

class ControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Translator Control")
        self.root.geometry("450x850")
        self.root.resizable(False, False)
        self.stop_event = None
        self.worker_threads = []
        self.subtitle_window = None
        self.subtitle_label = None
        self.subtitle_shadow_label = None
        self.background_canvas = None
        self.background_rect = None
        self.subtitle_history = []
        self.last_subtitle = ""
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
        title_label = tk.Label(self.root, text="Live Audio Translator", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=5)
        info_label = tk.Label(self.root, text="First run downloads models (see console/log).", font=("Helvetica", 8), fg="grey")
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
        self.dynamic_chunk_var = tk.BooleanVar(value=config.use_dynamic_chunking)
        self.dynamic_chunk_check = tk.Checkbutton(dynamic_frame, text="Enable Dynamic Chunks (Recommended)", variable=self.dynamic_chunk_var)
        self.dynamic_chunk_check.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 5))
        tk.Label(dynamic_frame, text="Silence Timeout (s):").grid(row=1, column=0, sticky="w", pady=2)
        self.dyn_silence_var = tk.StringVar(value=str(config.dynamic_silence_timeout))
        self.dyn_silence_entry = tk.Entry(dynamic_frame, textvariable=self.dyn_silence_var, width=8)
        self.dyn_silence_entry.grid(row=1, column=1, padx=5, sticky="w")
        tk.Label(dynamic_frame, text="Max Record Time (s):").grid(row=1, column=2, sticky="w", pady=2, padx=(10,0))
        self.dyn_max_dur_var = tk.StringVar(value=str(config.dynamic_max_chunk_duration))
        self.dyn_max_dur_entry = tk.Entry(dynamic_frame, textvariable=self.dyn_max_dur_var, width=8)
        self.dyn_max_dur_entry.grid(row=1, column=3, padx=5, sticky="w")
        tk.Label(dynamic_frame, text="Min Speech Time (s):").grid(row=2, column=0, sticky="w", pady=2)
        self.dyn_min_speech_var = tk.StringVar(value=str(config.dynamic_min_speech_duration))
        self.dyn_min_speech_entry = tk.Entry(dynamic_frame, textvariable=self.dyn_min_speech_var, width=8)
        self.dyn_min_speech_entry.grid(row=2, column=1, padx=5, sticky="w")

        basic_frame = tk.LabelFrame(settings_container, text="Audio Filter Settings", padx=10, pady=10)
        basic_frame.pack(pady=5, fill="x")
        tk.Label(basic_frame, text="Volume Threshold:").grid(row=0, column=0, sticky="w", pady=2)
        self.volume_var = tk.StringVar(value=str(config.volume_threshold))
        self.volume_entry = tk.Entry(basic_frame, textvariable=self.volume_var, width=8)
        self.volume_entry.grid(row=0, column=1, padx=5, sticky="w")
        self.vad_var = tk.BooleanVar(value=config.use_vad_filter)
        self.vad_check = tk.Checkbutton(basic_frame, text="Enable VAD Filter (for both modes)", variable=self.vad_var)
        self.vad_check.grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))
        tk.Label(basic_frame, text="VAD Threshold (%):").grid(row=2, column=0, sticky="w", pady=2)
        self.vad_threshold_var = tk.StringVar(value=str(int(config.vad_threshold * 100)))
        self.vad_threshold_entry = tk.Entry(basic_frame, textvariable=self.vad_threshold_var, width=8)
        self.vad_threshold_entry.grid(row=2, column=1, padx=5, sticky="w")

        appearance_frame = tk.LabelFrame(settings_container, text="Subtitle Appearance", padx=10, pady=10)
        appearance_frame.pack(pady=5, fill="x")
        tk.Label(appearance_frame, text="Font Size:").grid(row=0, column=0, sticky="w", pady=2)
        self.font_var = tk.StringVar(value=str(config.font_size))
        self.font_entry = tk.Entry(appearance_frame, textvariable=self.font_var, width=8)
        self.font_entry.grid(row=0, column=1, padx=5, sticky="w")
        self.font_entry.bind('<KeyRelease>', self.update_subtitle_style)
        tk.Label(appearance_frame, text="Font Weight:").grid(row=0, column=2, sticky="w", pady=2, padx=(10, 0))
        self.font_weight_var = tk.StringVar(value=config.font_weight)
        self.font_weight_menu = tk.OptionMenu(appearance_frame, self.font_weight_var, 'normal', 'bold', command=self.on_font_weight_change)
        self.font_weight_menu.grid(row=0, column=3, padx=5, sticky="w")
        tk.Label(appearance_frame, text="Opacity (%):").grid(row=1, column=0, sticky="w", pady=2)
        self.opacity_var = tk.StringVar(value=str(int(config.window_opacity * 100)))
        self.opacity_entry = tk.Entry(appearance_frame, textvariable=self.opacity_var, width=8)
        self.opacity_entry.grid(row=1, column=1, padx=5, sticky="w")
        self.opacity_entry.bind('<KeyRelease>', self.on_opacity_change)
        tk.Label(appearance_frame, text="BG Mode:").grid(row=1, column=2, sticky="w", pady=2, padx=(10,0))
        self.bg_mode_var = tk.StringVar(value=config.subtitle_bg_mode)
        self.bg_mode_menu = tk.OptionMenu(appearance_frame, self.bg_mode_var, 'transparent', 'solid', command=self.set_bg_mode)
        self.bg_mode_menu.grid(row=1, column=3, padx=5, sticky="w")
        tk.Label(appearance_frame, text="BG Color:").grid(row=2, column=0, sticky="w", pady=2)
        self.bg_color_var = tk.StringVar(value=config.subtitle_bg_color)
        self.bg_color_btn = tk.Button(appearance_frame, text="Pick", command=self.pick_bg_color)
        self.bg_color_btn.grid(row=2, column=1, padx=5, sticky="w")
        self.bg_color_display = tk.Label(appearance_frame, text='  ', bg=config.subtitle_bg_color, relief="solid", borderwidth=1)
        self.bg_color_display.grid(row=2, column=2, padx=5, sticky="w")
        tk.Label(appearance_frame, text="Font Color:").grid(row=3, column=0, sticky="w", pady=2)
        self.font_color_var = tk.StringVar(value=config.subtitle_font_color)
        self.font_color_btn = tk.Button(appearance_frame, text="Pick", command=self.pick_font_color)
        self.font_color_btn.grid(row=3, column=1, padx=5, sticky="w")
        self.font_color_display = tk.Label(appearance_frame, text='  ', bg=config.subtitle_font_color, relief="solid", borderwidth=1)
        self.font_color_display.grid(row=3, column=2, padx=5, sticky="w")
        
        self.text_shadow_var = tk.BooleanVar(value=getattr(config, 'text_shadow', True))
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
            
            if config.selected_audio_device and config.selected_audio_device in device_names:
                self.device_var.set(config.selected_audio_device)
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
                msg_type, data = gui_queue.get_nowait()
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
            
            if config.text_shadow and self.subtitle_shadow_label.winfo_exists():
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
        color = colorchooser.askcolor(title="Pick Background Color", initialcolor=config.subtitle_bg_color)
        if color and color[1]:
            config.subtitle_bg_color = color[1]
            self.bg_color_display.config(bg=color[1])
            self.update_subtitle_style()

    def pick_font_color(self):
        color = colorchooser.askcolor(title="Pick Font Color", initialcolor=config.subtitle_font_color)
        if color and color[1]:
            config.subtitle_font_color = color[1]
            self.font_color_display.config(bg=color[1])
            self.update_subtitle_style()

    def apply_and_save_settings(self, save_to_disk=True):
        try:
            config.volume_threshold = max(0.0, float(self.volume_var.get()))
            config.use_vad_filter = self.vad_var.get()
            config.vad_threshold = max(0.0, min(1.0, float(self.vad_threshold_var.get()) / 100.0))
            
            config.use_dynamic_chunking = self.dynamic_chunk_var.get()
            config.dynamic_silence_timeout = max(0.1, float(self.dyn_silence_var.get()))
            config.dynamic_max_chunk_duration = max(1.0, float(self.dyn_max_dur_var.get()))
            config.dynamic_min_speech_duration = max(0.1, float(self.dyn_min_speech_var.get()))

            config.font_size = int(self.font_var.get())
            config.window_opacity = max(0.0, min(1.0, float(self.opacity_var.get()) / 100.0))
            config.font_weight = self.font_weight_var.get()
            config.text_shadow = self.text_shadow_var.get()
            config.subtitle_bg_mode = self.bg_mode_var.get()
            
            config.selected_audio_device = self.device_var.get()
            if save_to_disk:
                config.save_config()
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
        stats.reset()
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
            
        recorder = threading.Thread(target=recorder_thread, args=(self.stop_event, audio_queue, selected_device_name), daemon=True)
        processor = threading.Thread(target=processor_thread, args=(self.stop_event, audio_queue), daemon=True)
        self.worker_threads = [recorder, processor]
        for t in self.worker_threads: t.start()
        self.check_gui_queue()

    def on_opacity_change(self, event=None):
        try: config.window_opacity = max(0.0, min(1.0, float(self.opacity_var.get()) / 100.0))
        except (ValueError, tk.TclError): pass
        self.update_subtitle_style()

    def on_font_weight_change(self, value=None):
        config.font_weight = self.font_weight_var.get()
        self.update_subtitle_style()

    def on_text_shadow_change(self):
        config.text_shadow = self.text_shadow_var.get()
        self.update_subtitle_style()

    def set_bg_mode(self, value=None):
        config.subtitle_bg_mode = self.bg_mode_var.get()
        self.update_subtitle_style()

    def update_subtitle_style(self, event=None):
        if not self.subtitle_window or not self.subtitle_label.winfo_exists(): return
        try:
            font_size = int(self.font_var.get())
            font_weight = self.font_weight_var.get()
            font_tuple = ("Helvetica", font_size, font_weight)
            self.subtitle_label.config(font=font_tuple, fg=config.subtitle_font_color, bg=config.subtitle_bg_color)
            if self.subtitle_shadow_label and self.subtitle_shadow_label.winfo_exists():
                self.subtitle_shadow_label.config(font=font_tuple, fg='#1c1c1c', bg=config.subtitle_bg_color)
            if self.background_canvas and self.background_rect:
                self.background_canvas.itemconfig(self.background_rect, fill=config.subtitle_bg_color, outline=config.border_color, width=config.border_width)
            if config.subtitle_bg_mode == 'transparent':
                self.subtitle_window.wm_attributes("-alpha", config.window_opacity)
            else:
                self.subtitle_window.wm_attributes("-alpha", 1.0)
            self._update_background_size()
        except (ValueError, tk.TclError): pass

    def on_device_select(self, *args):
        config.selected_audio_device = self.device_var.get()

    def download_model(self):
        self.status_label.config(text="Status: Downloading model...", fg="blue")
        self.download_button.config(state="disabled")
        self.start_button.config(state="disabled")
        self.stop_button.config(state="disabled")
        self.root.update_idletasks()
        
        def do_download():
            try:
                print(f"Starting model download...")
                ensure_model_downloaded()
                print("Model downloads/verifications complete.")
                gui_queue.put(("status_update", ("Status: All models are ready!", "green")))
            except Exception as e:
                error_msg = f"Download failed: {e}"
                print(error_msg)
                traceback.print_exc()
                gui_queue.put(("status_update", (f"Status: {error_msg}", "red")))
            finally:
                gui_queue.put(("download_finished", None))

        def process_download_queue():
            try:
                msg_type, data = gui_queue.get_nowait()
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
            self.preset_menu['menu'].delete(0, 'end')
            self.preset_menu['menu'].add_command(label="No presets found", state="disabled")
            self.preset_var.set("No presets found")
            return

        presets = [f.replace(".json", "") for f in os.listdir(preset_dir) if f.endswith(".json")]
        menu = self.preset_menu['menu']
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
            "volume_threshold": config.volume_threshold,
            "chunk_duration": config.chunk_duration,
            "language_code": config.language_code,
            "window_opacity": config.window_opacity,
            "font_size": config.font_size,
            "use_vad_filter": config.use_vad_filter,
            "vad_threshold": config.vad_threshold,
            "subtitle_bg_color": config.subtitle_bg_color,
            "subtitle_font_color": config.subtitle_font_color,
            "subtitle_bg_mode": config.subtitle_bg_mode,
            "font_weight": config.font_weight,
            "text_shadow": config.text_shadow,
            "border_width": config.border_width,
            "border_color": config.border_color,
            "output_mode": config.output_mode,
            "use_dynamic_chunking": config.use_dynamic_chunking,
            "dynamic_max_chunk_duration": config.dynamic_max_chunk_duration,
            "dynamic_silence_timeout": config.dynamic_silence_timeout,
            "dynamic_min_speech_duration": config.dynamic_min_speech_duration
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
                setattr(config, key, value)

            self.volume_var.set(str(config.volume_threshold))
            self.opacity_var.set(str(int(config.window_opacity * 100)))
            self.font_var.set(str(config.font_size))
            self.font_weight_var.set(config.font_weight)
            self.vad_var.set(config.use_vad_filter)
            self.vad_threshold_var.set(str(int(config.vad_threshold * 100)))
            self.bg_mode_var.set(config.subtitle_bg_mode)
            self.bg_color_display.config(bg=config.subtitle_bg_color)
            self.font_color_display.config(bg=config.subtitle_font_color)
            self.text_shadow_var.set(config.text_shadow)
            
            if 'use_dynamic_chunking' in preset_data:
                self.dynamic_chunk_var.set(config.use_dynamic_chunking)
                self.dyn_silence_var.set(str(config.dynamic_silence_timeout))
                self.dyn_max_dur_var.set(str(config.dynamic_max_chunk_duration))
                self.dyn_min_speech_var.set(str(config.dynamic_min_speech_duration))
            
            self.update_subtitle_style()
            
            messagebox.showinfo("Success", f"Preset '{preset_name}' loaded.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load preset: {e}")

if __name__ == "__main__":
    if hasattr(sys, '_MEIPASS'):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    try:
        config = Config()
        stats = TranslatorStats()
        root = tk.Tk()
        app = ControlGUI(root)
        root.protocol("WM_DELETE_WINDOW", app.on_close)
        root.mainloop()
    except Exception as e:
        print("FATAL: An unhandled exception occurred during application startup.")
        traceback.print_exc()
        messagebox.showerror("Fatal Error", f"An unexpected error occurred: {e}\n\nPlease check the translator_app.log file for details.")