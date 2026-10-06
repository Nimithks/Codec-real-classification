"""
Strict Dataset Downloader for Codec-Based Audio Deepfake Detection
Downloads 100% authentic benchmark datasets directly from HuggingFace Hub:
- Genuine Class: openslr/librispeech_asr (Real Human Speech)
- Codec-Fake Class: rogertseng/CodecFake (Authentic Neural Codec Deepfakes)

"""

import os
import sys
import io
import ssl
import numpy as np
import torch
import torchaudio

# Bypass SSL certificate verification for local proxies/firewalls
ssl._create_default_https_context = ssl._create_unverified_context

import soundfile as sf
from datasets import load_dataset, Audio

DATA_DIR = "cached_data"
SAMPLE_RATE = 16000
TARGET_LEN = 16000 * 3  # 3.0 seconds

def process_raw_audio_array(arr, sr):
    wav_t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    if sr != SAMPLE_RATE:
        wav_t = torchaudio.functional.resample(wav_t, sr, SAMPLE_RATE)

    # Acoustic Standardization: Biquad Bandpass Filter (80 Hz - 7.5 kHz)
    wav_t = torchaudio.functional.highpass_biquad(wav_t, SAMPLE_RATE, 80.0)
    wav_t = torchaudio.functional.lowpass_biquad(wav_t, SAMPLE_RATE, 7500.0)
    wav_np = wav_t.squeeze(0).numpy()

    if len(wav_np) < TARGET_LEN:
        wav_np = np.pad(wav_np, (0, TARGET_LEN - len(wav_np)), mode="constant")
    else:
        wav_np = wav_np[:TARGET_LEN]

    peak = np.abs(wav_np).max()
    if peak > 1e-8:
        wav_np = wav_np / peak

    return wav_np.astype(np.float32)

def download_and_cache_dataset(num_samples_per_class=8000):
    """
    Strictly downloads 16,000 real dataset samples from HuggingFace Hub:
    - 8,000 Real Speech samples from LibriSpeech ASR
    - 8,000 Codec-Fake Deepfake samples from CodecFake benchmark
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    real_cache_path = os.path.join(DATA_DIR, "real_samples.npy")
    fake_cache_path = os.path.join(DATA_DIR, "fake_samples.npy")

    print(f"=== Strict Download: Caching {num_samples_per_class * 2} Real Dataset Samples ===")

    # 1. Genuine Speech from LibriSpeech ASR
    if not os.path.exists(real_cache_path):
        print(f"\n[1/2] Streaming Genuine Speech from openslr/librispeech_asr...")
        ds_real = load_dataset("openslr/librispeech_asr", "clean", split="train.100", streaming=True)
        # Disable torchcodec auto-decoding to fix Windows DLL bug
        ds_real = ds_real.cast_column("audio", Audio(decode=False))
        
        real_samples = []
        count = 0
        for item in ds_real:
            if count >= num_samples_per_class:
                break
            audio_obj = item["audio"]
            if "bytes" in audio_obj and audio_obj["bytes"]:
                arr, sr = sf.read(io.BytesIO(audio_obj["bytes"]))
            else:
                arr, sr = np.array(audio_obj["array"], dtype=np.float32), int(audio_obj["sampling_rate"])

            wav_proc = process_raw_audio_array(arr, sr)
            real_samples.append(wav_proc)
            count += 1
            if count % 1000 == 0:
                print(f"  Downloaded & Processed {count}/{num_samples_per_class} Real samples...")

        np.save(real_cache_path, np.array(real_samples))
        print(f"✅ Successfully cached {len(real_samples)} Genuine samples to {real_cache_path}")
    else:
        print(f"✅ Found existing Real samples cache at {real_cache_path}")

    # 2. CodecFake Deepfake Audio
    if not os.path.exists(fake_cache_path):
        print(f"\n[2/2] Streaming Authentic Deepfakes from rogertseng/CodecFake...")
        ds_fake = load_dataset("rogertseng/CodecFake", split="train", streaming=True)
        ds_fake = ds_fake.cast_column("audio", Audio(decode=False))
        
        fake_samples = []
        count = 0
        for item in ds_fake:
            if count >= num_samples_per_class:
                break
            audio_obj = item["audio"]
            if "bytes" in audio_obj and audio_obj["bytes"]:
                arr, sr = sf.read(io.BytesIO(audio_obj["bytes"]))
            else:
                arr, sr = np.array(audio_obj["array"], dtype=np.float32), int(audio_obj["sampling_rate"])

            wav_proc = process_raw_audio_array(arr, sr)
            fake_samples.append(wav_proc)
            count += 1
            if count % 1000 == 0:
                print(f"  Downloaded & Processed {count}/{num_samples_per_class} CodecFake samples...")

        np.save(fake_cache_path, np.array(fake_samples))
        print(f"✅ Successfully cached {len(fake_samples)} CodecFake samples to {fake_cache_path}")
    else:
        print(f"✅ Found existing CodecFake samples cache at {fake_cache_path}")

    print("\n🎉 Strict dataset downloading and caching completed successfully!")

if __name__ == '__main__':
    download_and_cache_dataset()
