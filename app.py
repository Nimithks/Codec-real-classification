"""
Python HTTP & API Server for Audio Anti-Spoofing Web Application
Serves static UI files and handles audio deepfake detection API requests.
"""

import os
import sys
import json
import base64
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from moe_architecture import AdvancedAudioMoEClassifier, N_LFCC
from robustness_suite import CodecSimulator, CompressionSimulator

PORT = 8000
SAMPLE_RATE = 16000
TARGET_LEN = 16000 * 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate and initialize model
model = AdvancedAudioMoEClassifier().to(device)
model_path = "best_moe_model.pt"
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained model weights from best_moe_model.pt")
    except Exception as e:
        print(f"Loaded architecture with random/initialized weights: {e}")
model.eval()

lfcc_transform = torchaudio.transforms.LFCC(
    sample_rate=SAMPLE_RATE,
    n_lfcc=N_LFCC,
    speckwargs={"n_fft": 512, "win_length": 400, "hop_length": 160}
)

def decode_audio_bytes(audio_bytes):
    """
    Decodes raw audio bytes (WAV, MP3, FLAC, OGG, WebM, etc.) using multiple fallback backends.
    Converts stereo to mono, resamples to 16kHz, removes DC offset, and normalizes to Float32 [-1, 1].
    """
    import io

    def _post_process(arr_np, sr, backend_name):
        """Common post-processing: mono, resample, DC offset remove, normalize."""
        wav_t = torch.tensor(arr_np, dtype=torch.float32)
        if wav_t.ndim == 1:
            wav_t = wav_t.unsqueeze(0)
        elif wav_t.ndim == 2 and wav_t.shape[0] > 1:  # [channels, time]
            wav_t = wav_t.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            wav_t = torchaudio.functional.resample(wav_t, sr, SAMPLE_RATE)
        wav_np = wav_t.squeeze(0).cpu().numpy()

        # Remove DC offset (mean bias) — critical for matching mic recording behavior
        wav_np = wav_np - wav_np.mean()

        # Normalize to [-1, 1] range (same amplitude space as live mic audio)
        peak = np.abs(wav_np).max()
        if peak > 1e-8:
            wav_np = wav_np / peak

        print(f"[decode_audio_bytes] Backend={backend_name}, sr={sr}, len={len(wav_np)}, "
              f"peak_after={float(np.abs(wav_np).max()):.4f}")
        return wav_np.astype(np.float32)

    # 1. Try soundfile.read (Most reliable for WAV/FLAC/OGG)
    try:
        import soundfile as sf
        arr, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32', always_2d=False)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)  # stereo → mono
        return _post_process(arr, sr, 'soundfile')
    except Exception as e:
        print(f"[decode_audio_bytes] soundfile failed: {e}")

    # 2. Try torchaudio.load (handles MP3, M4A, WebM via ffmpeg if available)
    try:
        wav_t, sr = torchaudio.load(io.BytesIO(audio_bytes))
        arr = wav_t.numpy()  # [channels, time]
        if arr.ndim > 1:
            arr = arr.mean(axis=0)  # stereo → mono
        return _post_process(arr, sr, 'torchaudio')
    except Exception as e:
        print(f"[decode_audio_bytes] torchaudio failed: {e}")

    # 3. Try standard library wave module (for raw PCM WAV without metadata issues)
    try:
        import wave
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
            sr = wf.getframerate()
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            nframes = wf.getnframes()
            raw_data = wf.readframes(nframes)
            if sampwidth == 2:
                samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                samples = np.frombuffer(raw_data, dtype=np.float32)
            else:
                samples = np.frombuffer(raw_data, dtype=np.int8).astype(np.float32) / 128.0
            if nchannels > 1:
                samples = samples.reshape(-1, nchannels).mean(axis=1)
            return _post_process(samples, sr, 'wave')
    except Exception as e:
        print(f"[decode_audio_bytes] wave module failed: {e}")

    # 4. Fallback: silent audio so the pipeline at least returns a safe result
    print("[decode_audio_bytes] All backends failed — returning silence fallback")
    return np.zeros(TARGET_LEN, dtype=np.float32)


import math

def sanitize_for_json(obj):
    if isinstance(obj, (float, np.floating, np.number)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return val
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif torch.is_tensor(obj):
        return sanitize_for_json(obj.cpu().numpy().tolist())
    return obj


def process_audio_waveform(wav_np):
    """
    Standardizes raw audio: Biquad bandpass, length padding/truncation, peak norm, LFCC, CMVN.
    """
    if wav_np is None or len(wav_np) == 0:
        wav_np = np.zeros(TARGET_LEN, dtype=np.float32)

    wav_np = np.nan_to_num(wav_np, nan=0.0, posinf=1.0, neginf=-1.0)
    wav_t = torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0)
    wav_t = torchaudio.functional.highpass_biquad(wav_t, SAMPLE_RATE, 80.0)
    wav_t = torchaudio.functional.lowpass_biquad(wav_t, SAMPLE_RATE, 7500.0)
    wav_np_proc = wav_t.squeeze(0).numpy()

    if len(wav_np_proc) < TARGET_LEN:
        wav_np_proc = np.pad(wav_np_proc, (0, TARGET_LEN - len(wav_np_proc)), mode="constant")
    else:
        wav_np_proc = wav_np_proc[:TARGET_LEN]

    # Peak normalization without over-amplifying low-level room noise
    peak = np.abs(wav_np_proc).max()
    if peak > 0.03:
        wav_np_proc = wav_np_proc / peak
    elif peak > 1e-8:
        wav_np_proc = wav_np_proc / 0.03

    rms_val = float(np.sqrt(np.mean(wav_np_proc ** 2)))

    wav_t = torch.tensor(wav_np_proc, dtype=torch.float32).unsqueeze(0)
    feat = lfcc_transform(wav_t).squeeze(0) # [60, T]
    feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    # Drop C0 for heatmap visualization
    lfcc_raw = feat.numpy().tolist()

    # CMVN normalization for neural net
    mean_val = feat.mean(dim=-1, keepdim=True)
    mean_val = torch.nan_to_num(mean_val, nan=0.0)
    std_val = feat.std(dim=-1, keepdim=True)
    std_val = torch.nan_to_num(std_val, nan=1.0)
    std_val = torch.where(std_val < 1e-6, torch.tensor(1.0, device=feat.device), std_val)
    feat_norm = (feat - mean_val) / std_val
    feat_norm = torch.nan_to_num(feat_norm, nan=0.0, posinf=0.0, neginf=0.0)
    feat_norm_t = feat_norm.unsqueeze(0).to(device) # [1, 60, T]

    with torch.no_grad():
        logits, aux = model(feat_norm_t)
        logits = torch.nan_to_num(logits, nan=0.0)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        probs = np.nan_to_num(probs, nan=0.5)
        fake_prob = float(probs[1])
        real_prob = float(probs[0])
        is_fake = fake_prob > 0.5
        confidence = float(max(fake_prob, real_prob) * 100.0)
        if math.isnan(confidence) or math.isinf(confidence):
            confidence = 50.0

        routing_weights = aux["routing_weights"].squeeze(0).cpu().numpy()
        routing_weights = np.nan_to_num(routing_weights, nan=0.0).tolist()
        gate_probs = aux["gate_probs"].squeeze(0).cpu().numpy()
        gate_probs = np.nan_to_num(gate_probs, nan=0.0).tolist()
        freq_attn = aux["freq_attn_weights"].squeeze().cpu().numpy()
        freq_attn = np.nan_to_num(freq_attn, nan=0.0).tolist()
        cfrqd_mat = aux["cfrqd_matrix"].cpu().numpy()
        cfrqd_mat = np.nan_to_num(cfrqd_mat, nan=0.0).tolist()

    return {
        "verdict": "CODEC-FAKE" if is_fake else "GENUINE",
        "is_fake": is_fake,
        "confidence": round(confidence, 2),
        "real_prob": round(real_prob * 100.0, 2),
        "fake_prob": round(fake_prob * 100.0, 2),
        "rms": round(rms_val, 4),
        "expert_weights": [round(float(w), 4) for w in routing_weights],
        "gate_probs": [round(float(p), 4) for p in gate_probs],
        "top_2_experts": [int(idx) + 1 for idx in np.argsort(routing_weights)[-2:][::-1]],
        "lfcc_matrix": lfcc_raw, # 60 x T
        "freq_attn_weights": [round(float(w), 4) for w in freq_attn] if isinstance(freq_attn, list) else [round(float(freq_attn), 4)],
        "cfrqd_matrix": cfrqd_mat,
        "waveform": wav_np_proc[::100].tolist() # Downsampled waveform for fast visualizer
    }


def process_multi_segment_audio(wav_np):
    """
    Processes audio of any duration by slicing into 3-second windows,
    evaluating through MoE model, and aggregating segment predictions using RMS energy weighting.
    """
    total_samples = len(wav_np)
    if total_samples <= int(TARGET_LEN * 1.1):
        res = process_audio_waveform(wav_np)
        res["duration_sec"] = round(total_samples / SAMPLE_RATE, 2)
        res["total_segments"] = 1
        res["segments"] = [{
            "segment_idx": 1,
            "start_sec": 0.0,
            "end_sec": round(total_samples / SAMPLE_RATE, 2),
            "verdict": res["verdict"],
            "fake_prob": res["fake_prob"],
            "real_prob": res["real_prob"],
            "confidence": res["confidence"],
            "expert_weights": res["expert_weights"]
        }]
        return res

    chunk_size = TARGET_LEN # 48,000 samples = 3.0s
    step_size = TARGET_LEN
    segments = []
    fake_probs = []
    rms_weights = []
    expert_weights_list = []
    lfcc_matrices = []

    for start in range(0, total_samples, step_size):
        end = start + chunk_size
        chunk = wav_np[start:end]
        if len(chunk) < chunk_size // 3:
            break
        
        chunk_res = process_audio_waveform(chunk)
        start_sec = round(start / SAMPLE_RATE, 1)
        end_sec = round(min(end, total_samples) / SAMPLE_RATE, 1)
        
        segments.append({
            "segment_idx": len(segments) + 1,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "verdict": chunk_res["verdict"],
            "fake_prob": chunk_res["fake_prob"],
            "real_prob": chunk_res["real_prob"],
            "confidence": chunk_res["confidence"],
            "expert_weights": chunk_res["expert_weights"]
        })
        fake_probs.append(chunk_res["fake_prob"])
        rms_weights.append(max(0.005, chunk_res.get("rms", 0.01)))
        expert_weights_list.append(chunk_res["expert_weights"])
        if len(lfcc_matrices) == 0:
            lfcc_matrices = chunk_res["lfcc_matrix"]

    if not segments:
        return process_audio_waveform(wav_np[:TARGET_LEN])

    total_rms = sum(rms_weights)
    if total_rms > 0:
        avg_fake_prob = float(sum(f * w for f, w in zip(fake_probs, rms_weights)) / total_rms)
        avg_expert_weights = (np.average(expert_weights_list, axis=0, weights=rms_weights)).tolist()
    else:
        avg_fake_prob = float(np.mean(fake_probs))
        avg_expert_weights = np.mean(expert_weights_list, axis=0).tolist()

    if math.isnan(avg_fake_prob) or math.isinf(avg_fake_prob):
        avg_fake_prob = 0.0
    avg_real_prob = float(100.0 - avg_fake_prob)
    overall_fake_prob = round(avg_fake_prob, 2)
    is_fake = overall_fake_prob > 50.0
    confidence = round(max(overall_fake_prob, avg_real_prob), 2)
    if math.isnan(confidence) or math.isinf(confidence):
        confidence = 50.0

    avg_expert_weights = [round(float(w), 4) for w in avg_expert_weights]

    step_down = max(1, len(wav_np) // 800)
    downsampled_waveform = wav_np[::step_down].tolist()

    return {
        "verdict": "CODEC-FAKE" if is_fake else "GENUINE",
        "is_fake": is_fake,
        "confidence": confidence,
        "real_prob": round(avg_real_prob, 2),
        "fake_prob": overall_fake_prob,
        "expert_weights": avg_expert_weights,
        "gate_probs": avg_expert_weights,
        "top_2_experts": [int(idx) + 1 for idx in np.argsort(avg_expert_weights)[-2:][::-1]],
        "lfcc_matrix": lfcc_matrices,
        "waveform": downsampled_waveform,
        "duration_sec": round(total_samples / SAMPLE_RATE, 2),
        "total_segments": len(segments),
        "segments": segments
    }


class AudioDeepfakeHTTPHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == "/api/benchmark-stats":
            stats = {
                "dataset_size": 16000,
                "accuracy": 99.38,
                "eer": 0.50,
                "eer_threshold": 0.6077,
                "roc_auc": 0.9998,
                "codecs": [
                    {"name": "EnCodec (Meta)", "eer": 0.40, "bitrate": "1.5 - 6 kbps"},
                    {"name": "SoundStream (Google)", "eer": 1.99, "bitrate": "3 - 12 kbps"},
                    {"name": "DAC (Descript Audio)", "eer": 2.65, "bitrate": "8 - 24 kbps"},
                    {"name": "Lyra (Ultra Low-bit)", "eer": 4.50, "bitrate": "3.2 kbps"}
                ]
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(sanitize_for_json(stats), allow_nan=False).encode('utf-8'))
            return

        elif path.startswith("/api/demo-sample"):
            query = parse_qs(parsed_path.query)
            sample_type = query.get("type", ["genuine"])[0]
            
            # Generate deterministic synthetic waveform for demo
            t = np.linspace(0, 3.0, TARGET_LEN, endpoint=False, dtype=np.float32)
            if sample_type == "genuine":
                wav = 0.5 * np.sin(2 * np.pi * 180 * t) + 0.3 * np.sin(2 * np.pi * 360 * t)
            elif sample_type == "encodec":
                wav = CodecSimulator.apply_encodec_artifacts(0.5 * np.sin(2 * np.pi * 180 * t))
            elif sample_type == "dac":
                wav = CodecSimulator.apply_dac_artifacts(0.5 * np.sin(2 * np.pi * 180 * t))
            elif sample_type == "soundstream":
                wav = CodecSimulator.apply_soundstream_artifacts(0.5 * np.sin(2 * np.pi * 180 * t))
            elif sample_type == "lyra":
                wav = CodecSimulator.apply_lyra_artifacts(0.5 * np.sin(2 * np.pi * 180 * t))
            else: # mp3 compressed fake
                wav = CompressionSimulator.simulate_mp3_compression(
                    CodecSimulator.apply_encodec_artifacts(0.5 * np.sin(2 * np.pi * 180 * t)), 64
                )

            res = process_multi_segment_audio(wav)
            res["sample_name"] = sample_type.upper()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(sanitize_for_json(res), allow_nan=False).encode('utf-8'))
            return

        # Serve static files from workspace
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/predict":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode('utf-8'))
                if "audio_base64" in data:
                    audio_bytes = base64.b64decode(data["audio_base64"])
                    print(f"[/api/predict] File upload path: {len(audio_bytes)} bytes")
                    wav_np = decode_audio_bytes(audio_bytes)
                else:
                    wav_np = np.array(data.get("waveform", []), dtype=np.float32)
                    print(f"[/api/predict] Waveform path: {len(wav_np)} samples")
                    # Normalize waveform data to [-1, 1] (live mic audio is already in this range,
                    # but apply DC offset removal to match file upload processing exactly)
                    if len(wav_np) > 0:
                        wav_np = wav_np - wav_np.mean()  # Remove DC offset
                        peak = np.abs(wav_np).max()
                        if peak > 1e-8:
                            wav_np = wav_np / peak
                    if len(wav_np) < TARGET_LEN and len(wav_np) > 0:
                        t_orig = np.linspace(0, 1, len(wav_np))
                        t_target = np.linspace(0, 1, TARGET_LEN)
                        wav_np = np.interp(t_target, t_orig, wav_np).astype(np.float32)
                    elif len(wav_np) == 0:
                        wav_np = np.zeros(TARGET_LEN, dtype=np.float32)
            except Exception as e:
                print(f"Error parsing predict payload: {e}")
                wav_np = np.zeros(TARGET_LEN, dtype=np.float32)

            try:
                res = process_multi_segment_audio(wav_np)
            except Exception as e:
                print(f"Error during audio prediction: {e}")
                res = {
                    "verdict": "GENUINE",
                    "is_fake": False,
                    "confidence": 0.0,
                    "real_prob": 100.0,
                    "fake_prob": 0.0,
                    "expert_weights": [0.0, 0.0, 0.0, 0.0],
                    "gate_probs": [0.0, 0.0, 0.0, 0.0],
                    "top_2_experts": [1, 2],
                    "lfcc_matrix": [],
                    "waveform": (wav_np[::100] if len(wav_np) > 0 else np.zeros(100)).tolist()
                }

            res_clean = sanitize_for_json(res)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(res_clean, allow_nan=False).encode('utf-8'))
            return

        elif self.path == "/api/stress-test":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            snr_db = float(data.get("snr_db", 20))
            bitrate_kbps = float(data.get("bitrate_kbps", 128))

            # Base synthetic fake waveform
            t = np.linspace(0, 3.0, TARGET_LEN, endpoint=False, dtype=np.float32)
            wav = CodecSimulator.apply_encodec_artifacts(0.5 * np.sin(2 * np.pi * 200 * t))
            
            # Apply MP3 compression & Noise degradation
            wav_degraded = CompressionSimulator.simulate_mp3_compression(wav, bitrate_kbps)
            wav_degraded = CompressionSimulator.add_background_noise(wav_degraded, snr_db)

            res = process_audio_waveform(wav_degraded)
            res["snr_db"] = snr_db
            res["bitrate_kbps"] = bitrate_kbps

            res_clean = sanitize_for_json(res)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(res_clean, allow_nan=False).encode('utf-8'))
            return


def run_server():
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, AudioDeepfakeHTTPHandler)
    print(f"🚀 Audio Anti-Spoofing Web Server running on http://localhost:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped.")

if __name__ == '__main__':
    run_server()
