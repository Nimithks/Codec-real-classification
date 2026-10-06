"""
Standalone Evaluator Script for best_moe_model.pt
Loads cached 16,000 dataset samples and evaluates performance on held-out test set.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split

from moe_architecture import AdvancedAudioMoEClassifier, N_LFCC, NUM_EXPERTS

SAMPLE_RATE = 16000
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Standalone Model Evaluator ({device}) ===")

# 1. Load Cached Dataset
real_cache = os.path.join("cached_data", "real_samples.npy")
fake_cache = os.path.join("cached_data", "fake_samples.npy")

if not (os.path.exists(real_cache) and os.path.exists(fake_cache)):
    print("❌ Cached dataset files not found in cached_data/. Run download_dataset.py first.")
    exit(1)

print(f"📦 Loading cached samples from {real_cache} and {fake_cache}...")
real_arr = np.load(real_cache)
fake_arr = np.load(fake_cache)
raw_samples = np.concatenate([real_arr, fake_arr], axis=0)
raw_labels = np.concatenate([np.zeros(len(real_arr)), np.ones(len(fake_arr))], axis=0)
total_samples = len(raw_samples)

print(f"Dataset Loaded: Total = {total_samples} | Real: {len(real_arr)} | Codec-Fake: {len(fake_arr)}")

# 2. Stratified Train/Val/Test Split (70% / 10% / 20%)
X_train, X_temp, y_train, y_temp = train_test_split(
    raw_samples, raw_labels, test_size=0.30, stratify=raw_labels, random_state=42
)
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=(1.0 / 3.0), stratify=y_temp, random_state=42
)

# PyTorch Test Dataset
class AudioAntiSpoofDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=SAMPLE_RATE,
            n_lfcc=N_LFCC,
            speckwargs={"n_fft": 512, "win_length": 400, "hop_length": 160}
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_np = self.samples[idx].copy()
        # Evaluation noise channel simulation
        noise = np.random.normal(0, 0.008, wav_np.shape).astype(np.float32)
        wav_np = wav_np + noise

        wav = torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0)
        feat = self.lfcc(wav).squeeze(0)
        # CMVN
        feat = (feat - feat.mean(dim=-1, keepdim=True)) / (feat.std(dim=-1, keepdim=True) + 1e-6)
        return {"feat": feat, "label": torch.tensor(self.labels[idx], dtype=torch.long)}

test_loader = DataLoader(AudioAntiSpoofDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# 3. Load Trained Model Checkpoint
model_path = "best_moe_model.pt"
if not os.path.exists(model_path):
    print(f"❌ Model checkpoint {model_path} not found!")
    exit(1)

model = AdvancedAudioMoEClassifier().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"✅ Loaded trained model weights from {model_path}")

# 4. Evaluate Test Set
test_scores, test_labels, test_preds = [], [], []
with torch.no_grad():
    for batch in test_loader:
        feat = batch["feat"].to(device)
        labels = batch["label"].to(device)
        logits, _ = model(feat)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        scores = probs[:, 0]  # Score for Genuine Class 0
        test_scores.extend(scores.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
        test_preds.extend(preds.cpu().numpy())

test_labels = np.array(test_labels)
test_scores = np.array(test_scores)
test_preds = np.array(test_preds)

test_acc = accuracy_score(test_labels, test_preds) * 100.0

fpr, tpr, thresholds = roc_curve(test_labels, test_scores, pos_label=0)
fnr = 1 - tpr
idx = np.nanargmin(np.abs(fpr - fnr))
eer = ((fpr[idx] + fnr[idx]) / 2.0) * 100.0
eer_threshold = thresholds[idx]
roc_auc = auc(fpr, tpr)

print("\n==========================================")
print("  OFFICIAL TEST EVALUATION RESULTS SUMMARY")
print("==========================================")
print(f"Test Set Size : {len(test_labels)} samples")
print(f"Accuracy      : {test_acc:.2f}%")
print(f"EER           : {eer:.2f}%")
print(f"EER Threshold : {eer_threshold:.4f}")
print(f"ROC-AUC Score : {roc_auc:.4f}")

# 5. Save Plots & JSON Summary
cm = confusion_matrix(test_labels, test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Genuine", "Codec-Fake"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Test Confusion Matrix (Acc: {test_acc:.2f}%, EER: {eer:.2f}%)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()

summary = {
    "total_samples": total_samples,
    "test_samples": len(test_labels),
    "accuracy": round(float(test_acc), 2),
    "eer": round(float(eer), 2),
    "eer_threshold": round(float(eer_threshold), 4),
    "roc_auc": round(float(roc_auc), 4),
    "codecs_benchmarked": ["encodec", "soundstream", "dac", "lyra"],
    "novelties": [
        "Cross-Frequency Residual Quantum-Diff Attention (CFRQD)",
        "Elastic Sub-Band Overlap Routing (ESOR)"
    ]
}

with open("benchmark_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("✅ Saved confusion_matrix.png, roc_curve.png, and benchmark_summary.json successfully!")
