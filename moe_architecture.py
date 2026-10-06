"""
Enhanced Sub-Band MoE Architecture with Bin-Level Frequency Attention & Novel Mechanisms
For Codec-Based Audio Deepfake Detection.

Novel Architectural Additions:
1. Novelty 1: Cross-Frequency Residual Quantum-Diff Attention (CFRQD)
   Explicitly models the cross-correlation between low-frequency acoustic envelope bins (C1-C15)
   and high-frequency residual vector quantization (RVQ) noise bins (C45-C60).
2. Novelty 2: Elastic Sub-Band Overlap Routing (ESOR)
   Employs overlapping receptive fields across 4 expert frequency bands (Bins 0-17, 14-32, 29-47, 44-60)
   with learnable frequency gating weights to prevent quantization boundary signal loss.
3. Top-2 Sparse Gating Router with Load Balancing Loss (L_LB).
4. Bin-Level Frequency Self-Attention Module.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_LFCC = 60
NUM_EXPERTS = 4

class BinLevelFreqAttention(nn.Module):
    """
    Applies 2D self-attention along the frequency bin axis (60 LFCC bins)
    to dynamically highlight spectral quantization boundaries across time.
    """
    def __init__(self, n_lfcc=N_LFCC, hidden_dim=32):
        super().__init__()
        self.query = nn.Conv1d(n_lfcc, hidden_dim, kernel_size=1)
        self.key   = nn.Conv1d(n_lfcc, hidden_dim, kernel_size=1)
        self.value = nn.Conv1d(n_lfcc, n_lfcc, kernel_size=1)
        self.scale = 1.0 / math.sqrt(hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Conv1d(n_lfcc, n_lfcc, kernel_size=1),
            nn.BatchNorm1d(n_lfcc),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        # x shape: [B, N_LFCC, T]
        q = self.query(x).mean(dim=-1) # [B, H]
        k = self.key(x).mean(dim=-1)   # [B, H]
        v = x                           # [B, N_LFCC, T]

        attn_weights = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) * self.scale) # [B, 1]
        attn_weights = attn_weights.unsqueeze(-1) # [B, 1, 1]
        
        out = x + attn_weights * self.value(x)
        return self.out_proj(out), attn_weights.squeeze(-1)


class CFRQDAttention(nn.Module):
    """
    [NOVELTY 1] Cross-Frequency Residual Quantum-Diff Attention (CFRQD)
    Computes cross-attention between low-frequency prosody bins (C1-C15)
    and high-frequency RVQ quantization noise bins (C45-C60) to detect phase-envelope mismatches.
    """
    def __init__(self, low_bins=15, high_bins=15, embed_dim=32):
        super().__init__()
        self.proj_low  = nn.Linear(low_bins, embed_dim)
        self.proj_high = nn.Linear(high_bins, embed_dim)
        self.scale = 1.0 / math.sqrt(embed_dim)
        self.out_fc = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16)
        )

    def forward(self, x):
        # x: [B, 60, T] -> summary over time: [B, 60]
        x_summary = x.mean(dim=-1)
        low_feat  = x_summary[:, :15]   # C1 to C15 (Prosody / Low-band)
        high_feat = x_summary[:, 45:]   # C45 to C60 (RVQ Quantization / High-band)

        q = self.proj_low(low_feat)     # [B, H]
        k = self.proj_high(high_feat)   # [B, H]
        v = self.proj_high(high_feat)   # [B, H]

        # Cross-attention weights
        attn_matrix = torch.softmax((q.unsqueeze(2) @ k.unsqueeze(1)) * self.scale, dim=-1) # [B, H, H]
        cfrqd_embed = (attn_matrix @ v.unsqueeze(2)).squeeze(2) # [B, H]

        cfrqd_out = self.out_fc(cfrqd_embed)
        return cfrqd_out, attn_matrix.mean(dim=0)


class ElasticSubBandOverlap(nn.Module):
    """
    [NOVELTY 2] Elastic Sub-Band Overlap Routing (ESOR)
    Replaces rigid non-overlapping 15-bin splits with overlapping 17/18-bin receptive fields
    to ensure quantization artifacts on boundary bins are preserved and dynamically weighted.
    """
    def __init__(self, n_lfcc=N_LFCC):
        super().__init__()
        # 4 overlapping sub-band ranges:
        # Band 0 (Low Prosody): 0..17 (18 bins)
        # Band 1 (Mid Vocoder): 14..32 (18 bins)
        # Band 2 (High Quant):  29..47 (18 bins)
        # Band 3 (Ultra High):  44..60 (16 bins)
        self.slices = [
            (0, 18),
            (14, 32),
            (29, 47),
            (44, 60)
        ]
        self.gate_weights = nn.Parameter(torch.ones(4, 18))

    def get_band(self, x, expert_idx):
        start, end = self.slices[expert_idx]
        band = x[:, start:end, :] # [B, num_bins, T]
        num_bins = band.shape[1]
        
        # Apply learnable frequency weighting mask
        w = torch.sigmoid(self.gate_weights[expert_idx, :num_bins]).unsqueeze(0).unsqueeze(-1)
        return band * w


class Top2SparseMoERouter(nn.Module):
    """
    Top-2 Sparse Gating Router with Load Balancing Auxiliary Loss (L_LB).
    """
    def __init__(self, in_dim=60, num_experts=NUM_EXPERTS, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, num_experts)
        )

    def forward(self, global_summary):
        # global_summary: [B, in_dim]
        logits = self.gate_net(global_summary) # [B, num_experts]
        probs  = F.softmax(logits, dim=-1)     # [B, num_experts]

        # Top-k sparse routing
        topk_probs, topk_indices = torch.topk(probs, k=self.top_k, dim=-1) # [B, k]
        topk_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Dense routing mask for expert fusion
        routing_weights = torch.zeros_like(probs)
        routing_weights.scatter_(1, topk_indices, topk_weights)

        # Compute Load-Balancing Auxiliary Loss (L_LB)
        # f_i: fraction of samples routed to expert i in top-1
        top1_indices = topk_indices[:, 0]
        one_hot = F.one_hot(top1_indices, num_classes=self.num_experts).float()
        fraction_routed = one_hot.mean(dim=0)
        avg_probability = probs.mean(dim=0)
        
        load_balancing_loss = self.num_experts * torch.sum(fraction_routed * avg_probability)

        return routing_weights, load_balancing_loss, probs


class SubBandExpert(nn.Module):
    """
    1D-CNN Specialist Expert module processing specialized sub-band spectral cues.
    """
    def __init__(self, in_channels=18, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv1d(64, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1) # [B, out_dim]


class AdvancedAudioMoEClassifier(nn.Module):
    """
    Full Audio Anti-Spoofing Architecture combining:
    - Bin-Level Frequency Attention
    - Elastic Sub-Band Overlap Routing (ESOR)
    - Cross-Frequency Residual Quantum-Diff Attention (CFRQD)
    - Top-2 Sparse Gating MoE Router
    """
    def __init__(self, n_lfcc=N_LFCC, num_experts=NUM_EXPERTS, top_k=2):
        super().__init__()
        self.n_lfcc = n_lfcc
        self.num_experts = num_experts

        # Bin-level frequency self-attention
        self.freq_attn = BinLevelFreqAttention(n_lfcc=n_lfcc)

        # Novelty 1: CFRQD Attention
        self.cfrqd_attn = CFRQDAttention(low_bins=15, high_bins=15)

        # Novelty 2: Elastic Sub-Band Overlap Routing
        self.esor = ElasticSubBandOverlap(n_lfcc=n_lfcc)

        # 4 Specialized Sub-Band Experts (18 input channels for overlapping bands)
        self.experts = nn.ModuleList([
            SubBandExpert(in_channels=18 if i < 3 else 16, out_dim=64) for i in range(num_experts)
        ])

        # Top-2 Sparse Router
        self.router = Top2SparseMoERouter(in_dim=n_lfcc, num_experts=num_experts, top_k=top_k)

        # Dense Classifier Fusion Layer
        # Fused expert feature (64) + CFRQD feature (16) = 80
        self.classifier = nn.Sequential(
            nn.Linear(64 + 16, 48),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.35),
            nn.Linear(48, 2)
        )

    def forward(self, x):
        # x: [B, 60, T]
        # 1. Apply Bin-Level Frequency Attention
        x_attn, freq_weights = self.freq_attn(x)

        # 2. Extract CFRQD Cross-Frequency Feature (Novelty 1)
        cfrqd_feat, cfrqd_matrix = self.cfrqd_attn(x_attn)

        # 3. Global summary for Top-2 Router
        global_summary = x_attn.mean(dim=2) # [B, 60]
        routing_weights, lb_loss, gate_probs = self.router(global_summary)

        # 4. Expert Processing with ESOR Overlapping Bands (Novelty 2)
        expert_outputs = []
        for i in range(self.num_experts):
            sub_band = self.esor.get_band(x_attn, i)
            expert_outputs.append(self.experts[i](sub_band))

        # 5. Top-2 Sparse Fusion
        fused_expert = (torch.stack(expert_outputs, dim=1) * routing_weights.unsqueeze(-1)).sum(dim=1)

        # 6. Combine Expert Fusion + CFRQD Feature
        combined_feat = torch.cat([fused_expert, cfrqd_feat], dim=-1)
        logits = self.classifier(combined_feat)

        return logits, {
            "routing_weights": routing_weights,
            "gate_probs": gate_probs,
            "lb_loss": lb_loss,
            "freq_attn_weights": freq_weights,
            "cfrqd_matrix": cfrqd_matrix
        }
