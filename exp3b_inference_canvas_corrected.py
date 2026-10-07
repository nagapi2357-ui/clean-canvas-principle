#!/usr/bin/env python3
"""
Experiment 3b (corrected v1.1): Inference-Time Canvas — Stochastic Resonance Test
===================================================================================
Protocol: EXP3B_PROTOCOL.md v1.1 (corrected)
Question: Does structured noise at inference time improve OOD generalisation
          when the model has been trained on a clean (white noise) canvas?

FIXES vs Grok's v1.0:
  1. Training augmentation removed — matches 3a exactly (NormalizedDataset only)
  2. Exhaust source fixed — B_model processes its OWN training data via CyclingLoader,
     NOT the rotated test images. Exhaust is now independent of test input.
  3. NO-NOISE baseline arm added — enables direct apples-to-apples comparison.

Authors: Nagaπ + Adrian (Tusk Innovations)
Date: 2026-10-06 (corrected)
"""

from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json, time, os, random, math
from datetime import datetime
import pandas as pd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Paths ──
# Reuse 3a B-weights (same seeds = same models). Train new clean WHITE weights.
EXP3A_DIR  = '/content/drive/MyDrive/Exp3a Exhaust Canvas'   # reuse B_weights from here
SAVE_DIR   = '/content/drive/MyDrive/Exp3b Corrected'
WEIGHTS_DIR = os.path.join(SAVE_DIR, 'clean_weights')
B_SAVE_DIR  = os.path.join(EXP3A_DIR, 'B_weights')           # reuse 3a B-weights
RESULTS_DIR = os.path.join(SAVE_DIR, 'results')
for d in [SAVE_DIR, WEIGHTS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)
print(f"Saving to: {SAVE_DIR}")
print(f"Loading B-weights from: {B_SAVE_DIR}")

# ── Config ──
CFG = {
    'protocol': 'exp3b-v1.1-corrected',
    'epochs': 50,
    'batch_size': 128,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'noise_std': 0.05,
    'seeds': [42, 137, 2357, 1729, 8191],
    'ood_rotations': [30, 45, 60],
    'num_workers': 2,
    'b_seed_offsets': {'cifar': 1000, 'fmnist': 2000, 'random': 3000, 'dead': 4000},
    # 8 inference canvases: NO-NOISE is new baseline
    'inference_canvases': [
        'NO-NOISE',               # FIX 3: no noise added — anchors the comparison
        'WHITE',                  # unstructured baseline
        'PINK',                   # spectral structure (1/f)
        'CROSS-EXHAUST',          # cross-task semantic exhaust
        'SAME-EXHAUST',           # in-task semantic exhaust
        'RANDOM-LABEL-EXHAUST',   # uninformative but structured
        'SHUFFLED-EXHAUST',       # structure without spatial semantics
        'DEAD-EXHAUST',           # structured but informationally empty
    ],
}

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

with open(os.path.join(SAVE_DIR, 'config.json'), 'w') as f:
    json.dump(CFG, f, indent=2)
print("Config saved.")


# ══════════════════════════════════════════════════════════════════════
# MODEL (identical to Exp 3a)
# ══════════════════════════════════════════════════════════════════════

class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.block1 = self._make_block(32, 64)
        self.block2 = self._make_block(64, 128)
        self.block3 = self._make_block(128, 256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def _make_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(),
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        return self.fc(self.pool(x).flatten(1))

    def get_layer2_activations(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.block1(x); x = self.block2(x)
        return x

print(f"SmallResNet: {sum(p.numel() for p in SmallResNet().parameters()):,} params")


# ══════════════════════════════════════════════════════════════════════
# DATASETS (identical to Exp 3a — no augmentation)
# ══════════════════════════════════════════════════════════════════════

class NormalizedDataset(Dataset):
    """FIX 1: ToTensor + Normalize only. No RandomCrop or RandomHorizontalFlip."""
    def __init__(self, base_dataset, mean=CIFAR_MEAN, std=CIFAR_STD):
        self.base = base_dataset
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.transform(img), label

class ShuffledLabelDataset(Dataset):
    def __init__(self, base_dataset, seed):
        self.base = base_dataset
        self.transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
        rng = np.random.RandomState(seed)
        all_labels = [base_dataset[i][1] for i in range(len(base_dataset))]
        self.shuffled_labels = [all_labels[rng.permutation(len(base_dataset))[i]]
                                 for i in range(len(base_dataset))]
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return self.transform(img), self.shuffled_labels[idx]

class FashionMNISTDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.transform = T.Compose([
            T.Resize((32, 32)), T.ToTensor(),
            lambda x: x.repeat(3, 1, 1), T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.transform(img), label

class RotatedDataset(Dataset):
    """OOD: rotated CIFAR-10 test images. No noise added here."""
    def __init__(self, base_dataset, angle):
        self.base = base_dataset
        self.angle = angle
        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, label = self.base[idx]
        img = T.functional.rotate(T.functional.to_pil_image(T.ToTensor()(img)), self.angle)
        return self.normalize(T.ToTensor()(img)), label

class CyclingLoader:
    """Infinite iterator over a dataset with re-shuffling. Identical to Exp 3a."""
    def __init__(self, dataset, batch_size, initial_seed, num_workers=2):
        self.dataset = dataset; self.batch_size = batch_size
        self.num_workers = num_workers; self.rng = random.Random(initial_seed)
        self._make_loader()
    def _make_loader(self):
        seed = self.rng.randint(0, 2**31); g = torch.Generator(); g.manual_seed(seed)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                 num_workers=self.num_workers, drop_last=True, generator=g,
                                 persistent_workers=False)
        self.iter = iter(self.loader)
    def __next__(self):
        try: return next(self.iter)
        except StopIteration: self._make_loader(); return next(self.iter)

print("Dataset classes defined")


# ══════════════════════════════════════════════════════════════════════
# NOISE GENERATION (identical to Exp 3a)
# ══════════════════════════════════════════════════════════════════════

def make_frozen_projection():
    torch.manual_seed(0)
    proj = nn.Conv2d(128, 3, kernel_size=1, bias=False)
    proj.requires_grad_(False); proj.eval()
    return proj.to(DEVICE)

def normalise_noise(noise, noise_std):
    mean = noise.mean(dim=[1,2,3], keepdim=True)
    std  = noise.std(dim=[1,2,3],  keepdim=True) + 1e-8
    return (noise - mean) / std * noise_std

def generate_white_noise(shape, noise_std):
    return normalise_noise(torch.randn(shape, device=DEVICE), noise_std)

def generate_pink_noise_2d(shape, noise_std):
    B, C, H, W = shape
    fy = torch.fft.fftfreq(H, device=DEVICE).unsqueeze(1)
    fx = torch.fft.fftfreq(W, device=DEVICE).unsqueeze(0)
    freqs = fy**2 + fx**2; freqs[0,0] = 1.0
    amplitude = 1.0 / torch.sqrt(freqs)
    phase = torch.rand(B, C, H, W, device=DEVICE) * 2 * math.pi
    noise = torch.fft.ifft2(amplitude.unsqueeze(0).unsqueeze(0) * torch.exp(1j*phase)).real
    return normalise_noise(noise, noise_std)

def pixel_shuffle_per_sample(tensor):
    B, C, H, W = tensor.shape; result = tensor.clone()
    for i in range(B):
        perm = torch.randperm(H*W, device=tensor.device)
        result[i] = result[i].view(C,-1)[:,perm].view(C,H,W)
    return result

def get_exhaust_noise(B_model, projection, images_B, noise_std, shuffle_pixels=False):
    """FIX 2: images_B must come from B's OWN training data (CyclingLoader),
    NOT from the test images being evaluated."""
    with torch.no_grad():
        raw = B_model.get_layer2_activations(images_B)
        spatial = F.interpolate(raw, size=(32,32), mode='bilinear', align_corners=False)
        projected = projection(spatial)
        if shuffle_pixels: projected = pixel_shuffle_per_sample(projected)
        return normalise_noise(projected, noise_std)

print("Noise generation defined")


# ══════════════════════════════════════════════════════════════════════
# TRAINING (identical to Exp 3a WHITE arm)
# ══════════════════════════════════════════════════════════════════════

def evaluate(model, loader):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += model(x).argmax(1).eq(y).sum().item()
            total += x.size(0)
    return correct / total

def train_white_model(seed, cifar_train_ds, cifar_test_loader):
    """FIX 1: No augmentation. Exact replica of 3a WHITE arm training."""
    set_seed(seed)
    A = SmallResNet().to(DEVICE)
    optimizer = optim.AdamW(A.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'])
    criterion = nn.CrossEntropyLoss()

    g_a = torch.Generator(); g_a.manual_seed(seed)
    a_loader = DataLoader(cifar_train_ds, batch_size=CFG['batch_size'], shuffle=True,
                          num_workers=CFG['num_workers'], drop_last=True, generator=g_a)

    for epoch in range(CFG['epochs']):
        A.train()
        for x_a, y_a in a_loader:
            x_a, y_a = x_a.to(DEVICE), y_a.to(DEVICE)
            noise = generate_white_noise(x_a.shape, CFG['noise_std'])
            optimizer.zero_grad()
            loss = criterion(A(x_a + noise), y_a)
            loss.backward(); optimizer.step()
        scheduler.step()
        if epoch % 10 == 0 or epoch == CFG['epochs'] - 1:
            test_acc = evaluate(A, cifar_test_loader)
            print(f"  ep {epoch:>2} | test {test_acc:.4f}")

    weight_path = os.path.join(WEIGHTS_DIR, f'white_seed{seed}.pt')
    torch.save(A.state_dict(), weight_path)
    print(f"  Saved → {weight_path}")
    return A

def freeze_model(model):
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model

print("Training functions defined")


# ══════════════════════════════════════════════════════════════════════
# INFERENCE EVALUATION (key fix: exhaust from b_cycling, not test images)
# ══════════════════════════════════════════════════════════════════════

def evaluate_with_inference_canvas(model, ood_loader, canvas_name,
                                    B_model, projection, b_cycling, noise_std):
    """
    Evaluates model on rotated OOD images with specified inference canvas.

    FIX 2: For exhaust arms, noise is generated from B_model processing
    images from b_cycling (B's own training data), never from the test images.

    FIX 3: canvas_name='NO-NOISE' adds nothing — provides the clean baseline.
    """
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in ood_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # x is already rotated by RotatedDataset

            if canvas_name == 'NO-NOISE':
                x_input = x  # FIX 3: no noise — pure OOD baseline

            elif canvas_name == 'WHITE':
                x_input = x + generate_white_noise(x.shape, noise_std)

            elif canvas_name == 'PINK':
                x_input = x + generate_pink_noise_2d(x.shape, noise_std)

            else:
                # FIX 2: exhaust from B's OWN training data, independent of test input
                x_b, _ = next(b_cycling)
                x_b = x_b[:x.size(0)].to(DEVICE)  # match last-batch size
                is_shuffled = (canvas_name == 'SHUFFLED-EXHAUST')
                noise = get_exhaust_noise(B_model, projection, x_b, noise_std,
                                          shuffle_pixels=is_shuffled)
                x_input = x + noise

            correct += model(x_input).argmax(1).eq(y).sum().item()
            total += x.size(0)

    return correct / total

print("Inference evaluation defined")


# ══════════════════════════════════════════════════════════════════════
# PHASE 0 — Load B-models (reusing Exp 3a weights — same seeds, same models)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  PHASE 0: LOADING B-MODELS (reused from Exp 3a)")
print("="*70)

cifar_train_raw  = torchvision.datasets.CIFAR10('./data', train=True,  download=True)
cifar_test_raw   = torchvision.datasets.CIFAR10('./data', train=False, download=True)
fmnist_train_raw = torchvision.datasets.FashionMNIST('./data', train=True,  download=True)
fmnist_test_raw  = torchvision.datasets.FashionMNIST('./data', train=False, download=True)

cifar_train_ds  = NormalizedDataset(cifar_train_raw)
cifar_test_ds   = NormalizedDataset(cifar_test_raw)
fmnist_train_ds = FashionMNISTDataset(fmnist_train_raw)
fmnist_test_ds  = FashionMNISTDataset(fmnist_test_raw)

cifar_test_loader  = DataLoader(cifar_test_ds,  batch_size=CFG['batch_size'],
                                 shuffle=False, num_workers=CFG['num_workers'])
fmnist_test_loader = DataLoader(fmnist_test_ds, batch_size=CFG['batch_size'],
                                 shuffle=False, num_workers=CFG['num_workers'])

b_models = {}
b_accuracies = {}

for seed in CFG['seeds']:
    print(f"\n  ── Seed {seed} ──")
    for b_type, b_dataset, b_test_loader in [
        ('B_cifar',  cifar_train_ds,  cifar_test_loader),
        ('B_fmnist', fmnist_train_ds, fmnist_test_loader),
        ('B_random', None,            cifar_test_loader),
        ('C_dead',   None,            None),
    ]:
        path = os.path.join(B_SAVE_DIR, f'{b_type}_seed{seed}.pt')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"B-model not found: {path}\n"
                f"Run Exp 3a first (Phase 0), or copy B_weights/ from Exp3a Google Drive folder."
            )
        m = SmallResNet().to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m = freeze_model(m)
        b_models[f'{b_type}_seed{seed}'] = m
        if b_test_loader is not None:
            acc = evaluate(m, b_test_loader)
            b_accuracies[f'{b_type}_seed{seed}'] = acc
            print(f"    {b_type}: {acc:.4f}")
        else:
            print(f"    {b_type}: loaded (no eval)")

print("\nPhase 0 complete")


# ══════════════════════════════════════════════════════════════════════
# PHASE 1 — Train / Load clean WHITE models (one per seed)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  PHASE 1: CLEAN WHITE TRAINING (one model per seed)")
print("="*70)

clean_models = {}
for seed in CFG['seeds']:
    weight_path = os.path.join(WEIGHTS_DIR, f'white_seed{seed}.pt')
    if os.path.exists(weight_path):
        print(f"  Seed {seed}: loading cached weights")
        m = SmallResNet().to(DEVICE)
        m.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    else:
        print(f"  Seed {seed}: training from scratch (50 epochs)...")
        m = train_white_model(seed, cifar_train_ds, cifar_test_loader)
    clean_models[seed] = freeze_model(m)
    test_acc = evaluate(m, cifar_test_loader)
    print(f"  Seed {seed}: test acc = {test_acc:.4f}")

print("\nPhase 1 complete")


# ══════════════════════════════════════════════════════════════════════
# PHASE 2 — Inference canvas evaluation (5 seeds × 8 canvases × 3 angles)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  PHASE 2: INFERENCE CANVAS EVALUATION")
print(f"  {len(CFG['seeds'])} seeds × {len(CFG['inference_canvases'])} canvases × "
      f"{len(CFG['ood_rotations'])} angles = "
      f"{len(CFG['seeds']) * len(CFG['inference_canvases']) * len(CFG['ood_rotations'])} evaluations")
print("="*70)

projection = make_frozen_projection()

# OOD loaders (rotated, no noise — noise is applied in evaluate_with_inference_canvas)
ood_loaders = {}
for rot in CFG['ood_rotations']:
    ood_ds = RotatedDataset(cifar_test_raw, rot)
    ood_loaders[rot] = DataLoader(ood_ds, batch_size=CFG['batch_size'],
                                   shuffle=False, num_workers=CFG['num_workers'])

# B-dataset map for cycling loaders (same as 3a ARM_B_DATASET)
B_DATASET_MAP = {
    'CROSS-EXHAUST':        fmnist_train_ds,
    'SAME-EXHAUST':         cifar_train_ds,
    'RANDOM-LABEL-EXHAUST': cifar_train_ds,
    'SHUFFLED-EXHAUST':     cifar_train_ds,
    'DEAD-EXHAUST':         cifar_train_ds,
}
B_MODEL_KEY = {
    'CROSS-EXHAUST':        'B_fmnist',
    'SAME-EXHAUST':         'B_cifar',
    'RANDOM-LABEL-EXHAUST': 'B_random',
    'SHUFFLED-EXHAUST':     'B_cifar',
    'DEAD-EXHAUST':         'C_dead',
}

results = []
t_phase2 = time.time()

for seed_idx, seed in enumerate(CFG['seeds']):
    print(f"\n{'='*60}\n  SEED {seed} ({seed_idx+1}/{len(CFG['seeds'])})\n{'='*60}")
    model = clean_models[seed]

    for canvas_idx, canvas in enumerate(CFG['inference_canvases']):

        # Set up B-model and cycling loader for exhaust arms
        B_model, b_cycling = None, None
        if canvas in B_DATASET_MAP:
            b_key = f"{B_MODEL_KEY[canvas]}_seed{seed}"
            B_model = b_models[b_key]
            b_initial_seed = seed * 13 + canvas_idx   # reproducible, distinct per arm
            b_cycling = CyclingLoader(B_DATASET_MAP[canvas], CFG['batch_size'],
                                       b_initial_seed, num_workers=CFG['num_workers'])

        print(f"  Canvas: {canvas}")
        for rot in CFG['ood_rotations']:
            acc = evaluate_with_inference_canvas(
                model, ood_loaders[rot], canvas,
                B_model, projection, b_cycling,
                CFG['noise_std']
            )
            results.append({
                'seed': seed,
                'inference_canvas': canvas,
                'rotation_deg': rot,
                'ood_acc': round(acc, 6),
                'timestamp': datetime.now().isoformat()
            })
            print(f"    {rot}° → {acc:.4f}")

phase2_time = time.time() - t_phase2
print(f"\nPhase 2 complete in {phase2_time/60:.1f} min")


# ══════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════

df = pd.DataFrame(results)
out_path = os.path.join(RESULTS_DIR, 'exp3b_corrected_results.csv')
df.to_csv(out_path, index=False)
print(f"\nResults saved → {out_path}")

# Summary at 45°
print("\n=== Summary (OOD acc at 45°, 5-seed mean) ===")
summary_45 = (df[df['rotation_deg']==45]
              .groupby('inference_canvas')['ood_acc']
              .agg(['mean','std'])
              .sort_values('mean', ascending=False))
# Reference: NO-NOISE baseline
no_noise_45 = summary_45.loc['NO-NOISE','mean'] if 'NO-NOISE' in summary_45.index else None
if no_noise_45:
    summary_45['Δ_vs_NO-NOISE'] = (summary_45['mean'] - no_noise_45).round(4)
    summary_45['Δ_vs_WHITE']    = (summary_45['mean'] -
                                    summary_45.loc['WHITE','mean']).round(4)
print(summary_45.round(4).to_string())

# Full rotation breakdown
print("\n=== Full rotation breakdown ===")
print(df.groupby(['inference_canvas','rotation_deg'])['ood_acc'].mean().unstack().round(4).to_string())

print("\n✅ Exp 3b (corrected) complete")
