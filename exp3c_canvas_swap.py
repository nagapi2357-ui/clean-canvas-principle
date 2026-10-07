#!/usr/bin/env python3
"""
Experiment 3c: Canvas-Swap Matrix
===================================
Protocol: EXP3C_PROTOCOL.md v1.0
Question: Does the inference-time canvas benefit depend on the training canvas?
          Is there canvas-specific fossilisation (train X → need infer X)?

Design: 3 training canvases × 8 inference canvases × 5 seeds × 3 rotations
        = 360 evaluations

Training canvases:
  WHITE        - clean canvas (best in 3a), weights reused from 3b corrected
  CROSS-EXHAUST - worst training canvas (3a), worst OOD — can inference rescue it?
  SAME-EXHAUST  - in-task exhaust — tests matched-canvas dependency

Inference canvases (all 8 from 3b corrected):
  NO-NOISE, WHITE, PINK, CROSS-EXHAUST, SAME-EXHAUST,
  RANDOM-LABEL-EXHAUST, SHUFFLED-EXHAUST, DEAD-EXHAUST

Diagonal (extra control): train X + infer X
  - Train CROSS + infer CROSS
  - Train SAME  + infer SAME
  - Train WHITE + infer WHITE (already in 3b, but replicated here)

Pre-registered kill criteria:
  Primary: For CROSS-trained models, is NO-NOISE ≥ all structured inference canvases?
  H1 support: CROSS infer canvas beats NO-NOISE for at least one non-WHITE training arm
  Diagonal: does infer-X outperform infer-NO-NOISE for train-X models?
  Interaction (exploratory): does best inference canvas differ across training arms?

Authors: Nagaπ + Adrian (Tusk Innovations)
Date: 2026-10-07
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
EXP3A_DIR    = '/content/drive/MyDrive/Exp3a Exhaust Canvas'   # B-weights
EXP3B_DIR    = '/content/drive/MyDrive/Exp3b Corrected'        # WHITE clean weights
SAVE_DIR     = '/content/drive/MyDrive/Exp3c Canvas Swap'
WEIGHTS_DIR  = os.path.join(SAVE_DIR, 'train_weights')
RESULTS_DIR  = os.path.join(SAVE_DIR, 'results')
B_SAVE_DIR   = os.path.join(EXP3A_DIR, 'B_weights')

for d in [SAVE_DIR, WEIGHTS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)
print(f"Saving to: {SAVE_DIR}")

# ── Config ──
CFG = {
    'protocol': 'exp3c-v1.0',
    'epochs': 50,
    'batch_size': 128,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'noise_std': 0.05,
    'seeds': [42, 137, 2357, 1729, 8191],
    'ood_rotations': [30, 45, 60],
    'num_workers': 2,
    'b_seed_offsets': {'cifar': 1000, 'fmnist': 2000, 'random': 3000, 'dead': 4000},
    'train_canvases': ['WHITE', 'CROSS-EXHAUST', 'SAME-EXHAUST'],
    'inference_canvases': [
        'NO-NOISE', 'WHITE', 'PINK',
        'CROSS-EXHAUST', 'SAME-EXHAUST', 'RANDOM-LABEL-EXHAUST',
        'SHUFFLED-EXHAUST', 'DEAD-EXHAUST',
    ],
    # Diagonal: these train+infer pairs are the extra control
    'diagonal_pairs': [
        ('WHITE', 'WHITE'),
        ('CROSS-EXHAUST', 'CROSS-EXHAUST'),
        ('SAME-EXHAUST', 'SAME-EXHAUST'),
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

total_evals = len(CFG['train_canvases']) * len(CFG['inference_canvases']) * len(CFG['seeds']) * len(CFG['ood_rotations'])
print(f"Config: {len(CFG['train_canvases'])} train × {len(CFG['inference_canvases'])} infer × "
      f"{len(CFG['seeds'])} seeds × {len(CFG['ood_rotations'])} angles = {total_evals} evaluations")


# ══════════════════════════════════════════════════════════════════════
# MODEL (identical to 3a/3b)
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
# DATASETS (identical to 3a/3b — no augmentation)
# ══════════════════════════════════════════════════════════════════════

class NormalizedDataset(Dataset):
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
    """Infinite re-shuffling loader. Identical to 3a."""
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
# NOISE GENERATION (identical to 3a/3b)
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
    """images_B must come from B's OWN training data (CyclingLoader)."""
    with torch.no_grad():
        raw = B_model.get_layer2_activations(images_B)
        spatial = F.interpolate(raw, size=(32,32), mode='bilinear', align_corners=False)
        projected = projection(spatial)
        if shuffle_pixels: projected = pixel_shuffle_per_sample(projected)
        return normalise_noise(projected, noise_std)

print("Noise generation defined")


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def evaluate(model, loader):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += model(x).argmax(1).eq(y).sum().item()
            total += x.size(0)
    return correct / total

def freeze_model(model):
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model

print("Helpers defined")


# ══════════════════════════════════════════════════════════════════════
# TRAINING — one function for all three training canvases
# ══════════════════════════════════════════════════════════════════════

def train_model(train_canvas, seed, cifar_train_ds, cifar_test_loader,
                b_models, projection, b_datasets):
    """
    Train SmallResNet A under the specified training canvas.
    Identical hyperparameters for all three arms.
    Returns trained model (not frozen).
    """
    set_seed(seed)
    A = SmallResNet().to(DEVICE)
    optimizer = optim.AdamW(A.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'])
    criterion = nn.CrossEntropyLoss()

    # A's own data loader — no augmentation, matches 3a/3b exactly
    g_a = torch.Generator(); g_a.manual_seed(seed)
    a_loader = DataLoader(cifar_train_ds, batch_size=CFG['batch_size'], shuffle=True,
                          num_workers=CFG['num_workers'], drop_last=True, generator=g_a)

    # B loader for exhaust arms (independent of A's data)
    B_model, b_cycling = None, None
    is_exhaust = train_canvas not in ('WHITE', 'PINK')
    is_shuffled = (train_canvas == 'SHUFFLED-EXHAUST')

    if is_exhaust:
        b_key = TRAIN_B_KEY[train_canvas] + f'_seed{seed}'
        B_model = b_models[b_key]
        b_cycling = CyclingLoader(b_datasets[train_canvas], CFG['batch_size'],
                                   seed * 31 + 0, num_workers=CFG['num_workers'])

    for epoch in range(CFG['epochs']):
        A.train()
        for x_a, y_a in a_loader:
            x_a, y_a = x_a.to(DEVICE), y_a.to(DEVICE)

            if train_canvas == 'WHITE':
                noise = generate_white_noise(x_a.shape, CFG['noise_std'])
            elif train_canvas == 'PINK':
                noise = generate_pink_noise_2d(x_a.shape, CFG['noise_std'])
            else:
                x_b, _ = next(b_cycling)
                x_b = x_b.to(DEVICE)
                noise = get_exhaust_noise(B_model, projection, x_b, CFG['noise_std'],
                                          shuffle_pixels=is_shuffled)

            optimizer.zero_grad()
            loss = criterion(A(x_a + noise), y_a)
            loss.backward(); optimizer.step()

        scheduler.step()
        if epoch % 10 == 0 or epoch == CFG['epochs'] - 1:
            test_acc = evaluate(A, cifar_test_loader)
            print(f"    [{train_canvas}|seed {seed}] ep {epoch:>2} | test {test_acc:.4f}")

    return A


# B-model key mapping (which B to use for each training canvas)
TRAIN_B_KEY = {
    'CROSS-EXHAUST':        'B_fmnist',
    'SAME-EXHAUST':         'B_cifar',
    'RANDOM-LABEL-EXHAUST': 'B_random',
    'SHUFFLED-EXHAUST':     'B_cifar',
    'DEAD-EXHAUST':         'C_dead',
}

# Inference B-model mapping (same as 3b)
INFER_B_KEY = {
    'CROSS-EXHAUST':        'B_fmnist',
    'SAME-EXHAUST':         'B_cifar',
    'RANDOM-LABEL-EXHAUST': 'B_random',
    'SHUFFLED-EXHAUST':     'B_cifar',
    'DEAD-EXHAUST':         'C_dead',
}

print("Training function defined")


# ══════════════════════════════════════════════════════════════════════
# INFERENCE EVALUATION (identical to 3b corrected)
# ══════════════════════════════════════════════════════════════════════

def evaluate_with_inference_canvas(model, ood_loader, canvas_name,
                                    B_model, projection, b_cycling, noise_std):
    """
    Evaluate model on rotated OOD images with specified inference canvas.
    B_model processes its OWN training data (b_cycling), not test images.
    NO-NOISE adds nothing — pure OOD baseline.
    """
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in ood_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            if canvas_name == 'NO-NOISE':
                x_input = x
            elif canvas_name == 'WHITE':
                x_input = x + generate_white_noise(x.shape, noise_std)
            elif canvas_name == 'PINK':
                x_input = x + generate_pink_noise_2d(x.shape, noise_std)
            else:
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
# PHASE 0 — Load B-models (reused from 3a)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  PHASE 0: LOADING B-MODELS (reused from 3a)")
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
    for b_type, b_test_loader in [
        ('B_cifar',  cifar_test_loader),
        ('B_fmnist', fmnist_test_loader),
        ('B_random', cifar_test_loader),
        ('C_dead',   None),
    ]:
        path = os.path.join(B_SAVE_DIR, f'{b_type}_seed{seed}.pt')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"B-model not found: {path}\n"
                f"Copy B_weights/ from Exp3a Google Drive folder."
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
            print(f"    {b_type}: loaded")

with open(os.path.join(SAVE_DIR, 'b_accuracies.json'), 'w') as f:
    json.dump(b_accuracies, f, indent=2)

# Positive control check
bc_vals = [b_accuracies.get(f'B_cifar_seed{s}',0) for s in CFG['seeds']]
bf_vals = [b_accuracies.get(f'B_fmnist_seed{s}',0) for s in CFG['seeds']]
bc_mean, bf_mean = np.mean(bc_vals), np.mean(bf_vals)
print(f"\n  Positive controls: B_cifar={bc_mean:.4f} (need≥0.80) {'✅' if bc_mean>=0.80 else '❌'}")
print(f"                     B_fmnist={bf_mean:.4f} (need≥0.85) {'✅' if bf_mean>=0.85 else '❌'}")
print("\nPhase 0 complete")


# ══════════════════════════════════════════════════════════════════════
# PHASE 1 — Train / Load all three sets of models
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  PHASE 1: TRAINING — 3 CANVAS TYPES × 5 SEEDS")
print("  WHITE: reused from 3b. CROSS & SAME: trained fresh.")
print("="*70)

projection = make_frozen_projection()

# B-datasets for training (used by CyclingLoader during training)
train_b_datasets = {
    'CROSS-EXHAUST': fmnist_train_ds,
    'SAME-EXHAUST':  cifar_train_ds,
}

# Inference B-datasets (used by CyclingLoader during evaluation)
infer_b_datasets = {
    'CROSS-EXHAUST':        fmnist_train_ds,
    'SAME-EXHAUST':         cifar_train_ds,
    'RANDOM-LABEL-EXHAUST': cifar_train_ds,
    'SHUFFLED-EXHAUST':     cifar_train_ds,
    'DEAD-EXHAUST':         cifar_train_ds,
}

train_models = {tc: {} for tc in CFG['train_canvases']}
train_test_accs = {}

for train_canvas in CFG['train_canvases']:
    print(f"\n{'='*60}\n  TRAIN CANVAS: {train_canvas}\n{'='*60}")

    for seed in CFG['seeds']:
        weight_path = os.path.join(WEIGHTS_DIR, f'{train_canvas.lower().replace("-","_")}_seed{seed}.pt')

        if train_canvas == 'WHITE':
            # Load from 3b corrected (same training code, same seed)
            alt_path = os.path.join(EXP3B_DIR, 'clean_weights', f'white_seed{seed}.pt')
            if os.path.exists(weight_path):
                load_path = weight_path
            elif os.path.exists(alt_path):
                load_path = alt_path
                print(f"  Seed {seed}: loading WHITE from 3b corrected")
            else:
                load_path = None
        else:
            load_path = weight_path if os.path.exists(weight_path) else None

        if load_path:
            print(f"  Seed {seed}: loading cached weights")
            m = SmallResNet().to(DEVICE)
            m.load_state_dict(torch.load(load_path, map_location=DEVICE))
        else:
            print(f"  Seed {seed}: training from scratch...")
            t0 = time.time()
            m = train_model(train_canvas, seed, cifar_train_ds, cifar_test_loader,
                             b_models, projection, train_b_datasets)
            print(f"  Seed {seed}: trained in {(time.time()-t0)/60:.1f} min")
            torch.save(m.state_dict(), weight_path)
            print(f"  Saved → {weight_path}")

        test_acc = evaluate(m, cifar_test_loader)
        print(f"  Seed {seed} [{train_canvas}]: test acc = {test_acc:.4f}")
        train_test_accs[f'{train_canvas}_seed{seed}'] = test_acc
        train_models[train_canvas][seed] = freeze_model(m)

# Sanity check: in-distribution accuracy should be ~0.83 for all arms
print("\n  ── Test Accuracy Sanity Check ──")
for tc in CFG['train_canvases']:
    vals = [train_test_accs[f'{tc}_seed{s}'] for s in CFG['seeds']]
    print(f"  {tc}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  (expect ~0.83 for all)")

print("\nPhase 1 complete")


# ══════════════════════════════════════════════════════════════════════
# PHASE 2 — Canvas-swap evaluation (3 × 8 × 5 × 3 = 360 evaluations)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  PHASE 2: CANVAS-SWAP EVALUATION")
print(f"  {len(CFG['train_canvases'])} × {len(CFG['inference_canvases'])} × "
      f"{len(CFG['seeds'])} × {len(CFG['ood_rotations'])} = {total_evals} evaluations")
print("="*70)

# OOD loaders (rotated, noise applied in evaluate_with_inference_canvas)
ood_loaders = {}
for rot in CFG['ood_rotations']:
    ood_ds = RotatedDataset(cifar_test_raw, rot)
    ood_loaders[rot] = DataLoader(ood_ds, batch_size=CFG['batch_size'],
                                   shuffle=False, num_workers=CFG['num_workers'])

results = []
t_phase2 = time.time()

for train_canvas in CFG['train_canvases']:
    print(f"\n{'='*60}\n  TRAIN CANVAS: {train_canvas}\n{'='*60}")

    for seed_idx, seed in enumerate(CFG['seeds']):
        model = train_models[train_canvas][seed]
        print(f"  Seed {seed} ({seed_idx+1}/{len(CFG['seeds'])})")

        for infer_idx, infer_canvas in enumerate(CFG['inference_canvases']):

            # Set up B-model and cycling loader for exhaust inference arms
            B_model, b_cycling = None, None
            if infer_canvas in INFER_B_KEY:
                b_key = f"{INFER_B_KEY[infer_canvas]}_seed{seed}"
                B_model = b_models[b_key]
                # Unique seed per train×infer×seed combination for reproducibility
                b_initial_seed = seed * 17 + infer_idx
                b_cycling = CyclingLoader(infer_b_datasets[infer_canvas], CFG['batch_size'],
                                           b_initial_seed, num_workers=CFG['num_workers'])

            # Flag diagonal conditions
            is_diagonal = (train_canvas, infer_canvas) in CFG['diagonal_pairs']

            for rot in CFG['ood_rotations']:
                acc = evaluate_with_inference_canvas(
                    model, ood_loaders[rot], infer_canvas,
                    B_model, projection, b_cycling,
                    CFG['noise_std']
                )
                results.append({
                    'train_canvas': train_canvas,
                    'inference_canvas': infer_canvas,
                    'seed': seed,
                    'rotation_deg': rot,
                    'ood_acc': round(acc, 6),
                    'is_diagonal': is_diagonal,
                    'timestamp': datetime.now().isoformat()
                })

            # Print per-canvas result at 45°
            r45 = next(r for r in reversed(results) if r['rotation_deg']==45)
            diag_tag = " ◆DIAGONAL" if is_diagonal else ""
            print(f"    {infer_canvas}: {r45['ood_acc']:.4f}{diag_tag}")

phase2_time = time.time() - t_phase2
print(f"\nPhase 2 complete in {phase2_time/60:.1f} min")


# ══════════════════════════════════════════════════════════════════════
# RESULTS + ANALYSIS
# ══════════════════════════════════════════════════════════════════════

df = pd.DataFrame(results)
out_path = os.path.join(RESULTS_DIR, 'exp3c_results.csv')
df.to_csv(out_path, index=False)
print(f"\nResults saved → {out_path}")

# ── Heatmap: 3×8 mean OOD at 45° ──
print("\n=== HEATMAP: Mean OOD at 45° ===")
df45 = df[df['rotation_deg']==45]
heatmap = df45.groupby(['train_canvas','inference_canvas'])['ood_acc'].mean().unstack()
print(heatmap.round(4).to_string())

# ── Delta vs NO-NOISE (per training arm) ──
print("\n=== Δ vs NO-NOISE (per training arm) ===")
for tc in CFG['train_canvases']:
    sub = df45[df45['train_canvas']==tc]
    no_noise = sub[sub['inference_canvas']=='NO-NOISE']['ood_acc'].mean()
    print(f"\n  Train: {tc} | NO-NOISE baseline: {no_noise:.4f}")
    for ic in CFG['inference_canvases']:
        if ic == 'NO-NOISE': continue
        val = sub[sub['inference_canvas']==ic]['ood_acc'].mean()
        delta = val - no_noise
        diag = " ◆" if (tc, ic) in CFG['diagonal_pairs'] else "  "
        print(f"   {diag} {ic:<28} {val:.4f}  Δ={delta:+.4f}")

# ── Diagonal analysis ──
print("\n=== DIAGONAL TEST (train X + infer X vs NO-NOISE) ===")
for tc, ic in CFG['diagonal_pairs']:
    sub = df45[df45['train_canvas']==tc]
    no_noise_vals = sub[sub['inference_canvas']=='NO-NOISE']['ood_acc'].values
    diag_vals     = sub[sub['inference_canvas']==ic]['ood_acc'].values
    if len(diag_vals) == 0: continue
    diffs = diag_vals - no_noise_vals
    print(f"  Train={tc} + Infer={ic}:")
    print(f"    NO-NOISE: {no_noise_vals.mean():.4f}  |  Diagonal: {diag_vals.mean():.4f}  |  Δ={diffs.mean():+.4f}")
    print(f"    Per seed: {[round(d,4) for d in diffs]}")

# ── Interaction test: does best inference canvas differ across training arms? ──
print("\n=== INTERACTION: Best inference canvas per training arm ===")
for tc in CFG['train_canvases']:
    sub = df45[df45['train_canvas']==tc]
    ranked = sub.groupby('inference_canvas')['ood_acc'].mean().sort_values(ascending=False)
    best_ic = ranked.index[0]
    best_val = ranked.iloc[0]
    no_noise_val = sub[sub['inference_canvas']=='NO-NOISE']['ood_acc'].mean()
    print(f"  Train {tc}: best inference = {best_ic} ({best_val:.4f}, Δ={best_val-no_noise_val:+.4f})")

print("\n✅ Exp 3c complete")
print(f"   Total time: {(time.time()-t_phase2+0)/60:.1f} min (Phase 2 only)")
