#!/usr/bin/env python3
"""
Experiment 3a: Exhaust Canvas — Coupled Network Noise
======================================================
Protocol: EXP3A_PROTOCOL_FROZEN.md v1.2
Question: Does structured noise from another system's processing (exhaust)
          produce better OOD generalisation than unstructured white noise?

Colab-ready: single file, A100 GPU.
Authors: Nagaπ + Adrian (Tusk Innovations)
Date: 2026-10-02
"""

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 1: Mount Drive + Setup                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import json, time, os, random, math, copy
from datetime import datetime

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

SAVE_DIR = '/content/drive/MyDrive/Exp3a Exhaust Canvas'
B_SAVE_DIR = os.path.join(SAVE_DIR, 'B_weights')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(B_SAVE_DIR, exist_ok=True)
print(f"Results will save to: {SAVE_DIR}")

# ── Config ──
CFG = {
    'protocol': 'exp3a-v1.2-frozen',
    'epochs': 50,
    'batch_size': 128,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'noise_std': 0.05,
    'seeds': [42, 137, 2357, 1729, 8191],
    'backup_seed': 31337,
    'ood_rotations': [30, 45, 60],
    'num_workers': 2,
    'b_seed_offsets': {
        'cifar': 1000,
        'fmnist': 2000,
        'random': 3000,
        'dead': 4000,
    },
    'arms': ['WHITE', 'PINK', 'CROSS-EXHAUST', 'SAME-EXHAUST',
             'RANDOM-LABEL-EXHAUST', 'SHUFFLED-EXHAUST', 'DEAD-EXHAUST'],
}

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

with open(os.path.join(SAVE_DIR, 'config.json'), 'w') as f:
    json.dump(CFG, f, indent=2)
print(f"Config saved. Phase 1: {len(CFG['arms'])} arms × {len(CFG['seeds'])} seeds = {len(CFG['arms']) * len(CFG['seeds'])} runs")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 2: Model                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

class SmallResNet(nn.Module):
    """3-block ResNet for CIFAR-10. Identical to Exp 2."""
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
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def get_layer2_activations(self, x):
        """Return layer 2 (block2) output: [B, 128, H', W']"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        return x

print(f"✅ SmallResNet defined ({sum(p.numel() for p in SmallResNet().parameters()):,} params)")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 3: Datasets                                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

class NormalizedDataset(Dataset):
    """Applies ToTensor + Normalize. For CIFAR-10."""
    def __init__(self, base_dataset, mean=CIFAR_MEAN, std=CIFAR_STD):
        self.base = base_dataset
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.transform(img), label


class ShuffledLabelDataset(Dataset):
    """CIFAR-10 with a fixed random permutation of labels."""
    def __init__(self, base_dataset, seed):
        self.base = base_dataset
        self.transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
        # Create fixed label permutation
        rng = np.random.RandomState(seed)
        self.label_map = rng.permutation(len(base_dataset))
        # Extract all labels and shuffle
        all_labels = [base_dataset[i][1] for i in range(len(base_dataset))]
        self.shuffled_labels = [all_labels[self.label_map[i]] for i in range(len(base_dataset))]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return self.transform(img), self.shuffled_labels[idx]


class FashionMNISTDataset(Dataset):
    """Fashion-MNIST resized to 32×32, 3-channel, normalized to CIFAR stats."""
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            lambda x: x.repeat(3, 1, 1),  # 1ch → 3ch
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.transform(img), label


class RotatedDataset(Dataset):
    """CIFAR-10 test set with fixed rotation for OOD evaluation."""
    def __init__(self, base_dataset, angle):
        self.base = base_dataset
        self.angle = angle
        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        img = T.functional.rotate(T.functional.to_pil_image(T.ToTensor()(img)), self.angle)
        img = self.normalize(T.ToTensor()(img))
        return img, label


print("✅ Dataset classes defined")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 4: Noise Generation                                          ║
# ╚══════════════════════════════════════════════════════════════════════╝

def make_frozen_projection():
    """Create frozen 1×1 conv: 128 → 3 channels. Seed=0, never trained."""
    torch.manual_seed(0)
    proj = nn.Conv2d(128, 3, kernel_size=1, bias=False)
    proj.requires_grad_(False)
    proj.eval()
    return proj.to(DEVICE)


def normalise_noise(noise, noise_std):
    """Per-sample normalise to 0 mean / unit var, then scale."""
    # noise: [B, C, H, W]
    mean = noise.mean(dim=[1, 2, 3], keepdim=True)
    std = noise.std(dim=[1, 2, 3], keepdim=True) + 1e-8
    return (noise - mean) / std * noise_std


def generate_white_noise(shape, noise_std):
    """Gaussian i.i.d. noise, per-sample normalised."""
    noise = torch.randn(shape, device=DEVICE)
    return normalise_noise(noise, noise_std)


def generate_pink_noise_2d(shape, noise_std):
    """2D spatially-correlated 1/f noise via spectral synthesis."""
    B, C, H, W = shape
    fy = torch.fft.fftfreq(H, device=DEVICE).unsqueeze(1)
    fx = torch.fft.fftfreq(W, device=DEVICE).unsqueeze(0)
    freqs = fy ** 2 + fx ** 2  # [H, W]
    freqs[0, 0] = 1.0  # avoid div by zero
    amplitude = 1.0 / torch.sqrt(freqs)  # 1/f amplitude roll-off
    # Random phase per sample per channel
    phase = torch.rand(B, C, H, W, device=DEVICE) * 2 * math.pi
    spectrum = amplitude.unsqueeze(0).unsqueeze(0) * torch.exp(1j * phase)
    noise = torch.fft.ifft2(spectrum).real
    return normalise_noise(noise, noise_std)


def pixel_shuffle_per_sample(tensor):
    """Per-sample pixel-wise spatial permutation. Destroys spatial structure,
    preserves per-sample marginal statistics."""
    B, C, H, W = tensor.shape
    result = tensor.clone()
    for i in range(B):
        perm = torch.randperm(H * W, device=tensor.device)
        result[i] = result[i].view(C, -1)[:, perm].view(C, H, W)
    return result


def get_exhaust_noise(B_model, projection, images_B, noise_std, shuffle_pixels=False):
    """Extract layer 2 activations from frozen B, project, normalise."""
    with torch.no_grad():
        raw = B_model.get_layer2_activations(images_B)  # [B, 128, H', W']
        spatial = F.interpolate(raw, size=(32, 32), mode='bilinear', align_corners=False)
        projected = projection(spatial)  # [B, 3, 32, 32]
        if shuffle_pixels:
            projected = pixel_shuffle_per_sample(projected)
        return normalise_noise(projected, noise_std)


print("✅ Noise generation functions defined")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 5: Cycling B Loader (re-shuffle on exhaust)                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

class CyclingLoader:
    """Infinite iterator over a dataset that re-shuffles with a new seed
    each time the dataset is exhausted. Prevents finite noise vocabulary."""
    def __init__(self, dataset, batch_size, initial_seed, num_workers=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rng = random.Random(initial_seed)
        self._make_loader()

    def _make_loader(self):
        seed = self.rng.randint(0, 2**31)
        g = torch.Generator()
        g.manual_seed(seed)
        self.loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True, generator=g,
            persistent_workers=False
        )
        self.iter = iter(self.loader)

    def __next__(self):
        try:
            return next(self.iter)
        except StopIteration:
            self._make_loader()  # new shuffle seed
            return next(self.iter)


print("✅ CyclingLoader defined")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 6: Training + Evaluation                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += model(x).argmax(1).eq(y).sum().item()
            total += x.size(0)
    return correct / total


def weight_mag(model):
    mags = [p.data.abs().mean().item() for p in model.parameters() if p.dim() >= 2]
    return float(np.mean(mags))


def pretrain_B(model, train_dataset, epochs, seed, label=""):
    """Pre-train a B network. Returns final test accuracy."""
    set_seed(seed)
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(train_dataset, batch_size=CFG['batch_size'], shuffle=True,
                        num_workers=CFG['num_workers'], drop_last=True, generator=g)

    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"    {label} ep {epoch:>2}")

    model.eval()
    return model


def freeze_model(model):
    """Freeze all parameters, set eval mode."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


print("✅ Training functions defined")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 7: Phase 0 — Pre-train B Networks                            ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("  PHASE 0: PRE-TRAINING B NETWORKS")
print("=" * 70)

# Load raw datasets
cifar_train_raw = torchvision.datasets.CIFAR10('./data', train=True, download=True)
cifar_test_raw = torchvision.datasets.CIFAR10('./data', train=False, download=True)
fmnist_train_raw = torchvision.datasets.FashionMNIST('./data', train=True, download=True)
fmnist_test_raw = torchvision.datasets.FashionMNIST('./data', train=False, download=True)

# Normalized datasets for training B
cifar_train_ds = NormalizedDataset(cifar_train_raw)
cifar_test_ds = NormalizedDataset(cifar_test_raw)
fmnist_train_ds = FashionMNISTDataset(fmnist_train_raw)
fmnist_test_ds = FashionMNISTDataset(fmnist_test_raw)

# Test loaders for evaluating B accuracy
cifar_test_loader = DataLoader(cifar_test_ds, batch_size=CFG['batch_size'],
                               shuffle=False, num_workers=CFG['num_workers'])
fmnist_test_loader = DataLoader(fmnist_test_ds, batch_size=CFG['batch_size'],
                                shuffle=False, num_workers=CFG['num_workers'])

b_accuracies = {}
t_phase0 = time.time()

for seed in CFG['seeds']:
    print(f"\n  ── Seed {seed} ──")

    # B-CIFAR
    path_bc = os.path.join(B_SAVE_DIR, f'B_cifar_seed{seed}.pt')
    if os.path.exists(path_bc):
        print(f"    B-CIFAR: loading from cache")
        m = SmallResNet()
        m.load_state_dict(torch.load(path_bc, map_location=DEVICE))
        m = m.to(DEVICE)
    else:
        set_seed(seed + CFG['b_seed_offsets']['cifar'])
        m = SmallResNet()
        m = pretrain_B(m, cifar_train_ds, CFG['epochs'],
                       seed + CFG['b_seed_offsets']['cifar'], label="B-CIFAR")
        torch.save(m.state_dict(), path_bc)
    acc = evaluate(m, cifar_test_loader)
    b_accuracies[f'B_cifar_seed{seed}'] = acc
    print(f"    B-CIFAR acc: {acc:.4f}")

    # B-FMNIST
    path_bf = os.path.join(B_SAVE_DIR, f'B_fmnist_seed{seed}.pt')
    if os.path.exists(path_bf):
        print(f"    B-FMNIST: loading from cache")
        m = SmallResNet()
        m.load_state_dict(torch.load(path_bf, map_location=DEVICE))
        m = m.to(DEVICE)
    else:
        set_seed(seed + CFG['b_seed_offsets']['fmnist'])
        m = SmallResNet()
        m = pretrain_B(m, fmnist_train_ds, CFG['epochs'],
                       seed + CFG['b_seed_offsets']['fmnist'], label="B-FMNIST")
        torch.save(m.state_dict(), path_bf)
    acc = evaluate(m, fmnist_test_loader)
    b_accuracies[f'B_fmnist_seed{seed}'] = acc
    print(f"    B-FMNIST acc: {acc:.4f}")

    # B-RANDOM (CIFAR images, shuffled labels)
    path_br = os.path.join(B_SAVE_DIR, f'B_random_seed{seed}.pt')
    if os.path.exists(path_br):
        print(f"    B-RANDOM: loading from cache")
        m = SmallResNet()
        m.load_state_dict(torch.load(path_br, map_location=DEVICE))
        m = m.to(DEVICE)
    else:
        set_seed(seed + CFG['b_seed_offsets']['random'])
        shuffled_ds = ShuffledLabelDataset(cifar_train_raw, seed + CFG['b_seed_offsets']['random'])
        m = SmallResNet()
        m = pretrain_B(m, shuffled_ds, CFG['epochs'],
                       seed + CFG['b_seed_offsets']['random'], label="B-RANDOM")
        torch.save(m.state_dict(), path_br)
    acc = evaluate(m, cifar_test_loader)
    b_accuracies[f'B_random_seed{seed}'] = acc
    print(f"    B-RANDOM acc: {acc:.4f} (expected low)")

    # C-DEAD (random init, never trained)
    path_cd = os.path.join(B_SAVE_DIR, f'C_dead_seed{seed}.pt')
    if os.path.exists(path_cd):
        print(f"    C-DEAD: loading from cache")
    else:
        set_seed(seed + CFG['b_seed_offsets']['dead'])
        m = SmallResNet()
        torch.save(m.state_dict(), path_cd)
    print(f"    C-DEAD: saved (no training)")

phase0_time = time.time() - t_phase0
print(f"\n  Phase 0 complete in {phase0_time/60:.1f} min")
print(f"  B accuracies: {json.dumps({k: round(v, 4) for k, v in b_accuracies.items()}, indent=2)}")

# Save B accuracies
with open(os.path.join(SAVE_DIR, 'b_accuracies.json'), 'w') as f:
    json.dump(b_accuracies, f, indent=2)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 8: Phase 1 — Main Experiment                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("  PHASE 1: MAIN EXPERIMENT — 7 ARMS × 5 SEEDS")
print("=" * 70)

# Create frozen projection layer (shared across all exhaust arms)
projection = make_frozen_projection()
print(f"  Frozen projection: {projection}")

# OOD test loaders
ood_loaders = {}
for rot in CFG['ood_rotations']:
    ood_ds = RotatedDataset(cifar_test_raw, rot)
    ood_loaders[rot] = DataLoader(ood_ds, batch_size=CFG['batch_size'],
                                  shuffle=False, num_workers=CFG['num_workers'])

# Arm → B network mapping
ARM_B_MAP = {
    'CROSS-EXHAUST': ('B_fmnist', fmnist_train_ds),
    'SAME-EXHAUST': ('B_cifar', cifar_train_ds),
    'RANDOM-LABEL-EXHAUST': ('B_random', cifar_train_ds),
    'SHUFFLED-EXHAUST': ('B_cifar', cifar_train_ds),
    'DEAD-EXHAUST': ('C_dead', cifar_train_ds),
}

# Map B type to weight file prefix
B_FILE_MAP = {
    'B_cifar': 'B_cifar',
    'B_fmnist': 'B_fmnist',
    'B_random': 'B_random',
    'C_dead': 'C_dead',
}

# Which dataset does B's loader iterate over?
ARM_B_DATASET = {
    'CROSS-EXHAUST': fmnist_train_ds,   # B-FMNIST sees Fashion-MNIST
    'SAME-EXHAUST': cifar_train_ds,     # B-CIFAR sees CIFAR-10
    'RANDOM-LABEL-EXHAUST': cifar_train_ds,
    'SHUFFLED-EXHAUST': cifar_train_ds,
    'DEAD-EXHAUST': cifar_train_ds,
}


def load_frozen_B(arm, seed):
    """Load and freeze the appropriate B network for this arm."""
    b_type, _ = ARM_B_MAP[arm]
    prefix = B_FILE_MAP[b_type]
    path = os.path.join(B_SAVE_DIR, f'{prefix}_seed{seed}.pt')
    model = SmallResNet()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model = model.to(DEVICE)
    return freeze_model(model)


def run_arm(arm_name, seed, arm_idx):
    """Run one arm for one seed. Returns history dict."""
    set_seed(seed)

    # A network
    A = SmallResNet().to(DEVICE)
    optimizer = optim.AdamW(A.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'])
    criterion = nn.CrossEntropyLoss()

    # A's data loader (CIFAR-10) — independent seed
    g_a = torch.Generator()
    g_a.manual_seed(seed)
    a_loader = DataLoader(cifar_train_ds, batch_size=CFG['batch_size'], shuffle=True,
                          num_workers=CFG['num_workers'], drop_last=True, generator=g_a)

    # B network + loader (if exhaust arm)
    B_model = None
    b_cycling = None
    is_exhaust = arm_name not in ('WHITE', 'PINK')
    is_shuffled = arm_name == 'SHUFFLED-EXHAUST'

    if is_exhaust:
        B_model = load_frozen_B(arm_name, seed)
        b_dataset = ARM_B_DATASET[arm_name]
        # Independent B loader seed
        b_initial_seed = seed * 7 + arm_idx
        b_cycling = CyclingLoader(b_dataset, CFG['batch_size'], b_initial_seed,
                                  num_workers=CFG['num_workers'])

    history = []
    noise_std = CFG['noise_std']

    for epoch in range(CFG['epochs']):
        A.train()
        total_loss, correct, total = 0.0, 0, 0

        for x_a, y_a in a_loader:
            x_a, y_a = x_a.to(DEVICE), y_a.to(DEVICE)

            # Generate noise
            if arm_name == 'WHITE':
                noise = generate_white_noise(x_a.shape, noise_std)
            elif arm_name == 'PINK':
                noise = generate_pink_noise_2d(x_a.shape, noise_std)
            else:
                # Exhaust arms
                x_b, _ = next(b_cycling)
                x_b = x_b.to(DEVICE)
                noise = get_exhaust_noise(B_model, projection, x_b, noise_std,
                                          shuffle_pixels=is_shuffled)

            # Inject noise
            x_noisy = x_a + noise

            # Forward/backward (only A trains)
            optimizer.zero_grad()
            out = A(x_noisy)
            loss = criterion(out, y_a)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_a.size(0)
            correct += out.argmax(1).eq(y_a).sum().item()
            total += x_a.size(0)

        scheduler.step()

        # Evaluate
        train_acc = correct / total
        train_loss = total_loss / total
        test_acc = evaluate(A, cifar_test_loader)

        ood_accs = {}
        for rot in CFG['ood_rotations']:
            ood_accs[f'rot{rot}'] = evaluate(A, ood_loaders[rot])
        mean_ood = float(np.mean(list(ood_accs.values())))

        wmag = weight_mag(A)

        history.append({
            'epoch': epoch,
            'loss': train_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'ood_30': ood_accs['rot30'],
            'ood_45': ood_accs['rot45'],
            'ood_60': ood_accs['rot60'],
            'mean_ood': mean_ood,
            'weight_mag': wmag,
        })

        if epoch % 10 == 0 or epoch == CFG['epochs'] - 1:
            print(f"    ep {epoch:>2} | loss {train_loss:.4f} | train {train_acc:.4f} | "
                  f"test {test_acc:.4f} | OOD {mean_ood:.4f} | wmag {wmag:.4f}")

    return history


# ── Run all arms ──
all_results = {}
total_runs = len(CFG['arms']) * len(CFG['seeds'])
run_count = 0
t_phase1 = time.time()

for arm_idx, arm_name in enumerate(CFG['arms']):
    print(f"\n{'='*60}")
    print(f"  ARM: {arm_name}")
    print(f"{'='*60}")

    for seed in CFG['seeds']:
        run_count += 1
        run_key = f"{arm_name}_seed{seed}"

        # Check if already done
        save_path = os.path.join(SAVE_DIR, f'{run_key}.json')
        if os.path.exists(save_path):
            print(f"\n  [{run_count}/{total_runs}] {run_key} — cached, loading")
            with open(save_path) as f:
                data = json.load(f)
            all_results[run_key] = data['history']
            continue

        elapsed = time.time() - t_phase1
        if run_count > 1:
            avg_per_run = elapsed / (run_count - 1)
            remaining = avg_per_run * (total_runs - run_count + 1)
            eta = f"ETA {remaining/60:.0f}min"
        else:
            eta = ""

        print(f"\n  [{run_count}/{total_runs}] {run_key} {eta}")
        t0 = time.time()
        history = run_arm(arm_name, seed, arm_idx)
        dt = time.time() - t0
        all_results[run_key] = history

        # Save incrementally
        result_data = {
            'arm': arm_name,
            'seed': seed,
            'final_test': history[-1]['test_acc'],
            'final_ood': history[-1]['mean_ood'],
            'history': history,
        }
        with open(save_path, 'w') as f:
            json.dump(result_data, f, indent=2)

        final = history[-1]
        print(f"  ✅ {run_key} DONE ({dt:.0f}s) | test={final['test_acc']:.4f} OOD={final['mean_ood']:.4f}")

phase1_time = time.time() - t_phase1
print(f"\n  Phase 1 complete in {phase1_time/60:.1f} min")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 9: Analysis + VERDICT                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("  RESULTS SUMMARY — Experiment 3a: Exhaust Canvas")
print("=" * 70 + "\n")

# ── Positive Controls ──
print("  ── POSITIVE CONTROLS ──\n")
for btype in ['B_cifar', 'B_fmnist']:
    accs = [b_accuracies.get(f'{btype}_seed{s}', 0) for s in CFG['seeds']]
    m, s = np.mean(accs), np.std(accs)
    threshold = 0.80 if 'cifar' in btype else 0.85
    status = "✅" if m >= threshold else "❌"
    print(f"  {btype}: {m:.4f} ± {s:.4f} (need ≥{threshold}) {status}")

# ── Main Results Table ──
print(f"\n  ── MAIN RESULTS (final epoch) ──\n")
print(f"  {'Arm':>22} | {'Test':>12} | {'OOD mean':>12} | {'OOD 30':>8} | {'OOD 45':>8} | {'OOD 60':>8}")
print("  " + "-" * 85)

arm_stats = {}
for arm in CFG['arms']:
    tests, oods, o30s, o45s, o60s = [], [], [], [], []
    for seed in CFG['seeds']:
        h = all_results[f"{arm}_seed{seed}"][-1]
        tests.append(h['test_acc'])
        oods.append(h['mean_ood'])
        o30s.append(h['ood_30'])
        o45s.append(h['ood_45'])
        o60s.append(h['ood_60'])

    mt, st = np.mean(tests), np.std(tests)
    mo, so = np.mean(oods), np.std(oods)
    arm_stats[arm] = {'test_mean': mt, 'test_std': st, 'ood_mean': mo, 'ood_std': so}
    print(f"  {arm:>22} | {mt:.4f}±{st:.4f} | {mo:.4f}±{so:.4f} | "
          f"{np.mean(o30s):>8.4f} | {np.mean(o45s):>8.4f} | {np.mean(o60s):>8.4f}")

# ── Per-seed detail ──
print(f"\n  ── PER-SEED MEAN OOD ──\n")
header = f"  {'Seed':>6}"
for a in CFG['arms']:
    header += f"  {a[:10]:>10}"
print(header)
print("  " + "-" * (8 + 12 * len(CFG['arms'])))
for seed in CFG['seeds']:
    row = f"  {seed:>6}"
    for a in CFG['arms']:
        row += f"  {all_results[f'{a}_seed{seed}'][-1]['mean_ood']:>10.4f}"
    print(row)

# ── Kill Tests ──
print(f"\n  ── KILL TESTS ──\n")

from scipy import stats as scipy_stats

def paired_ttest(arm1, arm2):
    """Paired t-test on per-seed OOD means."""
    vals1 = [all_results[f"{arm1}_seed{s}"][-1]['mean_ood'] for s in CFG['seeds']]
    vals2 = [all_results[f"{arm2}_seed{s}"][-1]['mean_ood'] for s in CFG['seeds']]
    diffs = [a - b for a, b in zip(vals1, vals2)]
    mean_diff = np.mean(diffs)
    se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
    t_crit = scipy_stats.t.ppf(0.975, df=len(diffs)-1)
    ci_lo = mean_diff - t_crit * se
    ci_hi = mean_diff + t_crit * se
    t_stat, p_val = scipy_stats.ttest_rel(vals1, vals2)
    return mean_diff, ci_lo, ci_hi, t_stat, p_val

# Primary: CROSS > WHITE
d, lo, hi, t, p = paired_ttest('CROSS-EXHAUST', 'WHITE')
primary_pass = lo > 0
print(f"  PRIMARY: CROSS-EXHAUST vs WHITE")
print(f"    Diff = {d:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], t={t:.3f}, p={p:.4f}")
print(f"    {'✅ CI excludes zero — CANVAS EFFECT' if primary_pass else '❌ CI includes zero — NULL NOT REJECTED'}")

# Spectral control: CROSS vs PINK
d2, lo2, hi2, t2, p2 = paired_ttest('CROSS-EXHAUST', 'PINK')
print(f"\n  SPECTRAL: CROSS-EXHAUST vs PINK")
print(f"    Diff = {d2:+.4f}, 95% CI [{lo2:+.4f}, {hi2:+.4f}], t={t2:.3f}, p={p2:.4f}")

# Spatial: CROSS vs SHUFFLED
d3, lo3, hi3, t3, p3 = paired_ttest('CROSS-EXHAUST', 'SHUFFLED-EXHAUST')
print(f"\n  SPATIAL: CROSS-EXHAUST vs SHUFFLED-EXHAUST")
print(f"    Diff = {d3:+.4f}, 95% CI [{lo3:+.4f}, {hi3:+.4f}], t={t3:.3f}, p={p3:.4f}")

# Secondary comparisons
print(f"\n  ── SECONDARY COMPARISONS ──")
for a, b in [('SAME-EXHAUST', 'CROSS-EXHAUST'),
             ('DEAD-EXHAUST', 'WHITE'),
             ('RANDOM-LABEL-EXHAUST', 'DEAD-EXHAUST')]:
    d_, lo_, hi_, t_, p_ = paired_ttest(a, b)
    print(f"  {a} vs {b}: diff={d_:+.4f} CI[{lo_:+.4f},{hi_:+.4f}] p={p_:.4f}")

# ── Determine verdict ──
white_ood = arm_stats['WHITE']['ood_mean']
cross_ood = arm_stats['CROSS-EXHAUST']['ood_mean']
pink_ood = arm_stats['PINK']['ood_mean']
shuffled_ood = arm_stats['SHUFFLED-EXHAUST']['ood_mean']

if primary_pass:
    if cross_ood > pink_ood and pink_ood > white_ood:
        verdict = "FULL_CANVAS"
        msg = "🟢 Learned structure > spectral colour > white. Strongest canvas result."
    elif abs(cross_ood - pink_ood) < 0.005:
        verdict = "SPECTRAL_ONLY"
        msg = "🟡 1/f spectral colour is the active ingredient."
    else:
        verdict = "CANVAS_CONFIRMED"
        msg = "🟢 Canvas effect confirmed — structured exhaust beats white noise."
else:
    verdict = "NULL_NOT_REJECTED"
    msg = "🔴 Hypothesis not supported — CROSS ≤ WHITE."

print(f"\n  VERDICT: {verdict}")
print(f"  {msg}")

# ── Weight fossilisation ──
print(f"\n  ── WEIGHT FOSSILISATION ──")
for arm in CFG['arms']:
    w0 = np.mean([all_results[f"{arm}_seed{s}"][0]['weight_mag'] for s in CFG['seeds']])
    wf = np.mean([all_results[f"{arm}_seed{s}"][-1]['weight_mag'] for s in CFG['seeds']])
    growth = (wf - w0) / w0 * 100 if w0 > 0 else 0
    print(f"  {arm:>22}: {w0:.4f} → {wf:.4f} ({growth:+.1f}%)")

# ── Save VERDICT.json ──
summary = {
    'protocol': CFG['protocol'],
    'verdict': verdict,
    'message': msg,
    'b_accuracies': b_accuracies,
    'arm_stats': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in arm_stats.items()},
    'primary_test': {
        'cross_vs_white_diff': float(d), 'ci_lo': float(lo), 'ci_hi': float(hi),
        'p_value': float(p), 'significant': primary_pass,
    },
    'config': CFG,
    'phase0_time_min': phase0_time / 60,
    'phase1_time_min': phase1_time / 60,
    'total_time_min': (phase0_time + phase1_time) / 60,
    'timestamp': datetime.now().isoformat(),
}

verdict_path = os.path.join(SAVE_DIR, 'VERDICT.json')
with open(verdict_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n  💾 VERDICT saved: {verdict_path}")
print(f"\n  Total time: {(phase0_time + phase1_time)/60:.1f} min")
print("\n" + "=" * 70)
print(f"  EXPERIMENT 3a COMPLETE — {verdict}")
print("=" * 70)
