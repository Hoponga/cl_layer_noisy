# train_baseline.py – Permuted/Split MNIST (or CIFAR-10) with small replay buffer
# -------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Grayscale
import random

# ---------------- hyper-parameters ----------------
T          = 5          # number of tasks/permutations
BATCH      = 128
REPLAY_MB  = 128
REPLAY_CAP = 4096*32
LR         = 1e-3
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PERMUTED   = False      # set True for Permuted-MNIST/CIFAR10
isMNIST    = True       # set False for CIFAR-10 variants
# --------------------------------------------------

# ---------------- dataset definitions -------------
class PermutedMNIST(Dataset):
    def __init__(self, base, perm):
        self.base, self.perm = base, perm
    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x.view(-1)[self.perm], y
    def __len__(self):
        return len(self.base)

class PermutedCIFAR10(Dataset):
    def __init__(self, base, perm):
        self.base, self.perm = base, perm
    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x.view(-1)[self.perm], y
    def __len__(self):
        return len(self.base)

class SplitMNIST(Dataset):
    def __init__(self, base, classes):
        self.dataset = base
        self.classes = set(int(c) for c in classes)
        self.indices = [i for i,(_,lbl) in enumerate(self.dataset) if int(lbl) in self.classes]
        if not self.indices:
            raise ValueError(f"No examples for classes {classes}")
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        img, lbl = self.dataset[self.indices[i]]
        return img.view(-1), lbl

class SplitCIFAR10(Dataset):
    def __init__(self, base, classes):
        self.dataset = base
        self.classes = set(int(c) for c in classes)
        self.indices = [i for i,(_,lbl) in enumerate(self.dataset) if int(lbl) in self.classes]
        if not self.indices:
            raise ValueError(f"No examples for classes {classes}")
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        img, lbl = self.dataset[self.indices[i]]
        return img.view(-1), lbl

# ---------------- prepare tasks -------------------
# Base datasets
mnist_base = MNIST('./data', train=True, download=True, transform=ToTensor())
cifar_base = CIFAR10('./data', train=True, download=True,
                     transform=Compose([Grayscale(), ToTensor()]))

tasks = []
if isMNIST:
    if PERMUTED:
        perms = [torch.randperm(784) for _ in range(T)]
        for p in perms:
            ds = PermutedMNIST(mnist_base, p)
            tasks.append(DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True))
    else:
        temp = torch.randperm(10)
        splits = [temp[i:i+2].tolist() for i in range(0,10,2)]
        for cls in splits:
            ds = SplitMNIST(mnist_base, cls)
            tasks.append(DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True))
else:
    if PERMUTED:
        perms = [torch.randperm(1024) for _ in range(T)]
        for p in perms:
            ds = PermutedCIFAR10(cifar_base, p)
            tasks.append(DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True))
    else:
        temp = torch.randperm(10)
        splits = [temp[i:i+2].tolist() for i in range(0,10,2)]
        for cls in splits:
            ds = SplitCIFAR10(cifar_base, cls)
            tasks.append(DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True))

# ---------------- replay buffer -------------------
class ReplayBuf:
    def __init__(self, cap):
        self.x, self.y, self.cap = [], [], cap
    def add_batch(self, x, y):
        for xi, yi in zip(x.cpu(), y.cpu()):
            if len(self.x) < self.cap:
                self.x.append(xi); self.y.append(int(yi))
            else:
                k = random.randrange(self.cap)
                self.x[k] = xi; self.y[k] = int(yi)
    def sample(self, m):
        if not self.x: return None, None
        idx = random.sample(range(len(self.x)), min(m, len(self.x)))
        xs = torch.stack([self.x[i] for i in idx]).to(device)
        ys = torch.tensor([self.y[i] for i in idx], device=device)
        return xs, ys

replay = ReplayBuf(REPLAY_CAP)

# ---------------- baseline model ------------------
class BaselineModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128), 
            nn.ReLU(inplace = True), 
            nn.Linear(128,10)
        )
    def forward(self, x):
        return self.net(x)

input_dim = 784 if isMNIST else 1024
model = BaselineModel(input_dim).to(device)
optim = torch.optim.Adam(model.parameters(), lr=LR)

# --------------- accuracy helper ------------------
@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1)==y).sum().item()
        total   += y.size(0)
    return 100 * correct / total

# ---------------- training loop -------------------
def train_task(model, loader, optim, epochs=1):
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # sample replay batch
            xr, yr = replay.sample(REPLAY_MB)

            # forward new
            logits = model(x)
            loss   = F.cross_entropy(logits, y)

            # forward replay
            if xr is not None:
                logits_r = model(xr)
                loss    += F.cross_entropy(logits_r, yr)

            optim.zero_grad()
            loss.backward()
            optim.step()

            # add to buffer
            replay.add_batch(x.detach(), y.detach())

# --------------- main continual loop ---------------
print("Baseline continual learning with replay buffer\n")
for t, loader in enumerate(tasks):
    print(f"=== Task {t} ===")
    if not PERMUTED:
        print("Classes:", loader.dataset.classes if hasattr(loader.dataset, 'classes') else "N/A")
    train_task(model, loader, optim, epochs=1)
    for s in range(t+1):
        acc = accuracy(model, tasks[s])
        print(f"  Eval task {s}: {acc:5.2f}%")
