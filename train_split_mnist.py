# train_ot_split_mnist.py
# -- Continual split-MNIST (2 classes/task) with OT + replay + hard centroids

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import random
import numpy as np

# ------------------ Hyper-parameters ------------------
T            = 5         # number of tasks (pairs of digits)
BATCH        = 128
REPLAY_MB    = 64
REPLAY_CAP   = 800

D_CLS, D_TSK = 128, 32
K            = T         # one prototype per task
LR           = 1e-3

LAM_OT      = 2.0
LAM_DIV     = 3e-3
LAM_ORTH    = 1e-2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# ------------------ Split-MNIST tasks ------------------
base_ds = MNIST('./data', train=True, download=True, transform=ToTensor())
labels = np.array(base_ds.targets)

# Randomly pair digits into T tasks
digits = list(range(10))
random.shuffle(digits)
task_pairs = [digits[i:i+2] for i in range(0, 10, 2)]

tasks = []
for pair in task_pairs:
    idx = np.where((labels == pair[0]) | (labels == pair[1]))[0]
    subset = Subset(base_ds, idx.tolist())
    loader = DataLoader(subset, batch_size=BATCH, shuffle=True, drop_last=True)
    tasks.append(loader)

# ------------------ Replay Buffer ------------------
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
        if not self.x:
            return None, None
        idx = random.sample(range(len(self.x)), min(m, len(self.x)))
        xs = torch.stack([self.x[i] for i in idx]).to(device)
        ys = torch.tensor([self.y[i] for i in idx]).to(device)
        return xs, ys

replay = ReplayBuf(REPLAY_CAP)

# ------------------ Sinkhorn ------------------
def sinkhorn(scores, eps=0.1, iters=3):
    scores = scores - scores.max(1, keepdim=True).values
    Q = torch.exp(scores / eps)
    Q = Q / Q.sum()
    r = torch.ones(Q.size(0), device=Q.device) / Q.size(0)
    c = torch.ones(Q.size(1), device=Q.device) / Q.size(1)
    for _ in range(iters):
        Q *= (r / (Q.sum(1) + 1e-9)).unsqueeze(1)
        Q *= (c / (Q.sum(0) + 1e-9)).unsqueeze(0)
    return Q / Q.sum()

# ------------------ OT Head ------------------
class OTHead(nn.Module):
    def __init__(self, d_t, K, delta=0.3):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(K, d_t))
        nn.init.normal_(self.prototypes)
        self.delta = delta

    def forward(self, z_t, replay_z=None, *, tau=0.5, eps=0.1, beta=0.9):
        proto_n = F.normalize(self.prototypes, dim=1)        # [K, d_t]
        scores  = z_t @ proto_n.T / tau                      # [B, K]
        with torch.no_grad():
            Q = sinkhorn(scores, eps)                        # [B, K]
            # current batch mean
            bs   = Q.sum(0) + 1e-9
            mean = (Q.T @ z_t) / bs.unsqueeze(1)             # [K, d_t]
            # replay batch mean
            if replay_z is not None:
                Qr   = sinkhorn(replay_z @ proto_n.T / tau, eps)
                bs_r = Qr.sum(0) + 1e-9
                mean_r = (Qr.T @ replay_z) / bs_r.unsqueeze(1)
                mean = 0.5 * mean + 0.5 * mean_r
            # EMA update
            self.prototypes.data.mul_(beta).add_((1 - beta) * mean)
            self.prototypes.data = F.normalize(self.prototypes.data, dim=1)

        # losses
        ot_loss  = -(Q * F.log_softmax(scores, 1)).sum(1).mean()
        cos_m    = proto_n @ proto_n.T
        div_loss = F.relu((cos_m - self.delta) * torch.triu(torch.ones_like(cos_m), 1)).mean()

        # hard centroid per sample
        idx            = Q.argmax(1)
        hard_centroid  = self.prototypes[idx]               # [B, d_t]

        return ot_loss, div_loss, hard_centroid

# ------------------ Full Model ------------------
class OTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone   = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(inplace=True),
            nn.Linear(256, D_CLS + D_TSK)
        )
        self.task_head  = OTHead(D_TSK, K)
        self.classifier = nn.Linear(D_CLS + D_TSK, 10)

    def forward(self, x, replay_z=None):
        z         = self.backbone(x)
        z_cls     = F.normalize(z[:, :D_CLS], dim=1)
        z_task    = F.normalize(z[:, D_CLS:], dim=1)

        ot, div, ct= self.task_head(z_task, replay_z=replay_z)
        logits     = self.classifier(torch.cat([z_cls, ct.detach()], dim=1))
        return logits, ot, div

# ------------------ Accuracy ------------------
@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)
        #x, y = x.to(device), y.to(device)
        log, *_ = model(x)
        print(log.argmax(1), y)
        correct += (log.argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100 * correct / total

# ------------------ Train per Task ------------------
def train_task(model, loader, optimizer, epochs=1):
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            # replay embeddings
            xr, _ = replay.sample(REPLAY_MB)
            if xr is not None:
                #xr = xr.view(xr.size(0), -1)
                with torch.no_grad():
                    z_r = model.backbone(xr.to(device))[:, -D_TSK:]
                    z_r = F.normalize(z_r, dim=1)
            else:
                z_r = None
            logits, ot, div = model(x, replay_z=z_r)
            cls_loss = F.cross_entropy(logits, y)
            loss     = cls_loss + LAM_OT*ot + LAM_DIV*div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            replay.add_batch(x.detach(), y.detach())

# ------------------ Main Continual Loop ------------------
model = OTModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("Starting continual Split-MNIST training\n")
for t, loader in enumerate(tasks):
    print(f"=== Task {t}: digits {task_pairs[t]} ===")
    train_task(model, loader, optimizer, epochs=1)
    for s in range(t+1):
        print(f"  Eval task {s}: {accuracy(model, tasks[s]):5.2f}%")
