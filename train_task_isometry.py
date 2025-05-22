#!/usr/bin/env python3
# train_relative_replay_v4.py
#
# Continual learning with pairwise-distance anchoring
#  ✦ NO teacher network — we snapshot embeddings in the replay buffer
#  ✦ Balanced task×class buffer stores (image, label, z_ref)
#  ✦ Buffer updated only *after* each task is finished
#  ✦ Permuted-MNIST permutations are applied once up front
# -------------------------------------------------------------------
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST

# ───────────────────────────── flags ──────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--tasks",      choices=["permuted","split"], default="permuted",
               help="‘permuted’ or ‘split’ MNIST")
p.add_argument("--T",          type=int,   default=10,   help="# tasks (perm-MNIST)")
p.add_argument("--epochs",     type=int,   default=2,    help="epochs per task")
p.add_argument("--batch",      type=int,   default=128,  help="batch size")
p.add_argument("--latent",     type=int,   default=256,  help="latent dimension")
p.add_argument("--lr",         type=float, default=1e-3, help="learning rate")
p.add_argument("--lam_man",    type=float, default=1.0,  help="λ for manifold loss")
p.add_argument("--replay_cap", type=int,   default=600,  help="replay buffer capacity")
p.add_argument("--replay_mb",  type=int,   default=128,  help="replay minibatch size")
p.add_argument("--cuda",       action="store_true",       help="use CUDA if available")

p.add_argument("--lam_supcon",   type=float, default=0.01,
               help="weight for supervised-contrastive loss")
p.add_argument("--lam_centroid", type=float, default=0.5,
               help="weight for inter-bank centroid margin")
p.add_argument("--centroid_margin", type=float, default=1.5,
               help="desired min distance between bank centroids")


args = p.parse_args()

device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
print("Device:", device)



def supcon_loss(z, y, temperature=0.07):
    """
    Normalised embeddings z:[N,D], labels y:[N]
    Standard SupCon (from Khosla et al. 2020)
    """
    z = F.normalize(z, p=2, dim=1)
    sim = (z @ z.t()) / temperature             # cosine similarities
    # mask_{i,j}=1 when same label & i!=j
    mask_pos = (y.unsqueeze(1) == y.unsqueeze(0)).float()
    mask_pos.fill_diagonal_(0)
    # log-sum-exp of all except itself
    logits   = sim - 1e9*torch.eye(len(z), device=z.device)
    log_denom= torch.logsumexp(logits, dim=1, keepdim=True)
    loss_mat = -sim + log_denom
    loss     = (mask_pos * loss_mat).sum() / (mask_pos.sum() + 1e-8)
    return loss


# ─────────────────────────── datasets ─────────────────────────────
def get_tasks():
    # Load raw MNIST once (no per-sample transform)
    base = MNIST("./data", train=True, download=True, transform=None)
    X = base.data.view(-1, 784).float().div_(255.0)  # [60000, 784], in [0,1]
    Y = base.targets                                # [60000]

    loaders = []
    if args.tasks == "permuted":
        # Precompute T random permutations
        perms = [torch.randperm(784) for _ in range(args.T)]
        for perm in perms:
            Xp = X[:, perm]                         # apply perm once
            ds = TensorDataset(Xp, Y)
            loaders.append(DataLoader(ds,
                                      batch_size=args.batch,
                                      shuffle=True,
                                      drop_last=True))
    else:
        # Split-MNIST: five 2-class tasks (0–1, 2–3, …, 8–9)
        splits = [list(range(i, i+2)) for i in range(0, 10, 2)]
        for cls in splits:
            mask = (Y.unsqueeze(1) == torch.tensor(cls)).any(dim=1)
            Xs, Ys = X[mask], Y[mask]
            ds = TensorDataset(Xs, Ys)
            loaders.append(DataLoader(ds,
                                      batch_size=args.batch,
                                      shuffle=True,
                                      drop_last=True))
    return loaders

tasks = get_tasks()
C = 10  # always 10 classes total

import random, torch

class MultiBankReplayBuffer:
    """
    - Pre-allocates up to max_banks task-banks.
    - Each bank holds up to bank_capacity examples (x, y, z_ref).
    - Sampling draws m examples from *one* randomly chosen non-empty bank.
    - When you add a new task > max_banks, you evict the *oldest* bank.
    """
    def __init__(self, max_banks, bank_capacity, device):
        self.max_banks     = max_banks
        self.bank_capacity = bank_capacity
        self.device        = device
        self.banks         = {}   # tid → list of (x, y, z_ref)
        self.order         = []   # FIFO order of tids seen

    @torch.no_grad()
    def add(self, xs, ys, zs, tid):
        # 1) If this tid is new, maybe evict the oldest
        if tid not in self.banks:
            if len(self.banks) >= self.max_banks:
                oldest = self.order.pop(0)
                del self.banks[oldest]
            self.banks[tid] = []
            self.order.append(tid)

        bank = self.banks[tid]
        cap  = self.bank_capacity

        # 2) Reservoir‐style add into this one bank
        for x, y, z in zip(xs.cpu(), ys.cpu(), zs.cpu()):
            item = (x.clone(), int(y), z.clone())
            if len(bank) < cap:
                bank.append(item)
            else:
                idx = random.randrange(cap)
                bank[idx] = item

    def sample(self, m):
        # nothing cached yet?
        if not self.banks:
            return None, None, None

        # pick one non‐empty bank at random
        non_empty = [tid for tid, b in self.banks.items() if b]
        if not non_empty:
            return None, None, None

        tid = random.choice(non_empty)
        bank = self.banks[tid]

        k = min(m, len(bank))
        idxs = random.sample(range(len(bank)), k)

        xs = torch.stack([bank[i][0] for i in idxs]).to(self.device)
        ys = torch.tensor([bank[i][1] for i in idxs],
                          device=self.device)
        zs = torch.stack([bank[i][2] for i in idxs]).to(self.device)
        return xs, ys, zs


# ─────────────────── balanced task×class replay ─────────────────────
class TaskClassReplayBuf:
    """
    Stores triples (x_img, y_lbl, z_ref) with balanced reservoir over
    tasks and classes.
    """
    def __init__(self, cap, C, device):
        self.cap, self.C, self.dev = cap, C, device
        self.store = {}  # dict[task][cls] -> list[(x,y,z_ref)]

    def _cell_cap(self):
        return self.cap if not self.store else max(1, self.cap // (len(self.store)*self.C))

    def _total(self):
        return sum(len(bucket)
                   for task in self.store.values()
                   for bucket in task.values())

    @torch.no_grad()
    def add(self, xs, ys, z_refs, tid):
        xs, ys, z_refs = xs.cpu(), ys.cpu(), z_refs.cpu()
        if tid not in self.store:
            self.store[tid] = {c: [] for c in range(self.C)}
        cell_cap = self._cell_cap()

        for x, y, z in zip(xs, ys, z_refs):
            bucket = self.store[tid][int(y)]
            item = (x.clone(), int(y), z.clone())
            if len(bucket) < cell_cap:
                bucket.append(item)
            else:
                bucket[random.randrange(cell_cap)] = item

        # Trim if over capacity
        while self._total() > self.cap:
            cell_cap = self._cell_cap()
            for t in list(self.store):
                for c in range(self.C):
                    bucket = self.store[t][c]
                    while len(bucket) > cell_cap:
                        bucket.pop(random.randrange(len(bucket)))
                    if self._total() <= self.cap:
                        return

    def sample(self, m):
        if self._total() == 0:
            return None, None, None
        tasks = list(self.store)
        per_t = max(1, m // len(tasks))
        xs, ys, zs = [], [], []
        for t in tasks:
            per_c = max(1, per_t // self.C)
            for c in range(self.C):
                bucket = self.store[t][c]
                if not bucket:
                    continue
                k = min(per_c, len(bucket))
                for i in random.sample(range(len(bucket)), k):
                    x, y, z = bucket[i]
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
        if not xs:
            return None, None, None
        return (torch.stack(xs).to(self.dev),
                torch.tensor(ys, device=self.dev),
                torch.stack(zs).to(self.dev))

per_bank_cap = args.replay_cap // args.T
buf = MultiBankReplayBuffer(max_banks=args.T,
                            bank_capacity=per_bank_cap,
                            device=device)

# ───────────────────────── model ────────────────────────────────
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, args.latent)
        )
        self.cls = nn.Linear(args.latent, C)

    def embed(self, x):
        return F.normalize(self.backbone(x), p=2, dim=1)

    def forward(self, x):
        return self.cls(self.embed(x))

def pairwise(mat):
    # pairwise Euclidean distances
    return torch.cdist(mat, mat, p=2)

def manifold_loss(z_cur, z_ref):
    return (pairwise(z_cur) - pairwise(z_ref)).pow(2).mean()

student = Net().to(device)
opt     = torch.optim.Adam(student.parameters(), lr=args.lr)

# ─────────────────────── training loop ────────────────────────────
for tid, loader in enumerate(tasks):
    print(f"\n--- Task {tid} ---")
    student.train()
    for i in range(args.epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss  = F.cross_entropy(student(x), y)
             

            xr, yr, zr = buf.sample(args.replay_mb)   # ONE bank only
            if xr is not None:
                z_cur = student.embed(xr)

                # 1) standard replay CE
                loss += F.cross_entropy(student.cls(z_cur), yr)

                # 2) relative-distance anchoring
                m_loss = manifold_loss(z_cur, zr) 

                loss += args.lam_man * m_loss

                # 3) supervised-contrastive (labels = yr)
                if args.lam_supcon > 0:
                    sup_loss = supcon_loss(z_cur, yr)
                    
                    loss += args.lam_supcon * sup_loss
                print(m_loss, sup_loss)

                # 4) centroid-margin vs. *other* banks  (no task ids)
                if args.lam_centroid > 0 and len(buf.banks) > 1:
                    # centroid of the bank we just sampled
                    c_new = z_cur.mean(dim=0)
                    # centroids of all *other* banks
                    others = []
                    for t,b in buf.banks.items():
                        if b is buf.banks[buf.order[-1]]:  # buf.order[-1] is newest task id
                            continue
                        with torch.no_grad():
                            xs_, ys_, zs_ = zip(*b)
                            zs_ = torch.stack(zs_).to(device)
                            others.append(zs_.mean(dim=0))
                    if others:
                        others = torch.stack(others)                 # [B-1, D]
                        dists  = (others - c_new).norm(p=2, dim=1)
                        hinge  = F.relu(args.centroid_margin - dists)
                        loss  += args.lam_centroid * hinge.mean()



            opt.zero_grad()
            loss.backward()
            opt.step()

    # ── After task: snapshot embeddings into buffer ────────────────
    student.eval()
    snap_loader = DataLoader(loader.dataset,
                             batch_size=args.batch,
                             shuffle=False)
    with torch.no_grad():
        for xs, ys in snap_loader:
            xs, ys = xs.to(device), ys.to(device)
            zs     = student.embed(xs)
            buf.add(xs, ys, zs, tid)

    # quick accuracy report
    for s in range(tid+1):
        corr = tot = 0
        for xv, yv in tasks[s]:
            xv, yv = xv.to(device), yv.to(device)
            corr += (student(xv).argmax(1) == yv).sum().item()
            tot  += yv.size(0)
        print(f"  acc task {s}: {100*corr/tot:5.2f}%")

# ─────────────────────── t-SNE visualisation ──────────────────────
print("\nRunning t-SNE …")
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

student.eval()
Z, t_lbl, c_lbl = [], [], []
rng = np.random.default_rng(0)

with torch.no_grad():
    for tid, loader in enumerate(tasks):
        x_all, y_all = next(iter(loader))
        idx = rng.choice(len(x_all), size=min(512, len(x_all)), replace=False)
        x, y = x_all[idx].to(device), y_all[idx]
        Z.append(student.embed(x).cpu())
        t_lbl.extend([tid]*len(idx))
        c_lbl.extend(y.cpu().tolist())

Z = torch.cat(Z).numpy()
t_lbl = np.array(t_lbl)
c_lbl = np.array(c_lbl)
proj  = TSNE(n_components=2, init="pca", perplexity=30, random_state=0) \
        .fit_transform(Z)

def plot(col, ttl, fn):
    plt.figure(figsize=(5.5,5))
    plt.scatter(proj[:,0], proj[:,1], s=8, c=col, alpha=.85)
    plt.title(ttl)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(fn, dpi=180)
    plt.show()

pal_task  = sns.color_palette("tab10", n_colors=len(tasks))
pal_class = sns.color_palette("tab10", n_colors=C)

plot([pal_task[t]  for t in t_lbl],
     "Latent – colour = TASK",  "tsne_task.png")
plot([pal_class[c] for c in c_lbl],
     "Latent – colour = CLASS", "tsne_class.png")

print("Saved tsne_task.png / tsne_class.png")
