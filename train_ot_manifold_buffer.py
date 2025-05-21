# train_mean_teacher_anchor.py – MNIST / CIFAR10 CL with
#   ↳ EMA teacher        (no slow deepcopy)
#   ↳ per-class OT head
#   ↳ simple replay      (balanced exemplars)
#   ↳ Laplacian anchoring on teacher prototypes during replay
#   ↳ Wasserstein distillation on teacher logits

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# ─────────────── hyper-parameters ───────────────
T            = 10
BATCH        = 128
REPLAY_MB    = 128
REPLAY_CAP   = 4096           # total exemplars
D_CLS, D_TSK = 128, 32
LR           = 1e-3
EMA_ALPHA    = 0.98

LAM_ORTHO = 0.1
LAM_M        = 0.03     # Laplacian anchor weight
LAM_WD       = 0.1
TAU          = 0.5            # temperature for distillation
device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
isMNIST, PERMUTED = True, True
# ────────────────────────────────────────────────

# -------------- data loaders --------------------
class PMNIST(Dataset):
    def __init__(self, base, perm): self.base, self.perm = base, perm
    def __getitem__(self, i):
        x, y = self.base[i]
        return x.view(-1)[self.perm], y
    def __len__(self): return len(self.base)

class SplitMNIST(Dataset):
    def __init__(self, base, classes):
        self.data = [(x.view(-1), y) for x, y in base if y in classes]
    def __getitem__(self, i): return self.data[i]
    def __len__(self): return len(self.data)

def get_tasks():
    base = MNIST('./data', train=True, download=True, transform=ToTensor())
    if PERMUTED:
        perms = [torch.randperm(784) for _ in range(T)]
        return [DataLoader(PMNIST(base, p), batch_size=BATCH, shuffle=True, drop_last=True)
                for p in perms]
    else:
        splits = [list(range(i, i+2)) for i in range(0, 10, 2)]
        print(splits)
        return [DataLoader(SplitMNIST(base, c), batch_size=BATCH, shuffle=True, drop_last=True)
                for c in splits]

tasks = get_tasks()

# ---------------- replay buffer ------------------
class ReplayBuf:
    def __init__(self, cap):
        self.x, self.y, self.cap = [], [], cap
    def add(self, x, y):
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


class BalancedReplayBuf:
    """
    Class–balanced reservoir replay.
    • `cap` is the *total* number of exemplars the buffer may store.
    • When the buffer is not full it just grows.
    • Once full, every class that has appeared so far is allotted
        floor(cap / #seen_classes)   slots (±1 for rounding).
    • New items replace *random* items from their own class bucket
      (reservoir sampling) so the per–class distribution stays uniform.
    """
    def __init__(self, cap, device=torch.device("cpu")):
        self.cap     = cap
        self.device  = device         # for `.sample()`
        self.store   = {}             # cls → [tensor, …]

    # ---------- helpers ----------
    def _bucket_cap(self):
        """current max size of each class bucket"""
        return max(1, self.cap // len(self.store))

    def _total_size(self):
        return sum(len(v) for v in self.store.values())

    # ---------- public API ----------
    def add(self, xs, ys):
        """
        xs : Tensor [B, …]  (CPU or GPU)
        ys : Tensor [B]     (CPU or GPU)
        """
        for x, y in zip(xs.cpu(), ys.cpu()):
            c = int(y)

            # ensure bucket exists
            if c not in self.store:
                self.store[c] = []

            bucket = self.store[c]
            b_cap  = self._bucket_cap()

            if len(bucket) < b_cap:               # room inside bucket
                bucket.append(x.clone())
            else:                                 # bucket full → reservoir replace
                k = random.randrange(len(bucket))
                bucket[k] = x.clone()

            # global overflow possible *only* when a NEW class appeared:
            while self._total_size() > self.cap:
                # trim random items from buckets that exceed the new cap
                for cls, buf in list(self.store.items()):
                    b_cap = self._bucket_cap()
                    while len(buf) > b_cap:
                        buf.pop(random.randrange(len(buf)))
                    if self._total_size() <= self.cap:
                        break

    def sample(self, m):
        """
        Balanced sampling: try to draw ⌈m / #seen⌉ per class.
        """
        if self._total_size() == 0:
            return None, None

        per = max(1, m // len(self.store))
        xs, ys = [], []
        for cls, buf in self.store.items():
            if not buf:
                continue
            idxs = random.sample(range(len(buf)), min(per, len(buf)))
            xs.extend(buf[i] for i in idxs)
            ys.extend([cls] * len(idxs))

        if not xs:   # shouldn’t happen, but be safe
            return None, None

        xs = torch.stack(xs).to(self.device)
        ys = torch.tensor(ys, device=self.device)
        return xs, ys

buf = BalancedReplayBuf(REPLAY_CAP, device = device)

# ------------- mean-teacher EMA -----------------
def ema(student, teacher, alpha=EMA_ALPHA):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(alpha).add_(ps.data, alpha=1-alpha)

# ---------- Laplacian builder helper ------------
def build_laplacian(P_anchor, sigma=0.5):
    d2 = torch.cdist(P_anchor, P_anchor, p=2).pow(2)
    W  = torch.exp(-d2 / (2 * sigma**2))
    D  = torch.diag(W.sum(dim=1))
    return D - W

# --------- Wasserstein distillation -------------
def wasserstein_distill(p_new, p_old):
    cdf_new = p_new.cumsum(dim=1)
    cdf_old = p_old.cumsum(dim=1)
    return (cdf_new - cdf_old).abs().sum(dim=1).mean()

# ---------------- model components ---------------
class OTHead(nn.Module):
    def __init__(self, d_tsk, K):
        super().__init__()
        self.P = nn.Parameter(torch.randn(K, d_tsk))
        nn.init.normal_(self.P)
    def forward(self, z):
        Pn = F.normalize(self.P, p=2, dim=1)
        scr = z @ Pn.t() / TAU
        Q   = F.softmax(scr / TAU, dim=1)
        #ot  = -(Q * F.log_softmax(scr, dim=1)).sum(1).mean()
        idx = Q.argmax(1)
        cent = self.P[idx]
        return cent

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        inp = 784
        self.backbone = nn.Sequential(
            nn.Linear(inp, 256), nn.ReLU(),
            nn.Linear(256, D_CLS + D_TSK)
        )
        self.head = OTHead(D_TSK, K=10)
        self.fc   = nn.Linear(D_CLS + D_TSK, 10)
    def forward(self, x, replay_z=None):
        z = self.backbone(x)
        zc = F.normalize(z[:, :D_CLS], p=2, dim=1)
        zt = F.normalize(z[:, D_CLS:], p=2, dim=1)
        cent = self.head(zt if replay_z is None else replay_z)
        logits = self.fc(torch.cat([zc, cent], dim=1))
        return logits

# -------------------------------------------------

def orthogonality_loss(P, lam=LAM_ORTHO):
    # P shape: [K, d_tsk]
    Pn = F.normalize(P, p=2, dim=1)          # [K, d]
    G  = Pn @ Pn.t()                         # [K, K]
    I  = torch.eye(Pn.size(0), device=P.device)
    return lam * ((G - I).pow(2).sum() / Pn.size(0)**2)



student = Net().to(device)
teacher = Net().to(device)
teacher.load_state_dict(student.state_dict())
opt = torch.optim.Adam(student.parameters(), lr=LR)

centroid_history = []

# placeholders for anchor snapshot
P_anchor, L_anchor = None, None

# ------------------- helper ----------------------------------
def run_tsne(data, colors, title, fname):
    proj = TSNE(n_components=2,
                init='pca',
                perplexity=30,
                random_state=0).fit_transform(data)
    plt.figure(figsize=(5.5,5))
    plt.scatter(proj[:,0], proj[:,1], s=8, c=colors, alpha=0.85)
    plt.title(title); plt.xticks([]); plt.yticks([])
    plt.tight_layout(); plt.savefig(fname, dpi=180); plt.show()

for t, loader in enumerate(tasks):
    centroid_history.append(teacher.head.P.detach().cpu())
    print(f"\n=== Task {t} ===")
    student.train()

    # training epochs per task
    for _ in range(2):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # current-task loss
            logits= student(x, None)
            loss = F.cross_entropy(logits, y) 
            #print(f"CE loss: {loss}") 

            ortho = orthogonality_loss(student.head.P)
            #print(f"ortho loss: {ortho}")
            loss += ortho 

            # replay + OT-distill + Laplacian anchor
            xr, yr = buf.sample(REPLAY_MB)
            if xr is not None:
                # teacher soft targets
                with torch.no_grad():
                    logT = teacher(xr, None)
                    pT = F.softmax(logT / TAU, dim=1)
                # student replay outputs
                logS = student(xr, None)
                pS = F.softmax(logS / TAU, dim=1)
                # Wasserstein distillation
                wd = wasserstein_distill(pS, pT)
                #print(f"wd loss: {wd}")
                loss += F.cross_entropy(logS, yr) + LAM_WD* wd

                # Laplacian anchor on teacher prototypes
                if P_anchor is not None:
                    Delta = teacher.head.P - P_anchor
                    man = torch.trace(Delta.t() @ L_anchor @ Delta)
                    #print(f"manifold loss: {man}")
                    loss += LAM_M * man

                
            opt.zero_grad()
            loss.backward()
            opt.step()

            ema(student, teacher)
            buf.add(x.cpu(), y.cpu())

    # snapshot teacher prototypes and build Laplacian
    with torch.no_grad():
        P_anchor = teacher.head.P.detach().clone()
        L_anchor = build_laplacian(P_anchor).to(device)

    # evaluation
    student.eval()
    for s in range(t+1):
        correct = total = 0
        for xv, yv in tasks[s]:
            xv, yv = xv.to(device), yv.to(device)
            pred = student(xv, None)
            correct += (pred.argmax(1) == yv).sum().item()
            total   += yv.size(0)
        print(f" Task {s} acc: {100*correct/total:5.2f}%")



# =============================================================
# t-SNE visualisations for latent space and centroids
#   (a) z_task  coloured by TASK
#   (b) z_task  coloured by CLASS
#   (c) z_class coloured by TASK
#   (d) prototype centroids at the *start* of each task
# -------------------------------------------------------------

import torch, numpy as np, matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

device = next(student.parameters()).device
K       = 10                         # prototypes per class
n_vis   = 512                        # how many points per task to visualise
rng     = np.random.default_rng(0)

# store centroids snapped at the *beginning* of each task
centroid_history = []   # list of tensors [K,d_tsk]



# =============================================================
# (After training)  gather one minibatch per task
# =============================================================
Z_task, Z_class, task_lbls, cls_lbls = [], [], [], []

student.eval()
with torch.no_grad():
    for tid, loader in enumerate(tasks):
        x_all, y_all = next(iter(loader))
        # subsample to at most n_vis points for faster t-SNE
        idx = rng.choice(len(x_all), size=min(n_vis, len(x_all)), replace=False)
        x, y = x_all[idx].to(device), y_all[idx].to(device)

        z = student.backbone(x)                     # [B, d_cls+d_tsk]
        Zc = F.normalize(z[:, :D_CLS], dim=1).cpu() # [B, d_cls]
        Zt = F.normalize(z[:, D_CLS:], dim=1).cpu() # [B, d_tsk]

        Z_class.append(Zc)
        Z_task.append(Zt)
        task_lbls.extend([tid]*len(x))
        cls_lbls.extend(y.cpu().tolist())

# concat
Z_class = torch.cat(Z_class, 0).numpy()
Z_task  = torch.cat(Z_task , 0).numpy()
task_lbls = np.array(task_lbls)
cls_lbls  = np.array(cls_lbls)

# ------------------ a) z_task coloured by TASK ---------------
colors_a = [sns.color_palette('tab10')[t] for t in task_lbls]
run_tsne(Z_task, colors_a, "z_task   (color = TASK)", "tsne_z_task_by_task.png")

# ------------------ b) z_task coloured by CLASS --------------
colors_b = [sns.color_palette('tab10')[c] for c in cls_lbls]
run_tsne(Z_task, colors_b, "z_task   (color = CLASS)", "tsne_z_task_by_class.png")

# ------------------ c) z_class coloured by TASK --------------
colors_c = [sns.color_palette('tab10')[t] for t in task_lbls]
run_tsne(Z_class, colors_c, "z_class  (color = TASK)", "tsne_z_class_by_task.png")

# =============================================================
# (d)  centroid trajectory: prototypes at the START of each task
# =============================================================
#  → During training loop add:
#     centroid_history.append(P_anchor.cpu())     (right after snapshot)
#
# Here we visualise all stored centroids:

if centroid_history:     # make sure you collected them
    all_centroids = torch.stack(centroid_history)   # [T, K, d_tsk]
    C_flat  = all_centroids.view(-1, all_centroids.size(-1)).numpy()
    task_id = np.repeat(np.arange(len(centroid_history)), K)
    colors_d = [sns.color_palette('tab10')[t] for t in task_id]
    run_tsne(C_flat, colors_d,
             "Prototypes (at task start)  color = TASK",
             "tsne_prototypes_by_task.png")
else:
    print("centroid_history is empty – did you append snapshots during training?")
