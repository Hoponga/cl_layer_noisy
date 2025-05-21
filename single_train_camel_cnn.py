
# ------------------------------------------------------------
# Continual (Permuted-MNIST) training with:
#   • EMA teacher
#   • per-class OT head + orthogonality
#   • Laplacian anchoring of prototypes
#   • Wasserstein distillation
#   • task×class balanced replay buffer
#
# `train_and_eval(cfg)`  —>  list[float]  (accuracy per task)
# ------------------------------------------------------------
import random, math, itertools, time, os, copy
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor

# ---------------- helper datasets ---------------------------
class PMNIST(Dataset):
    """Permuted MNIST: a *fixed* permutation per task."""
    def __init__(self, base, perm):
        self.base, self.perm = base, perm
    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x.view(-1)[self.perm], y
    def __len__(self): return len(self.base)

class SplitCIFAR10(Dataset):
    def __init__(self, original_dataset, classes):
        self.dataset = original_dataset
        self.classes = set(int(c) for c in classes)
        self.indices = [
            idx for idx, (_, label) in enumerate(self.dataset)
            if int(label) in self.classes
        ]
        if not self.indices:
            raise ValueError(f"No examples found for classes {classes}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img, label = self.dataset[self.indices[index]]
        return img, label
# ------------------------------------------------------------

# ------------ task × class balanced replay ------------------
class TaskClassReplayBuf:
    def __init__(self, cap, n_classes, device):
        self.cap, self.C, self.device = cap, n_classes, device
        self.store = {}                           # task → {cls: [x …]}
    # ----- helpers -----
    def _cell_cap(self):
        return max(1, self.cap // (len(self.store)*self.C)) if self.store else self.cap
    def _total(self):
        return sum(len(b) for t in self.store.values() for b in t.values())
    def _trim(self):
        cell_cap = self._cell_cap()
        while self._total() > self.cap:
            for t in list(self.store):
                for c in range(self.C):
                    bucket = self.store[t].get(c, [])
                    while len(bucket) > cell_cap:
                        bucket.pop(random.randrange(len(bucket)))
                    if self._total() <= self.cap:
                        return
    # ----- public API -----
    @torch.no_grad()
    def add(self, xs, ys, task_id):
        xs, ys = xs.cpu(), ys.cpu()
        if task_id not in self.store:
            self.store[task_id] = {c: [] for c in range(self.C)}
        cell_cap = self._cell_cap()
        for x, y in zip(xs, ys):
            cls = int(y)
            bucket = self.store[task_id][cls]
            if len(bucket) < cell_cap:
                bucket.append(x.clone())
            else:
                bucket[random.randrange(cell_cap)] = x.clone()
        self._trim()

    def sample(self, m):
        if self._total() == 0:
            return None, None
        xs, ys = [], []
        tasks = list(self.store)
        per_t  = max(1, m // len(tasks))
        for t in tasks:
            per_c = max(1, per_t // self.C)
            for c in range(self.C):
                bucket = self.store[t][c]
                if bucket:
                    k = min(per_c, len(bucket))
                    idx = random.sample(range(len(bucket)), k)
                    xs.extend(bucket[i] for i in idx)
                    ys.extend([c]*k)
        xs = torch.stack(xs).to(self.device)
        ys = torch.tensor(ys, device=self.device)
        return xs, ys
# ------------------------------------------------------------

# ---------------- OT-Head & network --------------------------
class OTHead(nn.Module):
    def __init__(self, d_tsk, K, tau):
        super().__init__()
        self.P   = nn.Parameter(torch.randn(K, d_tsk))
        self.tau = tau
        nn.init.normal_(self.P)
    def forward(self, z):
        Pn = F.normalize(self.P, dim=1)
        scr = z @ Pn.t() / self.tau
        Q   = F.softmax(scr, 1)
        cent = self.P[Q.argmax(1)]
        return cent

class Net(nn.Module):
    def __init__(self, d_cls, d_tsk, K, tau):
        super().__init__()

        inc = 3
        outc = 256
        self.cnn = nn.Sequential(
            # block 1
            nn.Conv2d(inc, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.05),

            # block 3
            nn.Conv2d(128, outc, kernel_size=3, padding=1),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        inp = 4096
        self.backbone = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(inp, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, d_cls + d_tsk)
        )

        # self.backbone = nn.Sequential(
        #     nn.Linear(784, 256), nn.ReLU(),
        #     nn.Linear(256, d_cls+d_tsk))
        self.head = OTHead(d_tsk, K, tau)
        self.fc   = nn.Linear(d_cls+d_tsk, 10)
        self.d_cls, self.d_tsk = d_cls, d_tsk
    def forward(self, x):
        z_imm = self.cnn(x)
        z_flat = z_imm.view(z_imm.size(0), -1)
        z = self.backbone(z_flat)

        zc = F.normalize(z[:, :self.d_cls], dim=1)
        zt = F.normalize(z[:, self.d_cls:], dim=1)
        cent = self.head(zt)
        return self.fc(torch.cat([zc, cent], 1))
# ------------------------------------------------------------

# ---------------- Laplacian util -----------------------------
def build_laplacian(P, sigma=0.5):
    d2 = torch.cdist(P, P).pow(2)
    W  = torch.exp(-d2/(2*sigma*sigma))
    D  = torch.diag(W.sum(1))
    return D-W
# ------------------------------------------------------------

# ---------------- Wasserstein distill ------------------------
@torch.no_grad()
def _cdf(p): return p.cumsum(1)
def wasserstein(p_new, p_old):
    return (_cdf(p_new) - _cdf(p_old)).abs().sum(1).mean()
# ------------------------------------------------------------

# -------------------- main entry -----------------------------
def train_and_eval(cfg: dict):
    """
    cfg keys (must provide):  K, D_TSK, LAM_ORTHO, LAM_M, LAM_WD, device
    returns list[10] accuracies
    """
    # ---------- hyper-params ----------
    device     = torch.device(cfg["device"])
    K          = cfg["K"]
    D_TSK      = cfg["D_TSK"]
    D_CLS      = 128
    BATCH      = 128
    EPOCHS_T   = 10
    T_tasks    = 10
    REPLAY_MB  = 128
    REPLAY_CAP = 600
    EMA_ALPHA  = 0.98
    TAU        = 0.5
    lam_o, lam_m, lam_wd = cfg["LAM_ORTHO"], cfg["LAM_M"], cfg["LAM_WD"]
    torch.manual_seed(0); random.seed(0)

    # ---------- build tasks ----------
    base = CIFAR10("./data", train=True, download=True, transform=ToTensor())
    temp = torch.randperm(10)
    perms  = [[temp[i], temp[i+1]] for i in range(0, len(temp)-1, 2)]
    tasks = [DataLoader(SplitCIFAR10(base, p), batch_size=BATCH,
                        shuffle=True, drop_last=True)
            for p in perms]

    # ---------- nets + optimiser ----------
    student = Net(D_CLS, D_TSK, K, TAU).to(device)
    teacher = copy.deepcopy(student).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    buf = TaskClassReplayBuf(REPLAY_CAP, 10, device)

    P_anchor = L_anchor = None

    # -------------- training loop --------------
    for t, loader in enumerate(tasks):
        student.train()
        for _ in range(EPOCHS_T):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)

                # main CE
                logit = student(xb)
                loss = F.cross_entropy(logit, yb)

                # orthogonality on prototypes
                Pn = F.normalize(student.head.P, dim=1)
                G  = Pn @ Pn.t()
                I  = torch.eye(K, device=device)
                loss += lam_o * (G-I).pow(2).mean()

                # replay
                xr, yr = buf.sample(REPLAY_MB)
                if xr is not None:
                    logT = teacher(xr).detach()
                    pT   = F.softmax(logT/TAU, 1)
                    logS = student(xr)
                    pS   = F.softmax(logS/TAU, 1)
                    loss += F.cross_entropy(logS, yr)
                    loss += lam_wd * wasserstein(pS, pT)

                    # Laplacian anchor
                    if P_anchor is not None:
                        Delta = teacher.head.P - P_anchor
                        loss += lam_m * torch.trace(Delta.t() @ L_anchor @ Delta)

                opt.zero_grad(); loss.backward(); opt.step()
                # EMA
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    pt.data.mul_(EMA_ALPHA).add_(ps.data, alpha=1-EMA_ALPHA)

                buf.add(xb.cpu(), yb.cpu(), t)

        # snapshot prototypes
        with torch.no_grad():
            P_anchor = teacher.head.P.detach().clone()
            L_anchor = build_laplacian(P_anchor).to(device)

    # -------------- evaluation --------------
    student.eval()
    acc = []
    with torch.no_grad():
        for s, loader in enumerate(tasks):
            corr=tot=0
            for xb,yb in loader:
                xb,yb = xb.to(device), yb.to(device)
                pred  = student(xb).argmax(1)
                corr += (pred== yb).sum().item()
                tot  += yb.size(0)
            acc.append(100*corr/tot)
    return acc
# ------------------------------------------------------------

if __name__ == "__main__":
    # demo run
    cfg = dict(K=10, D_TSK=32,
               LAM_ORTHO=0.05, LAM_M=0.5, LAM_WD=0.5,
               device="cuda" if torch.cuda.is_available() else "cpu")
    acc = train_and_eval(cfg)
    print("Acc per task:", ["%.2f"%a for a in acc],
          "\nAvg:", "%.2f"%(sum(acc)/len(acc)))
