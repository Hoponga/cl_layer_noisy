# train_ot_centroid.py  –  Permuted-MNIST with OT + replay + centroid input
# ------------------------------------------------------------
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Grayscale
import random, numpy as np

# ---------------- hyper-parameters ----------------
T             = 5          # tasks / permutations
BATCH         = 128
REPLAY_MB     = 128 #512
REPLAY_CAP    = 4096
D_CLS, D_TSK  = 128, 32
K             = T          # one prototype per task
LR            = 1e-3
LAM_OT        = 2.0
LAM_DIV       = 3e-3
LAM_ORTH      = 1e-2
LAM_DRIFT     = 1e-2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --------------------------------------------------

# ---------------- dataset -------------------------
class PermutedMNIST(Dataset):
    def __init__(self, base, perm): self.base, self.perm = base, perm
    def __getitem__(self, idx):
        x,y = self.base[idx];  return x.view(-1)[self.perm], y
    def __len__(self): return len(self.base)

class PermutedCIFAR10(Dataset):
    def __init__(self, base, perm): self.base, self.perm = base, perm
    def __getitem__(self, idx):
        x,y = self.base[idx];  return x.view(-1)[self.perm], y
    def __len__(self): return len(self.base)

class SplitMNIST(Dataset):
    """Like PermutedMNIST but only keeps a subset of classes, returns labels 0–9."""
    def __init__(self, original_dataset, classes):
        """
        original_dataset: torchvision.datasets.MNIST instance
        classes: iterable of digit labels to include, e.g. [0,1]
        """
        self.dataset = original_dataset
        # ensure classes is a Python set of ints
        self.classes = set(int(c) for c in classes)
        # collect indices whose labels are in `classes`
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
        # flatten exactly like PermutedMNIST
        flat = img.view(-1)       # → Tensor of shape [784]
        return flat, label

class SplitCIFAR10(Dataset):
    """Like PermutedMNIST but only keeps a subset of classes, returns labels 0–9."""
    def __init__(self, original_dataset, classes):
        """
        original_dataset: torchvision.datasets.MNIST instance
        classes: iterable of digit labels to include, e.g. [0,1]
        """
        self.dataset = original_dataset
        # ensure classes is a Python set of ints
        self.classes = set(int(c) for c in classes)
        # collect indices whose labels are in `classes`
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
        # flatten exactly like PermutedMNIST
        flat = img.view(-1)       # → Tensor of shape [784]
        return flat, label

PERMUTED = False
isMNIST = True

base = MNIST('./data', train=True, download=True, transform=ToTensor())

transform = Compose([
    Grayscale(num_output_channels=1),  # Convert RGB to grayscale
    ToTensor()
])
cifar_base = CIFAR10(root='./data', train=True, download=True, transform=transform)

if isMNIST:
    if PERMUTED:
        perms  = [torch.randperm(784) for _ in range(T)]
        tasks  = [DataLoader(PermutedMNIST(base,p), batch_size=BATCH,
                            shuffle=True, drop_last=True) for p in perms]
    else:
        temp = torch.randperm(10)
        perms  = [[temp[i], temp[i+1]] for i in range(0, len(temp)-1, 2)]
        tasks  = [DataLoader(SplitMNIST(base,p), batch_size=BATCH,
                            shuffle=True, drop_last=True) for p in perms]
else:
    if PERMUTED:
        perms  = [torch.randperm(1024) for _ in range(T)]
        tasks  = [DataLoader(PermutedCIFAR10(cifar_base,p), batch_size=BATCH,
                            shuffle=True, drop_last=True) for p in perms]
    else:
        temp = torch.randperm(10)
        perms  = [[temp[i], temp[i+1]] for i in range(0, len(temp)-1, 2)]
        tasks  = [DataLoader(SplitCIFAR10(cifar_base,p), batch_size=BATCH,
                                shuffle=True, drop_last=True) for p in perms]

# ---------------- replay buffer -------------------
class ReplayBuf:
    def __init__(self, cap):
        self.x, self.y, self.cap = [], [], cap
    def add_batch(self, x, y):
        for xi,yi in zip(x.cpu(), y.cpu()):
            if len(self.x) < self.cap:
                self.x.append(xi); self.y.append(int(yi))
            else:
                k = random.randrange(self.cap)
                self.x[k] = xi;    self.y[k] = int(yi)
    def sample(self, m):
        if not self.x: return None, None
        idx = random.sample(range(len(self.x)), min(m, len(self.x)))
        xs = torch.stack([self.x[i] for i in idx]).to(device)
        ys = torch.tensor([self.y[i] for i in idx]).to(device)
        return xs, ys
replay = ReplayBuf(REPLAY_CAP)


# ---------------- OT prototype head ---------------
class OTHead(nn.Module):
    def __init__(self, d_t, K, delta=0.3):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(K, d_t))
        nn.init.normal_(self.prototypes)
        self.delta = delta
    # def forward(self, z_t, replay_z=None, *, tau=0.5, eps=0.1, alpha=0.7, beta=0.9):
    #     proto_n = F.normalize(self.prototypes, dim=1)             # [K,d]
    #     scores  = z_t @ proto_n.T / tau                            # [B,K]
    #     proto_loss = 0
    #     with torch.no_grad():
    #         Q = F.softmax(scores / eps, dim=1)                   # [B,K]
    #         bs = Q.sum(0) + 1e-9
    #         # mean = (mask.T @ z_t) / (mask.sum(0) + 1e-9).unsqueeze(1)
    #         mean = (Q.T @ z_t) / bs.unsqueeze(1)                   # [K,d]

    #         if replay_z is not None:
    #             replay_scores = replay_z @ proto_n.T / tau
    #             Qr = F.softmax(replay_scores / eps, dim=1)
    #             bs_r = Qr.sum(0) + 1e-9
    #             mean_r = (Qr.T @ replay_z) / bs_r.unsqueeze(1)
    #             mean = alpha * mean + (1 - alpha) * mean_r         # weighted combination

    #         pbefore = self.prototypes.data.clone()
    #         self.prototypes.data.mul_(beta).add_((1 - beta) * mean)
    #         self.prototypes.data = F.normalize(self.prototypes.data, dim=1)
    #         proto_loss += F.mse_loss(self.prototypes.data, pbefore)

    #     # losses
    #     ot_loss  = -(Q * F.log_softmax(scores, dim=1)).sum(1).mean()
    #     cos_m    = torch.matmul(proto_n, proto_n.T)
    #     div_loss = F.relu((cos_m - self.delta) *
    #                     torch.triu(torch.ones_like(cos_m), diagonal=1)).mean()
    #     idx = Q.argmax(1)
    #     hard_centroid = self.prototypes[idx]                       # [B,d]

    #     return ot_loss, div_loss, proto_loss, hard_centroid
    def forward(self, z_t, replay_z=None, *, tau=0.5, eps=0.1, alpha=0.5, beta=0.8):
        proto_n = F.normalize(self.prototypes, dim=1)          # [K,d]
        scores  = z_t @ proto_n.T / tau
        pbefore = self.prototypes.data.clone()                       # [B,K]
        with torch.no_grad():
            sim = F.cosine_similarity(z_t.unsqueeze(1), proto_n.unsqueeze(0), dim=2) #sinkhorn(scores, eps)    
            Q = F.softmax(sim / tau, dim=1)                   # [B,K]
            bs = Q.sum(0)+1e-9
            mean = (Q.T @ z_t) / bs.unsqueeze(1)
            if replay_z is not None:
                sim_r = F.cosine_similarity(replay_z.unsqueeze(1), proto_n.unsqueeze(0), dim=2)
                Qr   = F.softmax(sim_r / tau, dim=1) # sinkhorn(replay_z @ proto_n.T / tau, eps)
                bs_r = Qr.sum(0)+1e-9
                mean_r = (Qr.T @ replay_z)/bs_r.unsqueeze(1)
                mean   = alpha*mean + (1-alpha)*mean_r
            self.prototypes.data.mul_(beta).add_((1-beta)*mean)
            self.prototypes.data = F.normalize(self.prototypes.data, dim=1)
        proto_loss = F.mse_loss(pbefore, self.prototypes.data)
        # losses
        ot_loss  = -(Q * F.log_softmax(scores,1)).sum(1).mean()
        cos_m    = torch.matmul(proto_n, proto_n.T)
        div_loss = F.relu((cos_m-self.delta)
                          * torch.triu(torch.ones_like(cos_m),1)).mean()
        idx = Q.argmax(1)                             # [B]
        hard_centroid= self.prototypes[idx]                    # [B,d]

        return ot_loss, div_loss, hard_centroid, proto_loss

# ---------------- full model ----------------------
class OTModel(nn.Module):
    def __init__(self):
        super().__init__()
        #
        if isMNIST:
            self.backbone = nn.Sequential(
                nn.Linear(784,256), nn.ReLU(inplace=True),
                nn.Linear(256, D_CLS+D_TSK))
        else:
            self.backbone = nn.Sequential(
                nn.Linear(1024,256), nn.ReLU(inplace=True),
                nn.Linear(256, D_CLS+D_TSK))
        self.task_head = OTHead(D_TSK, K)
        self.classifier= nn.Linear(D_CLS+D_TSK, 10)
    def forward(self, x, replay_z=None):
        z       = self.backbone(x)
        z_cls   = F.normalize(z[:,:D_CLS], dim=1)
        z_task  = F.normalize(z[:, D_CLS:], dim=1)
        ot, div, cent, proto = self.task_head(z_task, replay_z)
        logits  = self.classifier(torch.cat([z_cls, cent.detach()], 1))
        return logits, ot, div, proto

# ---------------- accuracy helper -----------------
@torch.no_grad()
def accuracy(model, loader):
    model.eval(); correct=n=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        log,_,_,_ = model(x)
        correct += (log.argmax(1)==y).sum().item();  n += y.size(0)
    return 100*correct/n

replay_buffer_freq = 1

# ---------------- training loop -------------------
def train_task(model, loader, optim, epochs=1):
    model.train()
    for _ in range(epochs):
        for i, (x,y)  in enumerate(loader):
            x,y = x.to(device), y.to(device)

            # replay embeddings (no grad)
            xr,yr = replay.sample(REPLAY_MB)
            if xr is not None:
                with torch.no_grad():
                    z_r = model.backbone(xr)[:, -D_TSK:]
                    z_r = F.normalize(z_r, dim=1)
            else:
                z_r = None

            log, ot, div, proto = model(x, replay_z=z_r)
            cls = F.cross_entropy(log, y)
            loss = (cls + LAM_OT*ot + LAM_DIV*div + LAM_DRIFT*proto)
            # ADDED
            if xr is not None and i % replay_buffer_freq == 0:
                logits_r, ot_r, div_r, proto_r = model(xr, replay_z=None)
                loss += (F.cross_entropy(logits_r, yr) + LAM_OT*ot_r + LAM_DIV*div_r + LAM_DRIFT*proto_r)

            optim.zero_grad(); loss.backward(); optim.step()
            replay.add_batch(x.detach(), y.detach())

# ---------------- main continual loop -------------
model = OTModel().to(device)
optim = torch.optim.Adam(model.parameters(), lr=LR)

print("Training with centroid classifier and replay buffer\n")
for t, loader in enumerate(tasks):
    print(f"=== Task {t} ===")
    if not PERMUTED:
        print(perms[t])
    train_task(model, loader, optim)

    for s in range(t+1):
        if PERMUTED:
            print(f"  Eval task {s}: {accuracy(model, tasks[s]):5.2f}%")
        else:
            print(f"  Eval task {s}, {perms[s]}: {accuracy(model, tasks[s]):5.2f}%")


# -----------------------------------------------------------
# t‑SNE diagnostics: three plots
#  1. z_task  coloured by CLASS
#  2. z_class coloured by TASK
#  3. concat   coloured by CLASS, lightness = TASK
# -----------------------------------------------------------
import torch, numpy as np, matplotlib.pyplot as plt, colorsys
from sklearn.manifold import TSNE

device = next(model.parameters()).device
D_CLS = model.classifier.in_features - model.task_head.prototypes.size(1)
D_TSK = model.task_head.prototypes.size(1)
T     = len(tasks)

# ==== 1.  gather embeddings ====
z_cls_list, z_tsk_list, cls_lbls, task_lbls = [], [], [], []
model.eval(); 
with torch.no_grad():
    for tid, loader in enumerate(tasks):
        X, Y = next(iter(loader))               # one batch per task
        X = X.to(device)
        Z   = model.backbone(X)                 # [B, D_CLS + D_TSK]
        z_cls_list.append( F.normalize(Z[:,:D_CLS], dim=1).cpu() )
        z_tsk_list.append( F.normalize(Z[:,D_CLS:], dim=1).cpu() )
        cls_lbls.extend(Y.tolist())
        task_lbls.extend([tid]*Y.size(0))

Z_C   = torch.cat(z_cls_list,0).numpy()
Z_T   = torch.cat(z_tsk_list,0).numpy()
Z_ALL = np.concatenate([Z_C, Z_T],1)
classes = np.array(cls_lbls)
tasks_id= np.array(task_lbls)

# ---------------- helper to run TSNE --------------------
def run_tsne(data, title, filename, colors):
    tsne = TSNE(n_components=2, init='pca', perplexity=30, random_state=0)
    proj = tsne.fit_transform(data)
    plt.figure(figsize=(6,5))
    plt.scatter(proj[:,0], proj[:,1], s=12, c=colors, alpha=0.8)
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout(); plt.savefig(filename); plt.show()

# ==== 1. z_task coloured by CLASS ========================
pal = plt.cm.get_cmap('tab10', 10)
color1 = [pal(c) for c in classes]
run_tsne(Z_T, "z_task  (colour = CLASS)", "tsne_z_task_by_class.png", color1)

# ==== 2. z_class coloured by TASK ========================
palT = plt.cm.get_cmap('tab10', T)
color2 = [palT(t) for t in tasks_id]
run_tsne(Z_C, "z_class (colour = TASK)", "tsne_z_class_by_task.png", color2)

# ==== 3. concat: base colour = CLASS, lightness = TASK ===
base = [pal(c) for c in classes]     # RGBA base per class
def shade(rgba, tid):
    r,g,b,_ = rgba
    h,l,s   = colorsys.rgb_to_hls(r,g,b)
    # earlier task 0 => light (l=0.85), latest => dark (l=0.35)
    l_new   = 0.85 - 0.5*(tid/(T-1))
    r2,g2,b2 = colorsys.hls_to_rgb(h, l_new, s)
    return (r2,g2,b2,0.9)
color3 = [shade(p, t) for p,t in zip(base, tasks_id)]

run_tsne(Z_ALL, "concat z (hue=CLASS, shade=TASK)", 
         "tsne_concat_class_hue_task_shade.png", color3)
