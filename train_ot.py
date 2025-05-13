# train_ot_centroid.py  –  Permuted-MNIST with OT + replay + centroid input
# ------------------------------------------------------------
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import random, numpy as np

# ---------------- hyper-parameters ----------------
T             = 5          # tasks / permutations
BATCH         = 128
REPLAY_MB     = 64
REPLAY_CAP    = 800
D_CLS, D_TSK  = 128, 10
K             = T          # one prototype per task
LR            = 1e-3
LAM_OT        = 2.0
LAM_DIV       = 3e-3
LAM_ORTH      = 1e-2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --------------------------------------------------

# ---------------- dataset -------------------------
class PermutedMNIST(Dataset):
    def __init__(self, base, perm): self.base, self.perm = base, perm
    def __getitem__(self, idx):
        x,y = self.base[idx];  return x.view(-1)[self.perm], y
    def __len__(self): return len(self.base)

base   = MNIST('./data', train=True, download=True, transform=ToTensor())
perms  = [torch.randperm(784) for _ in range(T)]
tasks  = [DataLoader(PermutedMNIST(base,p), batch_size=BATCH,
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

# ---------------- Sinkhorn ------------------------
def sinkhorn(scores, eps=0.1, iters=3):
    scores = scores - scores.max(1, keepdim=True).values
    Q = torch.exp(scores / eps); Q = Q / Q.sum()
    r = torch.ones(Q.size(0), device=scores.device) / Q.size(0)
    c = torch.ones(Q.size(1), device=scores.device) / Q.size(1)
    for _ in range(iters):
        Q *= (r / (Q.sum(1)+1e-9)).unsqueeze(1)
        Q *= (c / (Q.sum(0)+1e-9)).unsqueeze(0)
    return Q / Q.sum()

# ---------------- OT prototype head ---------------
class OTHead(nn.Module):
    def __init__(self, d_t, K, delta=0.3):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(K, d_t))
        nn.init.normal_(self.prototypes)
        self.delta = delta
    def forward(self, z_t, replay_z=None, *, tau=0.5, eps=0.1, beta=0.9):
        proto_n = F.normalize(self.prototypes, dim=1)          # [K,d]
        scores  = z_t @ proto_n.T / tau                        # [B,K]
        with torch.no_grad():
            Q    = sinkhorn(scores, eps)                       # [B,K]
            bs   = Q.sum(0)+1e-9
            mean = (Q.T @ z_t) / bs.unsqueeze(1)
            if replay_z is not None:
                Qr   = sinkhorn(replay_z @ proto_n.T / tau, eps)
                bs_r = Qr.sum(0)+1e-9
                mean_r = (Qr.T @ replay_z)/bs_r.unsqueeze(1)
                mean   = 0.5*mean + 0.5*mean_r
            self.prototypes.data.mul_(beta).add_((1-beta)*mean)
            self.prototypes.data = F.normalize(self.prototypes.data, dim=1)
        # losses
        ot_loss  = -(Q * F.log_softmax(scores,1)).sum(1).mean()
        cos_m    = torch.matmul(proto_n, proto_n.T)
        div_loss = F.relu((cos_m-self.delta)
                          * torch.triu(torch.ones_like(cos_m),1)).mean()
        idx          = Q.argmax(1)                             # [B]
        hard_centroid= self.prototypes[idx]                    # [B,d]

        return ot_loss, div_loss, hard_centroid

# ---------------- full model ----------------------
class OTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(784,256), nn.ReLU(inplace=True),
            nn.Linear(256, D_CLS+D_TSK))
        self.task_head = OTHead(D_TSK, K)
        self.classifier= nn.Linear(D_CLS+D_TSK, 10)
    def forward(self, x, replay_z=None):
        z       = self.backbone(x)
        z_cls   = F.normalize(z[:,:D_CLS], dim=1)
        z_task  = F.normalize(z[:, D_CLS:], dim=1)
        ot, div, cent = self.task_head(z_task, replay_z)
        logits  = self.classifier(torch.cat([z_cls, cent.detach()], 1))
        return logits, ot, div

# ---------------- accuracy helper -----------------
@torch.no_grad()
def accuracy(model, loader):
    model.eval(); correct=n=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        log,_ ,_, = model(x)
        correct += (log.argmax(1)==y).sum().item();  n += y.size(0)
    return 100*correct/n

# ---------------- training loop -------------------
def train_task(model, loader, optim, epochs=1):
    model.train()
    for _ in range(epochs):
        for x,y in loader:
            x,y = x.to(device), y.to(device)

            # replay embeddings (no grad)
            xr,_ = replay.sample(REPLAY_MB)
            if xr is not None:
                with torch.no_grad():
                    z_r = model.backbone(xr)[:, -D_TSK:]
                    z_r = F.normalize(z_r, dim=1)
            else:
                z_r = None

            log, ot, div = model(x, replay_z=z_r)
            cls = F.cross_entropy(log, y)
            loss = (cls + LAM_OT*ot + LAM_DIV*div)

            optim.zero_grad(); loss.backward(); optim.step()
            replay.add_batch(x.detach(), y.detach())

# ---------------- main continual loop -------------
model = OTModel().to(device)
optim = torch.optim.Adam(model.parameters(), lr=LR)

print("Training with centroid classifier and replay buffer\n")
for t, loader in enumerate(tasks):
    print(f"=== Task {t} ===")
    train_task(model, loader, optim)

    for s in range(t+1):
        print(f"  Eval task {s}: {accuracy(model, tasks[s]):5.2f}%")


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
