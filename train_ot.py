# train_ot.py  ── continual Permuted-MNIST with OT-task codes
# -----------------------------------------------------------
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import random, math, numpy as np

# ---------- hyper-parameters ----------
T            = 5      # number of tasks (= permutations)
BATCH        = 128
D_CLS, D_TSK = 128, 64
K            = T      # one prototype per task (simple)
LR           = 1e-3

LAM_OT       = 1.0
LAM_DIV      = 1e-3
LAM_SMOOTH   = 1e-2
LAM_ORTH     = 1e-2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# -----------------------------------------------------------


# ---------- 0.  dataset ----------
class PermutedMNIST(Dataset):
    def __init__(self, base, perm):
        self.base, self.perm = base, perm
    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x.view(-1)[self.perm], y
    def __len__(self): return len(self.base)

base = MNIST('./data', train=True, download=True, transform=ToTensor())
perms  = [ torch.randperm(784) for _ in range(T) ]
tasks  = [ DataLoader(PermutedMNIST(base,p), batch_size=BATCH,
                      shuffle=True, drop_last=True)
           for p in perms ]


# ---------- 1.  balanced Sinkhorn ----------
def sinkhorn(scores, eps=0.1, n_iter=3):
    scores = scores - scores.max(dim=1, keepdim=True).values  # stabilise
    Q = torch.exp(scores / eps)          # [B,K]
    Q = Q / Q.sum()                      # total mass = 1
    r = torch.ones(Q.size(0), device=scores.device) / Q.size(0)
    c = torch.ones(Q.size(1), device=scores.device) / Q.size(1)
    for _ in range(n_iter):
        Q *= (r / (Q.sum(1)+1e-9)).unsqueeze(1)
        Q *= (c / (Q.sum(0)+1e-9)).unsqueeze(0)
    Q = Q / Q.sum()                      # final renorm
    return Q


# ---------- 2.  OT Task head ----------
class OTTaskHead(nn.Module):
    def __init__(self, d_task, K, delta=0.3):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(K, d_task))
        nn.init.normal_(self.prototypes)
        self.register_buffer('prev_proto',
                             F.normalize(self.prototypes.data.clone(), dim=1))
        self.delta = delta

    def forward(self, z_t, tau=0.5, eps=0.1, beta=0.95):
        # similarity & OT
        proto_n = F.normalize(self.prototypes, dim=1)
        scores  = z_t @ proto_n.T / tau                 # [B,K]
        with torch.no_grad():
            Q = sinkhorn(scores, eps=eps)               # [B,K]
            bs   = Q.sum(0) + 1e-9
            mean = (Q.T @ z_t) / bs.unsqueeze(1)
            self.prototypes.data.mul_(beta).add_((1-beta)*mean)
            self.prototypes.data = F.normalize(self.prototypes.data, dim=1)

        # diversity loss
        cos_mat = torch.matmul(proto_n, proto_n.T)      # [K,K]
        mask    = torch.triu(torch.ones_like(cos_mat), 1)
        div_loss = F.relu((cos_mat - self.delta) * mask).mean()

        # smoothness loss
        smooth_loss = (self.prototypes - self.prev_proto).pow(2).mean()
        self.prev_proto.copy_(self.prototypes.detach())

        # OT loss
        ot_loss = -(Q * F.log_softmax(scores, 1)).sum(1).mean()
        return ot_loss, div_loss, smooth_loss


# ---------- 3.  full model ----------
class OTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(inplace=True),
            nn.Linear(256, D_CLS + D_TSK))
        self.task_head  = OTTaskHead(D_TSK, K)
        self.classifier = nn.Linear(D_CLS + D_TSK, 10)

    def forward(self, x):
        z      = self.backbone(x)
        z_cls  = F.normalize(z[:, :D_CLS], dim=1)
        z_task = F.normalize(z[:, D_CLS:], dim=1)

        ot, div, sm = self.task_head(z_task)
        logits      = self.classifier(torch.cat([z_cls, z_task.detach()], 1))
        return logits, ot, div, sm


# ---------- 4.  helpers ----------
@torch.no_grad()
def accuracy(model, loader):
    model.eval(); c = n = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logit, *_ = model(x)
        c += (logit.argmax(1)==y).sum().item()
        n += y.size(0)
    return 100*c/n

def train_task(model, loader, opt, epochs=1):
    model.train()
    for _ in range(epochs):
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits, ot, div, sm= model(x)
            cls = F.cross_entropy(logits, y)
            loss = (cls
                    + LAM_OT * ot
                    + LAM_DIV * div
                    + LAM_SMOOTH * sm)
            opt.zero_grad(); loss.backward(); opt.step()


# ---------- 5.  main continual loop ----------
model = OTModel().to(device)
opt   = torch.optim.Adam(model.parameters(), lr=LR)

print("Starting continual training on", T, "permuted tasks")
for t, loader in enumerate(tasks):
    print(f"\n=== Task {t} ===")
    train_task(model, loader, opt, epochs=1)

    # evaluate on all tasks so far
    for s in range(t+1):
        acc = accuracy(model, tasks[s])
        print(f"  Eval task {s}: {acc:5.2f}%")


# --- t-SNE of z_task embeddings + prototypes (save as PNG) ---
import torch, numpy as np, matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D   # keep for 3-D option

device = next(model.parameters()).device

# 1. collect one batch per task
z_list, task_ids = [], []
model.eval();  D_TSK = z_task_size = model.task_head.prototypes.size(1)
with torch.no_grad():
    for tid, loader in enumerate(tasks):
        x,_ = next(iter(loader))
        z   = model.backbone(x.to(device))[:, -D_TSK:]      # take task slice
        z_list.append(z.cpu())
        task_ids.extend([tid]*z.size(0))

Z   = torch.cat(z_list, 0).numpy()
labs = np.array(task_ids)

# 2. get current prototypes
C = model.task_head.prototypes.detach().cpu().numpy()       # [K,d_t]

# 3. t-SNE
tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=0)
Z2d  = tsne.fit_transform(np.vstack([Z, C]))
Z_emb, C_emb = Z2d[:-C.shape[0]], Z2d[-C.shape[0]:]

# 4. plot
plt.figure(figsize=(7,6))
pal = plt.cm.get_cmap('tab10', len(tasks))
for tid in range(len(tasks)):
    sel = labs==tid
    plt.scatter(Z_emb[sel,0], Z_emb[sel,1],
                s=14, alpha=0.7, color=pal(tid), label=f"task {tid}")
plt.scatter(C_emb[:,0], C_emb[:,1],
            marker='X', s=120, c='black', label='prototypes')
plt.title("t-SNE of $z_{task}$ embeddings + prototypes")
plt.legend(bbox_to_anchor=(1.04,1), loc='upper left')
plt.tight_layout();  plt.savefig("z_task_tsne.png");  plt.show()