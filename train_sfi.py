import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.distributed as dist
import os 

import copy 


import random
import torch.nn.functional as F

class ReplayBuffer:
    """A simple ring buffer for replay."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.storage  = []
        self.ptr      = 0

    def add(self, X, y):
        # store on CPU
        X_cpu = X.detach().cpu()
        y_cpu = y.detach().cpu()
        batch = list(zip(X_cpu, y_cpu))
        for x_i, y_i in batch:
            if len(self.storage) < self.capacity:
                self.storage.append((x_i, y_i))
            else:
                self.storage[self.ptr] = (x_i, y_i)
                self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.storage, min(batch_size, len(self.storage)))
        Xr, yr = zip(*batch)
        return torch.stack(Xr), torch.tensor(yr)


class SFILayer(nn.Module):
    def __init__(self, input_dim, task_rep_dim=32):
        super().__init__()
        self.task_encoder = nn.Sequential(
            nn.Linear(task_rep_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * input_dim)
        )

    def forward(self, x, task_rep):
        # x: [B, H], task_rep: [B, TR]
        B, H = x.size()
        params = self.task_encoder(task_rep)    # [B, 3*H]
        mu, log_sigma, alpha = params.split(H, dim=1)  # each [B, H]
        sigma = log_sigma.exp()                         # [B, H]
        eps   = torch.randn_like(sigma)                 # [B, H]
        noise = mu + sigma * eps                        # [B, H]
        
        out = x + alpha * noise         # gate the noise
        
        # detach mu/sigma so that deepcopy won’t choke on graph leaves
        return out, alpha, mu.detach(), sigma.detach()

class SFI_MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, num_classes=10, task_rep_dim=64):
        super().__init__()
        # Main MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rep_proj   = nn.Linear(hidden_dim, task_rep_dim)
        self.sfi1 = SFILayer(hidden_dim, task_rep_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.sfi2 = SFILayer(hidden_dim, task_rep_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Task representation: pooled features from fc1
        self.task_rep_dim = task_rep_dim
        self.pool = nn.AdaptiveAvgPool1d(task_rep_dim)  # Reduces fc1 output to task_rep_dim
    
    def forward_features(self, x):
        h = torch.relu(self.fc1(x))           # [B, H]
        r = torch.relu(self.rep_proj(h))      # [B, TR]
        return r
    
    def forward(self, x):

        B = x.size(0) # batch size 


        # We could have a per sample representation 
        # Alternatively, under the assumption that every batch is a constant task, we can take the average over task representations of the batch 
        # and broadcast this back to each sample 
        per_rep = self.forward_features(x)   # [B, TR]
        task_rep = per_rep
        #task_rep = per_rep.mean(dim=0, keepdim=True) # [1, TR]
        #task_rep = task_rep.expand(B, -1)            # [B, TR]
        
        # first hidden → SFI
        h1, a1, m1, s1 = self.sfi1(torch.relu(self.fc1(x)), task_rep)
        h1 = torch.relu(h1)
        
        # second hidden → SFI
        h2, a2, m2, s2 = self.sfi2(torch.relu(self.fc2(h1)), task_rep)
        h2 = torch.relu(h2)
        
        # final logits
        logits = self.classifier(h2)
        
        # return everything we need for the losses
        return logits, (a1, a2), (m1, m2), (s1, s2)
# Hyperparameters: lambda_1, lambda_2, lambda_3, snapshot_freq
# Initialize: model, task_encoder, optimizer
# Memory: buffer for frozen snapshots (e.g., FIFO queue)

def sliced_wasserstein(h_cur, h_froz, n_projs=10):
    # h_cur, h_froz: [B, D]
    D = h_cur.size(1)
    loss = 0
    for _ in range(n_projs):
        # random unit direction
        u = torch.randn(D, device=h_cur.device)
        u /= u.norm(p=2)
        # 1-D projections
        p_cur  = h_cur @ u          # [B]
        p_froz = h_froz @ u
        # sort and compare
        loss += torch.mean(torch.abs(
            torch.sort(p_cur, 0)[0] -
            torch.sort(p_froz,0)[0]
        ))
    return loss / n_projs
class PermutedMNIST(Dataset):
    """Creates a permuted MNIST task with a fixed pixel permutation."""
    def __init__(self, original_dataset, permutation):
        self.dataset = original_dataset
        self.permutation = permutation
    
    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img.view(-1)[self.permutation], label
    
    def __len__(self):
        return len(self.dataset)





def train_sfi(
    model, tasks, device,
    lambda_anchor=1.0, lambda_kl=0, lambda_sp=0.01,
    lambda_rep=0.5, lambda_distill=0,
    lambda_div=1, margin=10.0,
    replay_capacity=2000, replay_batch=64,
    max_snapshots=8
):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    buffer = ReplayBuffer(replay_capacity)
    frozen_snapshots = []
    noise_protos = []
    teacher = None

    for task_id, loader in enumerate(tasks):
        print(f"\n=== TRAINING TASK {task_id} ===")
        mu_prev = sigma_prev = None

        for X, y in loader:
            X, y = X.to(device), y.to(device)

            # Frozen task rep for distillation
            if teacher is not None:
                teacher.eval()
            with torch.no_grad():
                task_rep = model.forward_features(X)

            # Forward pass
            logits, (a1, a2), (m1, m2), (s1, s2) = model(X)

            # 1) Cross-entropy on current task
            loss_ce = F.cross_entropy(logits, y)

            # 2) Manifold anchoring (MSE) against frozen snapshots
            loss_anchor = 0.0
            for snap in frozen_snapshots:
                with torch.no_grad():
                    h_froz = snap.forward_features(X)
                h_cur = model.forward_features(X)
                loss_anchor += F.mse_loss(h_cur, h_froz)
            loss_anchor /= max(1, len(frozen_snapshots))

            # 3) KL divergence on both SFI layers
            if mu_prev is not None:
                kl1 = (torch.log(sigma_prev[0]/s1) +
                       (s1**2 + (mu_prev[0]-m1)**2)/(2*sigma_prev[0]**2) -
                       0.5).mean()
                kl2 = (torch.log(sigma_prev[1]/s2) +
                       (s2**2 + (mu_prev[1]-m2)**2)/(2*sigma_prev[1]**2) -
                       0.5).mean()
                loss_kl = 0.5 * (kl1 + kl2)
            else:
                loss_kl = torch.tensor(0.0, device=device)

            # 4) Sparsity on alpha
            loss_sp = a1.abs().mean() + a2.abs().mean()

            # 5) Experience replay
            loss_rep = torch.tensor(0.0, device=device)
            if len(buffer.storage) >= replay_batch:
                Xr_cpu, yr_cpu = buffer.sample(replay_batch)
                Xr, yr = Xr_cpu.to(device), yr_cpu.to(device)
                logits_r, *_ = model(Xr)
                loss_rep = F.cross_entropy(logits_r, yr)

            # 6) Distillation on replay
            loss_distill = torch.tensor(0.0, device=device)
            if teacher is not None and len(buffer.storage) >= replay_batch:
                with torch.no_grad():
                    t_logits, *_ = teacher(Xr)
                log_p = F.log_softmax(logits_r / 2.0, dim=1)
                q = F.softmax(t_logits / 2.0, dim=1)
                loss_distill = F.kl_div(log_p, q, reduction='batchmean') * (2.0**2)

            # 7) Repulsion between noise prototypes
            mu1_cur = m1.detach().mean(dim=0)
            sig1_cur = s1.detach().mean(dim=0)
            mu2_cur = m2.detach().mean(dim=0)
            sig2_cur = s2.detach().mean(dim=0)
            loss_div = torch.tensor(0.0, device=device)
            if noise_protos:
                for (mu1_p, sig1_p, mu2_p, sig2_p) in noise_protos:
                    #print(mu1_p, sig1_p, mu2_p, sig2_p)
                    d_mu1 = torch.norm(mu1_cur - mu1_p, p=2)
                    d_s1  = torch.norm(sig1_cur - sig1_p, p=2)
                    d_mu2 = torch.norm(mu2_cur - mu2_p, p=2)
                    d_s2  = torch.norm(sig2_cur - sig2_p, p=2)
                    loss_div += F.relu(margin - d_mu1)
                    loss_div += F.relu(margin - d_s1)
                    loss_div += F.relu(margin - d_mu2)
                    loss_div += F.relu(margin - d_s2)
                loss_div = loss_div / (4 * len(noise_protos))

            # Total loss
            total_loss = (
                loss_ce
                + lambda_anchor * loss_anchor
                + lambda_kl     * loss_kl
                + lambda_sp     * loss_sp
                + lambda_rep    * loss_rep
                + lambda_distill* loss_distill
                + lambda_div    * loss_div
            )

            # Backprop and step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update buffer and previous noise
            buffer.add(X, y)
            mu_prev = (m1.detach(), m2.detach())
            sigma_prev = (s1.detach(), s2.detach())

        # End of task: snapshot model and record noise prototype
        snap = copy.deepcopy(model).eval()
        frozen_snapshots.append(snap)
        if len(frozen_snapshots) > max_snapshots:
            frozen_snapshots.pop(0)

        # Compute and store task prototype
        mu1_p = mu_prev[0].mean(dim=0)
        sig1_p = sigma_prev[0].mean(dim=0)
        mu2_p = mu_prev[1].mean(dim=0)
        sig2_p = sigma_prev[1].mean(dim=0)
        noise_protos.append((mu1_p, sig1_p, mu2_p, sig2_p))
        if len(noise_protos) > max_snapshots:
            noise_protos.pop(0)

        # Update teacher for next task
        teacher = snap

    return model

def evaluate_sfi(model, tasks, device, num_batches=2):
    model.eval()
    with torch.no_grad():
        for tid, loader in enumerate(tasks):
            it = iter(loader)
            acc = 0
            for b in range(num_batches):
                try:
                    X, y = next(it)
                except StopIteration:
                    break
                X, y = X.to(device), y.to(device)
                logits, *_ = model(X)
                acc += (logits.argmax(1) == y).float().mean().item()
            print(f"[Task {tid}]  → {acc/num_batches*100:5.2f}%")
    model.train()




def run_trial(params, tasks, device):
    torch.manual_seed(0)
    model = SFI_MLP(task_rep_dim=64).to(device)
    # train briefly
    train_sfi(model, tasks, device,
              lambda_anchor=params['lambda_anchor'],
              lambda_kl=params['lambda_kl'],
              lambda_sp=params['lambda_sp'],
              lambda_rep=params['lambda_rep'],
              lambda_distill=params['lambda_distill'],
              lambda_div=params['lambda_div'],
              max_snapshots=3)
    # evaluate
    eval_accs = []
    for loader in tasks:
        it = iter(loader)
        acc = 0.0
        for _ in range(2):
            try:
                X, y = next(it)
            except StopIteration:
                break
            X, y = X.to(device), y.to(device)
            logits, *_ = model(X)
            acc += (logits.argmax(1) == y).float().mean().item()
        eval_accs.append(acc / 2)
    row = params.copy()
    for i, a in enumerate(eval_accs):
        row[f'task_{i}_acc'] = a
    row['avg_acc'] = sum(eval_accs) / len(eval_accs)
    return row

# def main():
#     # rank = thread_num() i guess? 
#     dist.init_process_group(backend='nccl', init_method='env://')
#     rank = dist.get_rank()
#     print(rank)
#     world_size = dist.get_world_size()

#     # Number of GPUs and runs per GPU
#     num_gpus = torch.cuda.device_count()
#     runs_per_gpu = world_size // num_gpus

#     # Map each process to a GPU in a round-robin fashion
#     local_gpu = rank % num_gpus
#     torch.cuda.set_device(local_gpu)
#     device = torch.device(f'cuda:{local_gpu}')

#     # Build parameter grid
#     param_grid = {
#         'lambda_anchor':    [0.01, 0.1, 1.0],
#         'lambda_kl':        [0.001, 0.01],
#         'lambda_sp':        [1e-4, 1e-3],
#         'lambda_rep':       [0.01, 0.1],
#         'lambda_distill':   [0.01, 0.1],
#         'lambda_div':       [0.001, 0.01],
#     }
#     grid = list(ParameterGrid(param_grid))
#     print(f"Total combinations: {len(grid)}")

#     # Prepare tasks once
#     base_ds = MNIST('./data', train=True, download=True, transform=ToTensor())
#     T = 10
#     perms = [torch.randperm(784) for _ in range(T)]
#     tasks = [
#         DataLoader(PermutedMNIST(base_ds, p), batch_size=128, shuffle=True, drop_last=True)
#         for p in perms
#     ]

#     # Each rank handles a subset of the grid
#     local_results = []
#     for idx, params in enumerate(grid):
#         if idx % world_size != rank:
#             continue
#         local_results.append(run_trial(params, tasks, device))
#     df = pd.DataFrame(local_results)


#     out_file = f"sweep_results_rank{rank}.csv"
#     df.to_csv(out_file, index=False)

#     dist.destroy_process_group()

# if __name__ == "__main__":
#     main()

from torchvision.transforms import ToTensor

# Download MNIST
train_dataset = MNIST(root='./data', train=True, download=True, transform=ToTensor())

# Create T tasks (each with a random permutation)
T = 5# Number of tasks
tasks = []
perms = [] 
for _ in range(T):
    perm = torch.randperm(784)  # Random pixel permutation
    perms.append(perm)
    task_dataset = PermutedMNIST(train_dataset, perm)
    task_loader = DataLoader(task_dataset, batch_size=128, shuffle=True, drop_last = True)
    tasks.append(task_loader)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SFI_MLP().to(device)

# Train
train_sfi(model, tasks, device)

# -----------------------------------------------------------
# ─────────────────────────────────────────────────────────────
# t-SNE (3-D) visualisation of SFI activations
# ─────────────────────────────────────────────────────────────
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

device = next(model.parameters()).device

# 1.  Register a hook for the *post-noise* activations of SFI-2
acts, task_ids = [], []

def hook_sfi2(mod, inp, out):
    h = out[0].detach().cpu()       # out = (h, α, μ, σ)
    acts.append(h)

h = model.sfi2.register_forward_hook(hook_sfi2)

# 2.  Collect ONE batch per task (keeps memory & run-time low)
model.eval()
with torch.no_grad():
    for tid, loader in enumerate(tasks):
        X, _ = next(iter(loader))
        task_ids.extend([tid]*X.size(0))
        _ = model(X.to(device))     # triggers the hook

H   = torch.cat(acts, 0).numpy()    # [N, hidden_dim]
labs = np.array(task_ids)

# 3.  t-SNE → 3-D
tsne = TSNE(n_components=3,
            perplexity=30,
            learning_rate="auto",
            init="random",
            random_state=0)
H_3d = tsne.fit_transform(H)        # [N, 3]

# 4.  3-D scatter (default colour cycle handles palette)
fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(projection='3d')
for tid in range(len(tasks)):
    sel = labs == tid
    ax.scatter(H_3d[sel, 0], H_3d[sel, 1], H_3d[sel, 2],
               s=12, alpha=0.7, label=f"task {tid}")
ax.set_title("t-SNE (3-D) of SFI layer-2 activations")
ax.set_xlabel("t-SNE-1"); ax.set_ylabel("t-SNE-2"); ax.set_zlabel("t-SNE-3")
ax.legend()
plt.tight_layout()
plt.savefig("tsne_activations.png")

# 5.  Clean-up
h.remove()
model.train()



# ─────────────────────────────────────────────────────────────
# t‑SNE (3‑D) of SFI layer‑2 activations
# Colour = CLASS hue, gradient shade encodes TASK index
# ─────────────────────────────────────────────────────────────
import torch, numpy as np, matplotlib.pyplot as plt, colorsys
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

device = next(model.parameters()).device

# 1. Capture post‑noise activations with forward hook
acts, class_ids, task_ids = [], [], []

def hook_sfi2(mod, ins, outs):
    acts.append( outs[0].detach().cpu() )   # h tensor

handle = model.sfi2.register_forward_hook(hook_sfi2)

# 2. Iterate ONE batch per task, record labels + task idx
model.eval()
with torch.no_grad():
    for tid, loader in enumerate(tasks):
        X, y = next(iter(loader))
        class_ids.extend( y.tolist() )
        task_ids.extend( [tid]*y.size(0) )
        _ = model( X.to(device) )

# tensors → numpy
H = torch.cat(acts,0).numpy().astype(np.float32)
cls = np.array(class_ids)
tsk = np.array(task_ids)
T   = tsk.max()+1

# 3. t‑SNE → 3‑D
H3d = TSNE(n_components=3, perplexity=30, init="random",
           learning_rate="auto", random_state=0).fit_transform(H)

# 4. Build colour map: tab10 hue per class + lightness gradient over task
base_colors = plt.cm.tab10(np.linspace(0,1,10))  # RGBA
def shade(color_rgba, task_idx):
    r,g,b, _ = color_rgba
    h,l,s = colorsys.rgb_to_hls(r,g,b)
    # linearly vary lightness across tasks: older task = darker
    l_new = 0.4 + 0.4*(task_idx / max(T-1,1))     # l in [0.4,0.8]
    r2,g2,b2 = colorsys.hls_to_rgb(h, l_new, s)
    return (r2,g2,b2,0.8)

colors = np.array([ shade(base_colors[c], t) for c,t in zip(cls, tsk) ])

# 5. 3‑D scatter
fig = plt.figure(figsize=(8,6))
ax  = fig.add_subplot(projection='3d')
ax.scatter(H3d[:,0], H3d[:,1], H3d[:,2],
           s=14, c=colors, alpha=0.9)

# build legend patches: show gradient for class 0 as example
from matplotlib.patches import Patch
legend_patches=[]
for c in range(10):
    legend_patches.append( Patch(color=shade(base_colors[c], T-1), label=f"class {c}") )
ax.legend(handles=legend_patches, title="Hue = class\nShade = task", bbox_to_anchor=(1.05,1), loc="upper left")

ax.set_title("t‑SNE (3‑D): class hue, task gradient")
ax.set_xlabel("t‑SNE‑1"); ax.set_ylabel("t‑SNE‑2"); ax.set_zlabel("t‑SNE‑3")
plt.tight_layout()
plt.savefig("tsne_class_task_gradient.png")
plt.show()

# 6. Clean‑up
handle.remove()




# ─────────────────────────────────────────────────────────────
# PCA visualisation of task-embeddings (r) for each task
# ─────────────────────────────────────────────────────────────
import torch, matplotlib.pyplot as plt, numpy as np
from sklearn.decomposition import PCA

device = next(model.parameters()).device

# 1) grab one batch per task
model.eval()
reps, task_ids = [], []
with torch.no_grad():
    for tid, loader in enumerate(tasks):
        X, _ = next(iter(loader))                # one mini-batch / task
        r = model.forward_features(X.to(device)) # [B, task_rep_dim]
        reps.append(r.cpu())
        task_ids.extend([tid]*X.size(0))
R   = torch.cat(reps, 0).numpy()                 # [N, d]
labels = np.array(task_ids)

# 2) PCA → 2-D
pca  = PCA(n_components=2, random_state=0)
R2d  = pca.fit_transform(R)
print(f"Variance captured by first 2 PCs: {pca.explained_variance_ratio_.sum():.1%}")

# 3) scatter
plt.figure(figsize=(6,5))
palette = plt.cm.get_cmap("tab10", len(tasks))
for tid in range(len(tasks)):
    idx = labels == tid
    plt.scatter(R2d[idx,0], R2d[idx,1], s=12, alpha=0.7,
                color=palette(tid), label=f"task {tid}")
plt.title("PCA (2-D) of per-sample task embeddings")
plt.xlabel("PC-1"); plt.ylabel("PC-2")
plt.legend(); plt.tight_layout(); plt.savefig("task_embeddings.png")

model.train();


tasks = []
for i in range(T):
    perm = perms[i]
    task_dataset = PermutedMNIST(train_dataset, perm)
    task_loader = DataLoader(task_dataset, batch_size=128, shuffle=True, drop_last = True)
    tasks.append(task_loader)
# --- after training ---
evaluate_sfi(model, tasks, device, num_batches=3)

