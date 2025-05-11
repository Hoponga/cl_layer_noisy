import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm, trange
# ------------------------------------------------------------------ #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT   = 256          # width of flow & ctx vector
T_DIM    = 64           # task-embedding width
BLOCKS   = 6            # RealNVP coupling blocks
EPOCHS   = 3            # passes per task
BATCH    = 128
REPLAY_B = 128  
# ------------------------------------------------------------------ #
# Permuted-MNIST tasks
class PermutedMNIST(Dataset):
    def __init__(self, base, perm): self.base, self.perm = base, perm
    def __getitem__(self, i):
        x,y = self.base[i]; return x.view(-1)[self.perm], y
    def __len__(self): return len(self.base)

def build_tasks(T=10):
    base = MNIST(root='./data', train=True, download=True, transform=ToTensor())
    tasks, perms = [], []
    for _ in range(T):
        p = torch.randperm(28*28)
        loader = DataLoader(PermutedMNIST(base,p), BATCH, True, drop_last=True)
        tasks.append(loader); perms.append(p)
    return tasks, perms
# ------------------------------------------------------------------ #

# take in mnist inputs and generate a task embedding vector of size T_DIM from this 
class TaskEmbed(nn.Module):
    def __init__(self):   # ctx_dim == LATENT
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT, 128), nn.ReLU(),
            nn.Linear(128, T_DIM))
    def forward(self, ctx):            # ctx (B,d)
        return self.net(ctx.mean(0,keepdim=True))  # (1,T_DIM)

# RealNVP conditional block
#self.nn generates a hidden vector of size hid 
# self.scale and self.shift scale half of the input vecotr 
# the other half is passed through unchagnged 
class Coupling(nn.Module):
    def __init__(self, d=LATENT, ctx=T_DIM, hid=256):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(d//2+ctx, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU())
        self.scale = nn.Linear(hid, d//2)
        self.shift = nn.Linear(hid, d//2)
    def forward(self, x, r, rev=False):
        x1, x2 = x.chunk(2,-1)
        r = r.expand(x1.size(0), -1)
        h = self.nn(torch.cat([x1,r],-1))
        s = torch.tanh(self.scale(h)); t = self.shift(h)
        if rev:
            y2 = (x2 - t)*torch.exp(-s); logdet = -s.sum(-1)
        else:
            y2 = x2*torch.exp(s) + t;   logdet =  s.sum(-1)
        return torch.cat([x1,y2],-1), logdet

class Flow(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = nn.ModuleList([Coupling() for _ in range(BLOCKS)])
        self.base   = torch.distributions.Normal(0,1)
    def encode(self, x, r):
        z, ld = x, 0.
        for b in self.blocks:
            #print(z.shape, r.shape)
            z, d = b(z,r,rev=False); ld += d
        logp = self.base.log_prob(z).sum(-1)
        return z, logp - ld            # log q_r(z)
    def decode(self, z, r):
        x = z
        for b in reversed(self.blocks):
            x,_ = b(x,r,rev=True)
        return x

class FlowCL(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_embed = TaskEmbed()
        self.flow    = Flow()
        self.pre     = nn.Linear(784, LATENT)        # ① NEW
        self.head    = nn.Linear(LATENT,10)
    def forward(self, x):
        x = F.relu(self.pre(x))
        r   = self.t_embed(x)              # (1,T_DIM)
        z, _ = self.flow.encode(x, r)      # logq ignored ⇒ no NLL term
        return self.head(z), z, r

# ------------------------------------------------------------------ #
def sliced_w2(A,B,proj=50):
    d=A.size(1); v = torch.randn(proj,d,device=A.device); v=F.normalize(v,dim=-1)
    Ap = (A@v.T).sort(0)[0]; Bp=(B@v.T).sort(0)[0]
    return F.mse_loss(Ap,Bp,reduction='mean')

### REPLAY memory structure ######################################
replay_lat  = []          # list of (z,y,r) tensors  (CPU)
REPLAY_MAX  = 5000        # total latent samples kept

def add_to_replay(z,y,r):
    if len(replay_lat)*BATCH > REPLAY_MAX: replay_lat.pop(0)
    replay_lat.append((z.detach().cpu(), y.detach().cpu(), r.detach().cpu()))

def sample_replay(batch):
    z,y,r = random.choice(replay_lat)
    idx   = torch.randint(0,z.size(0), (batch,))
    return z[idx].to(device), y[idx].to(device), r.to(device)
# --------------------------------------------------------------- #
def train_task(loader,model,opt,lat_mem,r_mem,lamb, t = 0):
    model.train()
    for _ in range(EPOCHS):
        for x,y in loader:
            x=x.view(x.size(0),-1).to(device); y=y.to(device)
            # -------------- forward current ------------------
            logits,z,r = model(x)
            collect_r(r, task_id=t)
            cls = F.cross_entropy(logits,y)
            # -------------- replay ---------------------------
            rep_ce = torch.tensor(0.,device=device)
            if replay_lat:
                z_old,y_old,r_old = sample_replay(REPLAY_B)

                # step-a: decode to ctx space (128-dim)
                ctx_old = model.flow.decode(z_old, r_old)

                # step-b: re-encode ctx with *current* flow to get z_cur
                z_cur, _ = model.flow.encode(ctx_old, r_old)

                # step-c: classify
                logits_old = model.head(z_cur)
                rep_ce = F.cross_entropy(logits_old, y_old)
            # -------------- anchor & repulse -----------------
            anch=rep=torch.tensor(0.,device=device); r_h=0.
            for Z,ro in zip(lat_mem,r_mem):
                Z=Z.to(device); ro=ro.to(device)
                anch += sliced_w2(z,Z)
                rep  += F.relu(lamb['margin']-sliced_w2(z,Z))
                r_h  += F.relu(lamb['r_m']-(r-ro).norm())
            if lat_mem: anch/=len(lat_mem)
            scale_reg = sum(b.scale.weight.abs().mean() for b in model.flow.blocks)
            loss = cls + lamb['replay']*rep_ce + lamb['anch']*anch + \
                   lamb['rep']*rep + 0.1*r_h + 1e-3*scale_reg
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
    # -------- store memory & replay latents ---------------------
    with torch.no_grad():
        lat_mem.append(z[:256].detach().cpu()); r_mem.append(r.detach().cpu())
        add_to_replay(z[:512], y[:512], r)   ### add to latent buffer
# --------------------------------------------------------------- #
@torch.no_grad()
def evaluate(tasks,model,upto):
    model.eval(); acc=[]
    for i in range(upto):
        tot=0; it=iter(tasks[i])
        for _ in range(5):
            x,y=next(it); x=x.view(x.size(0),-1).to(device); y=y.to(device)
            pred=model(x)[0].argmax(1); tot+=(pred==y).float().mean().item()
        acc.append(tot/5)
    print("Acc:", [f"{a*100:5.2f}" for a in acc], "Avg", f"{sum(acc)/len(acc)*100:5.2f}")
# --------------------------------------------------------------- #
import random, math, time

# ---------- 1. helper containers ---------------------------------
R_STORE = []           # list:   [[r_vecs_task0], [r_vecs_task1], ...]
MAX_R   = 200          # how many r-vectors to keep per task

def collect_r(r_tensor, task_id):
    """Call inside the training loop.  r_tensor is shape (1, T_DIM)."""
    r_cpu = r_tensor.squeeze(0).detach().cpu()
    if task_id == len(R_STORE):
        R_STORE.append([])                       # first batch of new task
    if len(R_STORE[task_id]) < MAX_R:
        R_STORE[task_id].append(r_cpu)           # keep up to MAX_R

# ---------- 2. plotting routine ----------------------------------
def plot_r_space(method='pca', id = 0):
    import numpy as np, matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # stack → (N_total, T_DIM)  and   labels → (N_total,)
    all_r   = torch.vstack([torch.stack(task) for task in R_STORE]).numpy()
    labels  = np.concatenate([[tid]*len(task) for tid,task in enumerate(R_STORE)])

    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, init='pca')
    else:
        raise ValueError('method must be "pca" or "tsne"')
    xy = reducer.fit_transform(all_r)

    plt.figure(figsize=(6,5))
    scatter = plt.scatter(xy[:,0], xy[:,1], c=labels, cmap='tab10', s=18)
    plt.title(f'Task-embedding space ({method.upper()})')
    plt.xlabel('Dim 1'); plt.ylabel('Dim 2')
    cb = plt.colorbar(scatter, ticks=range(len(R_STORE)))
    cb.set_label('task-id')
    plt.tight_layout()
    plt.savefig(f"r_space_{method}_{id}.png")
def main():
    tasks,_=build_tasks()
    model=FlowCL().to(device)
    opt  = torch.optim.Adam(model.parameters(),1e-4)
    #lamb = dict(anch=5, rep=0.5, margin=1, r_m=4, replay=1)
    lamb = dict(anch=20, rep=0.2, margin=0.8, r_m=8, replay=1)
    lat_mem=[]; r_mem=[]
    for t,loader in enumerate(tasks):
        print(f"\n=== Task {t} ===")
        train_task(loader,model,opt,lat_mem,r_mem,lamb, t = t)
        for p in model.head.parameters(): p.requires_grad_(False)
        evaluate(tasks,model,t+1)
        if t % 2 == 1:                       # plot every 2 tasks (optional)
            plot_r_space('pca', id = t)
if __name__=="__main__":
    import random, torch
    random.seed(0); torch.manual_seed(0)
    main()