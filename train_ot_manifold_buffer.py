# train_mean_teacher_anchor.py – MNIST / CIFAR10 CL with
#   ↳ EMA teacher        (no slow deepcopy)
#   ↳ per-class OT head
#   ↳ iCaRL herding      (balanced exemplars)
#   ↳ Laplacian anchoring on teacher prototypes during replay

import random, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Grayscale

# ─────────────── hyper-parameters ───────────────
T            = 5
BATCH        = 128
REPLAY_MB    = 128
REPLAY_CAP   = 4096*32            # 20 exemplars × 10 classes
D_CLS, D_TSK = 128, 10
LR           = 1e-3
EMA_ALPHA    = 0.99
LAM_OT       = 2.0
LAM_M        = 5e-3             # Laplacian anchor weight
device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
isMNIST, PERMUTED = True, False
# ────────────────────────────────────────────────

# ---------------- data loaders (unchanged) ----------------
class PMNIST(Dataset):
    def __init__(self, base, perm): self.base, self.perm = base, perm
    def __getitem__(s, i): x,y=s.base[i];return x.view(-1)[s.perm],y
    def __len__(s): return len(s.base)
class SplitMNIST(Dataset):
    def __init__(s, base, cl):
        s.data=[(x.view(-1),y) for x,y in base if y in cl]
    def __getitem__(s,i): return s.data[i]
    def __len__(s): return len(s.data)
def get_tasks():
    base=MNIST('./data',train=True,download=True,transform=ToTensor())
    if PERMUTED:
        perms=[torch.randperm(784) for _ in range(T)]
        return [DataLoader(PMNIST(base,p),batch_size=BATCH,shuffle=True,
                           drop_last=True) for p in perms]
    else:
        splits=[list(range(i,i+2)) for i in range(0,10,2)]
        return [DataLoader(SplitMNIST(base,c),batch_size=BATCH,shuffle=True,
                           drop_last=True) for c in splits]
tasks=get_tasks()

# # ---------------- herding replay buffer ----------------
# class HerdBuf:
#     def __init__(s, cap_pc, model): s.cap, s.m, s.mem=cap_pc,model,{c:[] for c in range(10)}
#     def add(s,x,y):
#         with torch.no_grad():
#             z=s.m.backbone(x.to(device))[:,-D_TSK:]
#             z=F.normalize(z,1).cpu()
#         for xi,yi,zi in zip(x.cpu(),y.cpu(),z):
#             buf=s.mem[int(yi)]
#             if len(buf)<s.cap: buf.append((xi,yi,zi)); continue
#             Z=torch.stack([z for _,_,z in buf]);mu=Z.mean(0)
#             worst=(Z-mu).norm(2,1).argmax()
#             if (zi-mu).norm()< (Z[worst]-mu).norm(): buf[worst]=(xi,yi,zi)
#     def sample(s,m):
#         xs,ys=[] ,[]
#         take=max(1,m//10)
#         for c,buf in s.mem.items():
#             for xi,yi,_ in random.sample(buf,min(take,len(buf))):
#                 xs.append(xi);ys.append(yi)
#         return (torch.stack(xs).to(device),torch.tensor(ys,device=device)) if xs else (None,None)

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

buf = ReplayBuf(REPLAY_CAP)

# ---------------- mean-teacher helpers ----------------
def ema(student,teacher,a=EMA_ALPHA):
    for ps,pt in zip(student.parameters(),teacher.parameters()):
        pt.data.mul_(a).add_(ps.data,alpha=1-a)

# ---------------- Laplacian builder on teacher prototypes ------------
def build_laplacian(P_anchor, sigma=0.5):
    d2=torch.cdist(P_anchor,P_anchor,2).pow(2)
    W=torch.exp(-d2/(2*sigma**2))
    D=torch.diag(W.sum(1)); return D-W

# ---------------- model components --------------------
class OTHead(nn.Module):
    def __init__(s,d,K):
        super().__init__(); s.P=nn.Parameter(torch.randn(K,d)); nn.init.normal_(s.P)
    def forward(s,z):
        Pn=F.normalize(s.P,1);scr=z@Pn.T/0.5
        Q=F.softmax(scr/0.1,1); ot=-(Q*F.log_softmax(scr,1)).sum(1).mean()
        idx=Q.argmax(1);cent=s.P[idx]
        return ot,cent
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        inp=784
        s.backbone=nn.Sequential(nn.Linear(inp,256),nn.ReLU(),nn.Linear(256,D_CLS+D_TSK))
        s.head=OTHead(D_TSK,10)
        s.fc=nn.Linear(D_CLS+D_TSK,10)
    def forward(s,x,replay_z=None):
        z=s.backbone(x)
        zc=F.normalize(z[:,:D_CLS],1);zt=F.normalize(z[:,D_CLS:],1)
        ot,cent=s.head(zt if replay_z is None else replay_z)
        log=s.fc(torch.cat([zc,cent.detach()],1))
        return log,ot
# ------------------------------------------------------

student=Net().to(device); teacher=Net().to(device)
teacher.load_state_dict(student.state_dict())
opt=torch.optim.Adam(student.parameters(),lr=LR)

# placeholders for prototype anchors
P_anchor=None; L_anchor=None

for t,loader in enumerate(tasks):
    print(f"\nTask {t}")
    student.train()
    for x,y in loader:
        x,y=x.to(device),y.to(device)

        # -------- current batch -----------
        log,ot=student(x,None); loss=F.cross_entropy(log,y)+LAM_OT*ot

        # -------- replay branch -----------
        xr,yr=buf.sample(REPLAY_MB)
        if xr is not None:
            with torch.no_grad():
                logT,_=teacher(xr,None)
                pT=F.softmax(logT/0.5,1)
            logS,ot_r=student(xr,None)
            pS=F.softmax(logS/0.5,1)
            wd=(pS.cumsum(1)-pT.cumsum(1)).abs().sum(1).mean()
            loss+=F.cross_entropy(logS,yr)+LAM_OT*ot_r+wd
            # ----- manifold anchor on TEACHER prototypes -----
            if P_anchor is not None:
                Delta=teacher.head.P-P_anchor
                man=torch.trace(Delta.T@L_anchor@Delta)
                loss+=LAM_M*man

        opt.zero_grad();loss.backward();opt.step()
        ema(student,teacher); buf.add(x.cpu(),y.cpu())

    # ------ after task: snapshot teacher prototypes & Laplacian ------
    with torch.no_grad():
        P_anchor=teacher.head.P.detach().clone()
        L_anchor=build_laplacian(P_anchor).to(device)

    # ------ evaluation ----------
    student.eval()
    for s in range(t+1):
        tot=ok=0
        for xv,yv in tasks[s]:
            xv,yv=xv.to(device),yv.to(device)
            pred,_=student(xv,None); ok+=(pred.argmax(1)==yv).sum().item(); tot+=len(yv)
        print(f"  Task {s} acc: {100*ok/tot:5.2f}%")
