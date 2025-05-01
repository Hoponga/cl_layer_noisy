import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import random

# 1) PermutedMNIST dataset wrapper
class PermutedMNIST(Dataset):
    def __init__(self, base_ds, perm):
        self.base = base_ds
        self.perm = perm

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, lbl = self.base[idx]
        return img.view(-1)[self.perm], lbl

# 2) Simple 2-layer MLP baseline
class MLPBaseline(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)

# 3) Simple replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.storage = []
        self.ptr = 0

    def add(self, X, y):
        X_cpu = X.detach().cpu()
        y_cpu = y.detach().cpu()
        for xi, yi in zip(X_cpu, y_cpu):
            if len(self.storage) < self.capacity:
                self.storage.append((xi, yi))
            else:
                self.storage[self.ptr] = (xi, yi)
                self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.storage, min(len(self.storage), batch_size))
        Xs, ys = zip(*batch)
        return torch.stack(Xs), torch.tensor(ys)

# 4) Evaluation helper
def evaluate(model, loaders, device, num_batches=2):
    model.eval()
    accs = []
    with torch.no_grad():
        for loader in loaders:
            it = iter(loader)
            batch_acc = []
            for _ in range(num_batches):
                try:
                    X, y = next(it)
                except StopIteration:
                    break
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(dim=1)
                batch_acc.append((preds == y).float().mean().item())
            accs.append(sum(batch_acc) / len(batch_acc) if batch_acc else 0.0)
    model.train()
    return accs

# 5) Training loop with replay
def train_baseline_with_replay(
    model, task_loaders, device,
    lr=1e-3, epochs_per_task=1,
    replay_capacity=2000, replay_batch=64,
    lambda_rep=1.0
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    buffer = ReplayBuffer(replay_capacity)
    seen_loaders = []

    for task_id, loader in enumerate(task_loaders):
        print(f"\n--- Training Task {task_id} ---")
        seen_loaders.append(loader)

        for epoch in range(epochs_per_task):
            for X, y in loader:
                X, y = X.to(device), y.to(device)

                # Forward on new data
                logits = model(X)
                loss = criterion(logits, y)

                # Replay if buffer has enough samples
                if len(buffer.storage) >= replay_batch:
                    Xr, yr = buffer.sample(replay_batch)
                    Xr, yr = Xr.to(device), yr.to(device)
                    logits_r = model(Xr)
                    loss_rep = criterion(logits_r, yr)
                    loss = loss + lambda_rep * loss_rep

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Add new batch to buffer
                buffer.add(X, y)

        # Evaluate on all seen tasks
        accs = evaluate(model, seen_loaders, device, num_batches=3)
        for tid, acc in enumerate(accs):
            print(f"  Eval on Task {tid}: {acc*100:5.2f}%")

    return model

if __name__ == "__main__":
    torch.manual_seed(0)
    T = 5
    base_ds = MNIST('./data', train=True, download=True, transform=ToTensor())
    perms = [torch.randperm(784) for _ in range(T)]
    task_loaders = [
        DataLoader(PermutedMNIST(base_ds, p), batch_size=128, shuffle=True, drop_last=True)
        for p in perms
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPBaseline().to(device)
    train_baseline_with_replay(model, task_loaders, device)
