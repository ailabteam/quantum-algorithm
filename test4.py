#!/usr/bin/env python3
# ================================================================
# quantum_moons_lightning_gpu.py
# Benchmark: Classical FC vs 4-qubit VQC (PennyLane Lightning-GPU)
# ================================================================
import time, random, torch, torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pennylane as qml

# --------------------- 1) DEVICE & SEED -------------------------
SEED = 42
random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

qml_device = qml.device("lightning.gpu", wires=4, shots=None)  # 4 qubit
print(">>> PennyLane backend:", qml_device.name, "| GPU:", qml_device.CUDA)

device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(">>> Torch running on:", device_torch)

# --------------------- 2) DATASET -------------------------------
X, y = make_moons(n_samples=1000, noise=0.15, random_state=SEED)
X = StandardScaler().fit_transform(X).astype("float32")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
X_train = torch.tensor(X_train, device=device_torch)
X_test  = torch.tensor(X_test,  device=device_torch)
y_train = torch.tensor(y_train, dtype=torch.float32, device=device_torch)
y_test  = torch.tensor(y_test,  dtype=torch.float32, device=device_torch)

# --------------------- 3) CLASSICAL BASELINE --------------------
class ClassicalFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x): return self.net(x)

# --------------------- 4) 4-QUBIT VQC ---------------------------
n_qubits = 4
n_layers = 6                    # tăng / giảm số layer tùy ý

@qml.qnode(qml_device, interface="torch", diff_method="parameter-shift")
def vqc_circuit(x, weights):
    """x: (2,) – 2 feature; weights: (n_layers, n_qubits, 3)"""
    # -- feature embedding: angle-encode vào qubit 0 & 1
    qml.RX(torch.pi * x[0], wires=0)
    qml.RX(torch.pi * x[1], wires=1)
    # (qubit 2-3 giữ nguyên, nhưng vẫn tham gia entangle)
    # -- variational layers
    for l in range(n_layers):
        for q in range(n_qubits):
            qml.RY(weights[l, q, 0], wires=q)
            qml.RZ(weights[l, q, 1], wires=q)
            qml.RX(weights[l, q, 2], wires=q)
        # entangle full ring
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q+1) % n_qubits])
    # -- đo PauliZ trên qubit 0
    return qml.expval(qml.PauliZ(0))

class QuantumNet(nn.Module):
    def __init__(self):
        super().__init__()
        # tham số: (n_layers, n_qubits, 3) ~ 6*4*3 = 72
        init = 0.01 * torch.randn(n_layers, n_qubits, 3)
        self.weights = nn.Parameter(init)
    def forward(self, x_batch):
        # vector hoá với torch.vmap, chạy song song trên GPU
        exp_z = torch.vmap(lambda x: vqc_circuit(x, self.weights))(x_batch)
        p1 = (1 - exp_z) * 0.5
        return torch.log(p1 / (1 - p1)).unsqueeze(1)  # logits

# --------------------- 5) TRAINING UTILS ------------------------
def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    preds = (torch.sigmoid(model(X)).squeeze() > 0.5).int()
    return (preds == y.int()).float().mean().item()

def train(model, X, y, *, lr=0.05, epochs=250, batch=128, scheduler=True):
    model.to(device_torch)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if scheduler else None
    loss_fn = nn.BCEWithLogitsLoss()
    N = len(X)
    for ep in range(epochs):
        idx = torch.randperm(N, device=device_torch)
        for i in range(0, N, batch):
            b = idx[i:i+batch]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X[b]).squeeze(), y[b])
            loss.backward(); opt.step()
        if sch: sch.step()

def bench(model_ctor, name, **train_cfg):
    model = model_ctor()
    t0 = time.perf_counter()
    train(model, X_train, y_train, **train_cfg)
    if device_torch.type == "cuda": torch.cuda.synchronize()
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    acc = accuracy(model, X_test, y_test)
    if device_torch.type == "cuda": torch.cuda.synchronize()
    infer_time = (time.perf_counter() - t1) / len(X_test) * 1000

    return dict(model=name,
                params=n_params(model),
                train_s=round(train_time,2),
                infer_ms=round(infer_time,3),
                acc=round(acc,3))

# --------------------- 6) RUN BENCHMARK -------------------------
if __name__ == "__main__":
    cfg = dict(lr=0.05, epochs=250, batch=128)

    res_fc  = bench(ClassicalFC, "FC-16-1", **cfg)
    res_vqc = bench(QuantumNet,  f"VQC-{n_qubits}q-{n_layers}L", **cfg)

    print("\n=== BENCHMARK RESULTS ===")
    for r in (res_fc, res_vqc):
        print(f"{r['model']:14s} | params={r['params']:3d}"
              f" | train={r['train_s']:6.2f}s"
              f" | infer={r['infer_ms']:6.3f} ms/-sample"
              f" | acc={r['acc']:.3f}")

