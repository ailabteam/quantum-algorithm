#!/usr/bin/env python3
# ================================================================
# quantum_moons_lightning_gpu.py
# Benchmark: Classical FC vs. 4-qubit VQC (PennyLane Lightning-GPU)
# ================================================================
import time, random, torch, torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pennylane as qml

# ---------------- 1) Thiết lập thiết bị & seed ------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

qml_device = qml.device("lightning.gpu", wires=4, shots=None)  # 4 qubit
print(">>> PennyLane device:", qml_device.name)

device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(">>> PyTorch device :", device_torch)

# ---------------- 2) Dataset Moons (chuẩn hoá) ------------------
X, y = make_moons(n_samples=1000, noise=0.15, random_state=SEED)
X = StandardScaler().fit_transform(X).astype("float32")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

X_train = torch.tensor(X_train, device=device_torch)
X_test  = torch.tensor(X_test,  device=device_torch)
y_train = torch.tensor(y_train, dtype=torch.float32, device=device_torch)
y_test  = torch.tensor(y_test,  dtype=torch.float32, device=device_torch)

# ---------------- 3) Classical baseline (FC-16-1) ---------------
class ClassicalFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

# ---------------- 4) 4-qubit VQC (6 layer) ----------------------
n_qubits = 4
n_layers = 6   # thay đổi sâu nông tại đây

@qml.qnode(qml_device, interface="torch", diff_method="parameter-shift")
def vqc_circuit(x, weights):
    """x: (2,)   weights: (n_layers, n_qubits, 3)"""
    # -- feature encoding (2 qubit đầu)
    qml.RX(torch.pi * x[0], wires=0)
    qml.RX(torch.pi * x[1], wires=1)

    # -- variational layers
    for l in range(n_layers):
        for q in range(n_qubits):
            qml.RY(weights[l, q, 0], wires=q)
            qml.RZ(weights[l, q, 1], wires=q)
            qml.RX(weights[l, q, 2], wires=q)
        # full-ring entanglement
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])

    # -- đo Pauli-Z trên qubit 0
    return qml.expval(qml.PauliZ(0))

class QuantumNet(nn.Module):
    def __init__(self):
        super().__init__()
        init = 0.01 * torch.randn(n_layers, n_qubits, 3)
        self.weights = nn.Parameter(init)

    def forward(self, X):
        # vector-hóa toàn batch
        expz = torch.vmap(lambda xi: vqc_circuit(xi, self.weights))(X)
        p1 = 0.5 * (1 - expz)
        return torch.log(p1 / (1 - p1)).unsqueeze(1)  # logits

# ---------------- 5) Utilities ----------------------------------
def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    pred = (torch.sigmoid(model(X)).squeeze() > 0.5).int()
    return (pred == y.int()).float().mean().item()

def train(model, X, y, *, lr=0.05, epochs=250, batch=128):
    model.to(device_torch)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    N = len(X)
    for _ in range(epochs):
        idx = torch.randperm(N, device=device_torch)
        for i in range(0, N, batch):
            b = idx[i:i+batch]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X[b]).squeeze(), y[b])
            loss.backward()
            opt.step()
        sched.step()

def benchmark(model_ctor, name, **cfg):
    model = model_ctor()
    t0 = time.perf_counter()
    train(model, X_train, y_train, **cfg)
    if device_torch.type == "cuda":
        torch.cuda.synchronize()
    train_t = time.perf_counter() - t0

    t1 = time.perf_counter()
    acc = accuracy(model, X_test, y_test)
    if device_torch.type == "cuda":
        torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t1) / len(X_test) * 1000

    return dict(model=name, params=n_params(model),
                train_s=round(train_t, 2),
                infer_ms=round(infer_ms, 3),
                acc=round(acc, 3))

# ---------------- 6) Execute ------------------------------------
if __name__ == "__main__":
    hyper = dict(lr=0.05, epochs=250, batch=128)

    res_fc  = benchmark(ClassicalFC, "FC-16-1", **hyper)
    res_vqc = benchmark(QuantumNet,  f"VQC-{n_qubits}q-{n_layers}L", **hyper)

    print("\n=== BENCHMARK RESULTS ===")
    for r in (res_fc, res_vqc):
        print(f"{r['model']:14s} | params={r['params']:3d}"
              f" | train={r['train_s']:6.2f}s"
              f" | infer={r['infer_ms']:6.3f} ms/sample"
              f" | acc={r['acc']:.3f}")

