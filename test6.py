#!/usr/bin/env python3
# ================================================================
# quantum_moons_lightning_gpu.py
# Classical FC vs 4-qubit VQC (PennyLane Lightning-GPU, no vmap)
# ================================================================
import time, random, torch, torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pennylane as qml

# ---------- 1) Thiết lập thiết bị & seed ------------------------
SEED = 42
random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

qml_device = qml.device("lightning.gpu", wires=4, shots=None)  # 4 qubit
print(">>> PennyLane device:", qml_device.name)

device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(">>> PyTorch device :", device_torch)

# ---------- 2) Dataset Moons -----------------------------------
X, y = make_moons(n_samples=1000, noise=0.15, random_state=SEED)
X = StandardScaler().fit_transform(X).astype("float32")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                      stratify=y, random_state=SEED)

Xtr = torch.tensor(Xtr, device=device_torch)
Xte = torch.tensor(Xte, device=device_torch)
ytr = torch.tensor(ytr, dtype=torch.float32, device=device_torch)
yte = torch.tensor(yte, dtype=torch.float32, device=device_torch)

# ---------- 3) Classical baseline ------------------------------
class ClassicalFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x): return self.net(x)

# ---------- 4) 4-qubit VQC -------------------------------------
n_qubits, n_layers = 4, 6

@qml.qnode(qml_device, interface="torch", diff_method="parameter-shift")
def circuit(x, w):
    # feature embedding
    qml.RX(torch.pi * x[0], 0)
    qml.RX(torch.pi * x[1], 1)
    # variational layers
    for l in range(n_layers):
        for q in range(n_qubits):
            qml.RY(w[l, q, 0], q)
            qml.RZ(w[l, q, 1], q)
            qml.RX(w[l, q, 2], q)
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q+1) % n_qubits])
    return qml.expval(qml.PauliZ(0))

class QuantumNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(0.01*torch.randn(n_layers, n_qubits, 3))
    def forward(self, X):
        # batch loop (không dùng vmap)
        expz = torch.stack([circuit(x, self.weights) for x in X])
        p1 = 0.5*(1-expz)
        return torch.log(p1/(1-p1)).unsqueeze(1)

# ---------- 5) Utils -------------------------------------------
def n_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

@torch.no_grad()
def accuracy(m, X, y):
    m.eval()
    pred = (torch.sigmoid(m(X)).squeeze()>0.5).int()
    return (pred == y.int()).float().mean().item()

def train(m, X, y, lr=0.05, epochs=250, batch=128):
    m.to(device_torch)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    N = len(X)
    for _ in range(epochs):
        idx = torch.randperm(N, device=device_torch)
        for i in range(0,N,batch):
            b = idx[i:i+batch]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(m(X[b]).squeeze(), y[b])
            loss.backward(); opt.step()
        sched.step()

def bench(make_model, name, **cfg):
    model = make_model()
    t0=time.perf_counter(); train(model, Xtr, ytr, **cfg)
    if device_torch.type=="cuda": torch.cuda.synchronize()
    tr=time.perf_counter()-t0
    t1=time.perf_counter(); acc=accuracy(model, Xte, yte)
    if device_torch.type=="cuda": torch.cuda.synchronize()
    inf=(time.perf_counter()-t1)/len(Xte)*1000
    return dict(model=name, params=n_params(model),
                train_s=round(tr,2), infer_ms=round(inf,3), acc=round(acc,3))

# ---------- 6) Chạy benchmark ----------------------------------
if __name__=="__main__":
    cfg=dict(lr=0.05, epochs=250, batch=128)
    res_fc  = bench(ClassicalFC, "FC-16-1", **cfg)
    res_vqc = bench(QuantumNet,  f"VQC-{n_qubits}q-{n_layers}L", **cfg)
    print("\n=== BENCHMARK RESULTS ===")
    for r in (res_fc, res_vqc):
        print(f"{r['model']:14s}|params={r['params']:3d}"
              f"|train={r['train_s']:6.2f}s"
              f"|infer={r['infer_ms']:6.3f} ms"
              f"|acc={r['acc']:.3f}")

