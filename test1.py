#!/usr/bin/env python3
# ================================================================
# classical_vs_quantum.py
# Benchmark mạng fully–connected vs. 2-qubit VQC nhiều tầng
# ================================================================
import time, random, torch, torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------------------
# 1) Dataset (2-D binary classification)
# ----------------------------------------------------------------
X, y = make_moons(n_samples=800, noise=0.15, random_state=SEED)
X = StandardScaler().fit_transform(X).astype("float32")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
X_train, X_test = torch.tensor(X_train), torch.tensor(X_test)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(
    y_test, dtype=torch.float32
)

# ----------------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------------
def n_params(model):
    """Số tham số trainable."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def accuracy(model, X, y):
    """Độ chính xác 0–1."""
    model.eval()
    logits = model(X)
    pred = (torch.sigmoid(logits).squeeze() > 0.5).int()
    return (pred == y.int()).float().mean().item()


# ----------------------------------------------------------------
# 3) Classical fully-connected network
# ----------------------------------------------------------------
class ClassicalMLNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1),  # ra logits
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------
# 4) “Quantum-inspired” 2-qubit VQC nhiều tầng
#    (ma trận 4×4; không cần Qiskit/PennyLane)
# ----------------------------------------------------------------
I2 = torch.eye(2, dtype=torch.complex64)
Xg = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)        # Pauli-X
Yg = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)     # Pauli-Y
Zg = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)       # Pauli-Z


def Rx(theta):
    return torch.cos(theta / 2) * I2 - 1j * torch.sin(theta / 2) * Xg


def Ry(theta):
    return torch.cos(theta / 2) * I2 - 1j * torch.sin(theta / 2) * Yg


def kron2(a, b):
    """Kronecker product 2 qubit."""
    return torch.kron(a, b)


# CNOT (target = qubit 1, control = qubit 0)
CNOT_01 = torch.tensor(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]], dtype=torch.complex64)

Z0I = kron2(Zg, I2)      # phép đo Z trên qubit 0


class QuantumMLNet(nn.Module):
    """
    2 qubit – mỗi layer: Ry(w0) ⊗ Ry(w1) -> CNOT
    Feature map: Rx(x1) ⊗ Rx(x2)
    """

    def __init__(self, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        # weight shape (n_layers, 2) – mỗi qubit một tham số/layer
        self.weight = nn.Parameter(0.1 * torch.randn(n_layers, 2))

    # ----------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------
    def _layer_U(self, w):
        """Một layer: Ry(w0)⊗Ry(w1) tiếp CNOT."""
        U_param = kron2(Ry(w[0]), Ry(w[1]))
        return CNOT_01 @ U_param

    def _expectation(self, state):
        """<Z0⊗I> trên ket (4,)"""
        return torch.vdot(state, Z0I @ state).real

    # ----------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------
    def circuit(self, sample):
        x1, x2 = sample
        # trạng thái |00>
        state = torch.zeros(4, dtype=torch.complex64)
        state[0] = 1.0
        # feature encoding
        U = kron2(Rx(x1), Rx(x2))
        # stack n_layers
        for l in range(self.n_layers):
            U = self._layer_U(self.weight[l]) @ U
        state = U @ state
        # map expectation -> logit
        exp_z = self._expectation(state)
        p1 = (1 - exp_z) / 2
        return torch.log(p1 / (1 - p1))

    def forward(self, x):
        logits = [self.circuit(s) for s in x]
        return torch.stack(logits).unsqueeze(1)


# ----------------------------------------------------------------
# 5) Train & benchmark
# ----------------------------------------------------------------
def train(
    model,
    Xtr,
    ytr,
    lr=0.02,
    epochs=200,
    batch=64,
    use_scheduler=True,
):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs
        )
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(Xtr)

    for ep in range(epochs):
        idx = torch.randperm(n)
        for i in range(0, n, batch):
            b = idx[i : i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]).squeeze(), ytr[b])
            loss.backward()
            opt.step()
        if use_scheduler:
            scheduler.step()


def benchmark(model_ctor, name, **train_cfg):
    model = model_ctor()
    t0 = time.time()
    train(model, X_train, y_train, **train_cfg)
    tr_time = time.time() - t0

    t1 = time.time()
    acc = accuracy(model, X_test, y_test)
    inf_ms = (time.time() - t1) / len(X_test) * 1000

    return dict(
        model=name,
        params=n_params(model),
        train_time_s=round(tr_time, 3),
        infer_ms=round(inf_ms, 3),
        accuracy=round(acc, 3),
    )


# ----------------------------------------------------------------
# 6) Main
# ----------------------------------------------------------------
if __name__ == "__main__":
    cfg = dict(epochs=200, lr=0.02, batch=64)

    res_fc = benchmark(ClassicalMLNet, "Classical FC", **cfg)
    res_vqc = benchmark(lambda: QuantumMLNet(n_layers=3), "VQC-2q-3layer", **cfg)

    print("\n=== BENCHMARK RESULTS ===")
    for r in (res_fc, res_vqc):
        print(
            f"{r['model']:15s} | params={r['params']:3d}"
            f" | train={r['train_time_s']:6.2f}s"
            f" | infer={r['infer_ms']:6.3f} ms/-sample"
            f" | acc={r['accuracy']:.3f}"
        )

