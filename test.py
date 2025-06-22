# ================================================================
# Classical vs “Quantum-inspired” ML example (CPU-only, PyTorch)
# ================================================================
import time, random, math, torch, torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED); torch.manual_seed(SEED)

# ---------- generate toy dataset (2D, binary) ----------
X, y = make_moons(n_samples=800, noise=0.15, random_state=SEED)
X = StandardScaler().fit_transform(X).astype("float32")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
X_train, X_test = torch.tensor(X_train), torch.tensor(X_test)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# ---------- helpers ----------
def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    logits = model(X)
    preds = (torch.sigmoid(logits).squeeze() > 0.5).int()
    return (preds == y.int()).float().mean().item()

# ================================================================
# 1) Classical fully-connected network
# ================================================================
class ClassicalMLNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10), nn.ReLU(),
            nn.Linear(10, 1)
        )
    def forward(self, x):
        return self.net(x)

# ================================================================
# 2) “Quantum-inspired” 2-qubit variational circuit
#    (matrix simulation, no external QML libs)
# ================================================================
I2 = torch.eye(2, dtype=torch.complex64)
Xg = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
Yg = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)

def Rx(theta):
    return torch.cos(theta / 2) * I2 - 1j * torch.sin(theta / 2) * Xg

def Ry(theta):
    return torch.cos(theta / 2) * I2 - 1j * torch.sin(theta / 2) * Yg

def kron2(A, B):
    # Kronecker product for 2 qubits
    return torch.kron(A, B)

CNOT_01 = torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=torch.complex64)

# Measurement operator Z ⊗ I  (comment only – safe for Python)
Zg = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
Z0I = kron2(Zg, I2)        # shape (4,4)

class QuantumMLNet(nn.Module):
    """Encode two real features as Rx rotations on 2 qubits,
       apply one Ry weight per qubit, CNOT, then measure Z0."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2))  # w0, w1

    def circuit(self, x_single):
        θ1, θ2 = x_single
        # start in |00>
        state = torch.zeros(4, dtype=torch.complex64)
        state[0] = 1.0

        # feature encoding
        U_feat = kron2(Rx(θ1), Rx(θ2))
        # trainable layer
        U_param = kron2(Ry(self.weight[0]), Ry(self.weight[1]))
        # total evolution
        U = CNOT_01 @ U_param @ U_feat
        state = U @ state

        # expectation value ⟨Z⊗I⟩  → prob
        exp_z = torch.vdot(state, Z0I @ state).real
        p1 = (1 - exp_z) / 2          # map to [0,1]
        logit = torch.log(p1 / (1 - p1))
        return logit

    def forward(self, x):
        logits = [self.circuit(sample) for sample in x]
        return torch.stack(logits).unsqueeze(1)

# ================================================================
# Training & benchmarking
# ================================================================
def train(model, Xtr, ytr, lr=0.01, epochs=50, batch=64):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(Xtr)
    for _ in range(epochs):
        idx = torch.randperm(n)
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            opt.zero_grad()
            out = model(Xtr[b])
            loss = loss_fn(out.squeeze(), ytr[b])
            loss.backward()
            opt.step()

def benchmark(model_cls, name):
    model = model_cls()
    t0 = time.time()
    train(model, X_train, y_train)
    train_time = time.time() - t0

    t1 = time.time()
    acc = accuracy(model, X_test, y_test)
    infer_time = (time.time() - t1) / len(X_test)

    return {
        "model": name,
        "params": n_params(model),
        "train_time_s": round(train_time, 3),
        "infer_ms_per_sample": round(infer_time * 1000, 3),
        "accuracy": round(acc, 3)
    }

if __name__ == "__main__":
    res1 = benchmark(ClassicalMLNet, "Classical FC")
    res2 = benchmark(QuantumMLNet,   "2-Qubit VQC")
    for r in (res1, res2):
        print(r)

