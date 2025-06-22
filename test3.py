#!/usr/bin/env python3
# ================================================================
# classical_vs_quantum_vectorized.py
# ================================================================
import time, random, torch, torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --------------------- 1) DEVICE & SEED -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(">>> Running on:", device)
SEED = 42
random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# --------------------- 2) DATASET -------------------------------
X, y = make_moons(n_samples=800, noise=0.15, random_state=SEED)
X = StandardScaler().fit_transform(X).astype("float32")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                      stratify=y, random_state=SEED)
Xtr, Xte  = torch.tensor(Xtr, device=device), torch.tensor(Xte, device=device)
ytr, yte  = torch.tensor(ytr, dtype=torch.float32, device=device), \
            torch.tensor(yte, dtype=torch.float32, device=device)

# --------------------- 3) HELPERS -------------------------------
def n_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

@torch.no_grad()
def accuracy(m, X, y):
    m.eval(); p = (torch.sigmoid(m(X)).squeeze() > .5).int()
    return (p == y.int()).float().mean().item()

# --------------------- 4) CLASSICAL NET -------------------------
class ClassicalMLNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2,10), nn.ReLU(), nn.Linear(10,1))
    def forward(self,x): return self.net(x)

# --------------------- 5) VECTORISED VQC ------------------------
# Pauli & const on correct dtype/device
I2 = torch.eye(2, dtype=torch.complex64, device=device)
Xg = torch.tensor([[0,1],[1,0]], dtype=torch.complex64, device=device)
Yg = torch.tensor([[0,-1j],[1j,0]], dtype=torch.complex64, device=device)
Zg = torch.tensor([[1,0],[0,-1]], dtype=torch.complex64, device=device)

# 2-qubit CNOT (control 0 → target 1)
CNOT = torch.tensor([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,0,1],
                     [0,0,1,0]], dtype=torch.complex64, device=device)

Z0I  = torch.kron(Zg, I2)                      # đo Z trên qubit 0
state0 = torch.zeros(4,1, dtype=torch.complex64, device=device); state0[0]=1

def rx_batch(theta):
    """theta:(B,)  → (B,2,2) complex64"""
    c = torch.cos(theta/2).to(torch.complex64)
    s = torch.sin(theta/2).to(torch.complex64)
    return c[:,None,None]*I2 - 1j*s[:,None,None]*Xg

def ry_scalar(theta):
    """theta: scalar → (2,2) complex64 on device"""
    c, s = torch.cos(theta/2), torch.sin(theta/2)
    return c*I2 - 1j*s*Yg

def kron2_batch(A,B):
    """A,B:(B,2,2) → (B,4,4)"""
    return torch.einsum('bij,bkl->bikjl', A, B).reshape(-1,4,4)

def kron2_scalar(A,B, Bsz):
    """A,B:(2,2) scalar → repeat to (Bsz,4,4)"""
    return torch.kron(A,B).unsqueeze(0).expand(Bsz,-1,-1)

class QuantumMLNet(nn.Module):
    def __init__(self, n_layers=6):
        super().__init__()
        self.n_layers = n_layers
        self.weight = nn.Parameter(0.1*torch.randn(n_layers,2, device=device))
    # ------------------------------------------
    def forward(self, X):
        B = X.size(0)
        # ----- feature map Rx(x) ⊗ Rx(x) -------
        U = kron2_batch(rx_batch(X[:,0]), rx_batch(X[:,1]))  # (B,4,4)

        CNOT_batch = CNOT.unsqueeze(0).expand(B,4,4)

        for l in range(self.n_layers):
            w0, w1 = self.weight[l]
            U_layer = kron2_scalar(ry_scalar(w0), ry_scalar(w1), B)  # (B,4,4)
            U = torch.bmm(CNOT_batch, torch.bmm(U_layer, U))        # (B,4,4)

        # ----- apply to |00> and measure --------
        ψ = torch.bmm(U, state0.expand(B,4,1)).squeeze(-1)          # (B,4)
        expZ = torch.einsum('bi,ij,bj->b', ψ.conj(), Z0I, ψ).real    # (B,)
        p1   = (1 - expZ)*0.5
        return torch.log(p1/(1-p1)).unsqueeze(1)                    # (B,1)

# --------------------- 6) TRAINING ------------------------------
def train(model, X, y, *, lr=0.05, epochs=300, batch=128):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    N = len(X)
    for _ in range(epochs):
        idx = torch.randperm(N, device=device)
        for i in range(0,N,batch):
            b = idx[i:i+batch]
            opt.zero_grad()
            loss = loss_fn(model(X[b]).squeeze(), y[b])
            loss.backward(); opt.step()
        sched.step()

def benchmark(make_model, name, **train_cfg):
    model = make_model().to(device)
    t0=time.time(); train(model, Xtr, ytr, **train_cfg); torch.cuda.synchronize(); tr=time.time()-t0
    t1=time.time(); acc=accuracy(model, Xte, yte); torch.cuda.synchronize(); inf=(time.time()-t1)/len(Xte)*1000
    return dict(model=name, params=n_params(model),
                train=round(tr,2), infer_ms=round(inf,3), acc=round(acc,3))

# --------------------- 7) MAIN ---------------------------------
if __name__=="__main__":
    cfg = dict(epochs=300, lr=0.05, batch=128)

    res_fc  = benchmark(ClassicalMLNet, "FC-10-1", **cfg)
    res_vqc = benchmark(lambda:QuantumMLNet(6), "VQC-2q-6L", **cfg)

    print("\n=== VECTORISED GPU BENCHMARK ===")
    for r in (res_fc, res_vqc):
        print(f"{r['model']:12s}|params={r['params']:3d}"
              f"|train={r['train']:6.2f}s"
              f"|infer={r['infer_ms']:6.3f} ms"
              f"|acc={r['acc']:.3f}")

