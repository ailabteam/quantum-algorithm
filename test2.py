#!/usr/bin/env python3
# ================================================================
# classical_vs_quantum_gpu.py
# ================================================================
import time, random, torch, torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ------------ 1) Thiết lập thiết bị -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(">>> Running on:", device)

SEED = 42
random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ------------ 2) Dataset ---------------------------------------
X, y = make_moons(n_samples=800, noise=0.15, random_state=SEED)
X = StandardScaler().fit_transform(X).astype("float32")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
X_train = torch.tensor(X_train, device=device)
X_test  = torch.tensor(X_test,  device=device)
y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
y_test  = torch.tensor(y_test,  dtype=torch.float32, device=device)

# ------------ 3) Helpers ---------------------------------------
def n_params(model):      # đếm tham số
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()          # accuracy
def accuracy(model, X, y):
    model.eval()
    logits = model(X)
    pred = (torch.sigmoid(logits).squeeze() > 0.5).int()
    return (pred == y.int()).float().mean().item()

# ------------ 4) Classical fully-connected ----------------------
class ClassicalMLNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10), nn.ReLU(),
            nn.Linear(10, 1)
        )
    def forward(self, x): return self.net(x)

# ------------ 5) VQC 2-qubit nhiều tầng -------------------------
I2 = torch.eye(2, dtype=torch.complex64, device=device)
Xg = torch.tensor([[0,1],[1,0]], dtype=torch.complex64, device=device)
Yg = torch.tensor([[0,-1j],[1j,0]], dtype=torch.complex64, device=device)
Zg = torch.tensor([[1,0],[0,-1]], dtype=torch.complex64, device=device)

def Rx(t): return torch.cos(t/2)*I2 - 1j*torch.sin(t/2)*Xg
def Ry(t): return torch.cos(t/2)*I2 - 1j*torch.sin(t/2)*Yg
def kron2(a,b): return torch.kron(a,b)

CNOT_01 = torch.tensor([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,0,1],
                        [0,0,1,0]],
                       dtype=torch.complex64, device=device)
Z0I = kron2(Zg, I2)

class QuantumMLNet(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        self.weight = nn.Parameter(0.1*torch.randn(n_layers,2, device=device))
    # ---- một layer: Ry ⊗ Ry + CNOT
    def _layerU(self, w):
        return CNOT_01 @ kron2(Ry(w[0]), Ry(w[1]))
    # ---- circuit cho 1 sample
    def circuit(self, x):
        x1,x2 = x
        state = torch.zeros(4, dtype=torch.complex64, device=device); state[0]=1
        U = kron2(Rx(x1), Rx(x2))
        for l in range(self.n_layers):
            U = self._layerU(self.weight[l]) @ U
        state = U @ state
        exp = torch.vdot(state, Z0I @ state).real
        p1 = (1-exp)/2
        return torch.log(p1/(1-p1))
    # ---- batch forward
    def forward(self, X):
        return torch.stack([self.circuit(s) for s in X]).unsqueeze(1)

# ------------ 6) Train loop (chuyển model lên GPU) --------------
def train(model, Xtr, ytr, lr=0.02, epochs=200, batch=64):
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(Xtr)
    for ep in range(epochs):
        idx = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            b = idx[i:i+batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]).squeeze(), ytr[b])
            loss.backward(); opt.step()
        sched.step()

def benchmark(model_ctor, name):
    model = model_ctor()
    t0=time.time(); train(model, X_train, y_train); tr=time.time()-t0
    t1=time.time(); acc=accuracy(model, X_test, y_test); inf=(time.time()-t1)/len(X_test)*1000
    return dict(model=name, params=n_params(model),
                train_s=round(tr,2), infer_ms=round(inf,3), acc=round(acc,3))

# ------------ 7) Main ------------------------------------------
if __name__=="__main__":
    cfg_fc  = benchmark(ClassicalMLNet, "Classical FC")
    cfg_vqc = benchmark(lambda:QuantumMLNet(n_layers=3), "VQC-3layer")
    print("\n=== GPU BENCHMARK ===")
    for r in (cfg_fc, cfg_vqc):
        print(f"{r['model']:13s}|params={r['params']:3d}"
              f"|train={r['train_s']:6.2f}s"
              f"|infer={r['infer_ms']:6.3f} ms/≈"
              f"|acc={r['acc']:.3f}")

