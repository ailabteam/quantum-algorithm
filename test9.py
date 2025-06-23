import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#   RIS-Assisted Satellite Link: Joint Phase & Routing Design
#   Successive Convex Approximation (SCA) with Flow Penalty
# ============================================================
#   • Capacity metric  :  C = log2(1+SNR)  (bit/s/Hz)
#   • Delay metric     :  sum(d * x)       (ms)
#   • Objective        :  𝑓 = C - λ·delay - µ·‖flow-b‖²
# ============================================================

# ---------- Reproducibility ----------
torch.manual_seed(0)
np.random.seed(0)

# ---------- System parameters ----------
N_RIS   = 16          # RIS elements
N_nodes = 5           # ground nodes
P_tx    = 10.0        # transmit power (W)
sigma2  = 1e-9        # noise power (W)   ~ -90 dBm
lambda_delay = 1.0    # weight for delay term
mu_flow     = 50.0    # penalty for flow-constraint

# Optimisation hyper-parameters
max_outer   = 50      # outer SCA iterations
inner_v     = 100     # inner steps for phi
inner_x     = 100     # inner steps for x
tol         = 1e-3    # convergence threshold

# Link budget: include large path-loss (in dB, power)
path_loss_dB = 110     # tweak to match realistic SNR
atten = 10 ** (-path_loss_dB / 20)  # amplitude attenuation

# ---------- Channel generation ----------
# Rayleigh small-scale fading, unit variance → apply path-loss
h_d = (torch.randn(N_RIS, dtype=torch.complex64) + 1j*torch.randn(N_RIS, dtype=torch.complex64)) / np.sqrt(2)
h_r = (torch.randn(N_RIS, dtype=torch.complex64) + 1j*torch.randn(N_RIS, dtype=torch.complex64)) / np.sqrt(2)
h_d *= atten
h_r *= atten

# ---------- Delay matrix between ground nodes ----------
d = torch.rand(N_nodes, N_nodes) * 9 + 1  # [1,10] ms
d = (d + d.t()) / 2
d.fill_diagonal_(0)

# Source & destination indices
s, t = 0, 4

# ---------- Helper functions ----------

def v_complex(phi: torch.Tensor) -> torch.Tensor:
    """Convert phase vector φ (real) → unit-modulus complex v."""
    return torch.exp(1j * phi)


def exact_SNR(v: torch.Tensor) -> torch.Tensor:
    """Exact end-to-end SNR (scalar)."""
    h = h_r.conj() @ (v * h_d)
    return (P_tx * torch.abs(h)**2) / sigma2


def approx_SNR(v: torch.Tensor, v_k: torch.Tensor) -> torch.Tensor:
    """First-order (affine) approximation of SNR around v_k for SCA."""
    g = h_r * h_d                     # element-wise product
    A = torch.outer(g, g.conj())
    term1 = 2 * torch.real(v_k.conj() @ (A @ v))
    term2 = torch.real(v_k.conj() @ (A @ v_k))
    return (P_tx / sigma2) * (term1 - term2)


def objective(v: torch.Tensor, v_k: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """SCA surrogate objective to maximise."""
    # capacity surrogate (bit/s/Hz)
    snr_lin = torch.clamp(approx_SNR(v, v_k), min=0)
    capacity_approx = torch.log2(1 + snr_lin)

    # delay term (ms)
    delay = torch.sum(d * x)

    # flow constraint penalty
    flow = torch.sum(x, dim=1) - torch.sum(x, dim=0)
    b = torch.zeros_like(flow)
    b[s], b[t] = 1.0, -1.0
    penalty = mu_flow * torch.sum((flow - b) ** 2)

    return capacity_approx - lambda_delay * delay - penalty

# ---------- Optimisation variables ----------
phi = nn.Parameter(2 * np.pi * torch.rand(N_RIS))   # phase vector (real)
x   = nn.Parameter(torch.zeros(N_nodes, N_nodes))   # soft routing matrix ∈ [0,1]

# simple initial path: s → 1 → t
init_path = [s, 1, t]
for i in range(len(init_path) - 1):
    x.data[init_path[i], init_path[i+1]] = 1.0

opt_phi = torch.optim.Adam([phi], lr=0.01)
opt_x   = torch.optim.Adam([x],   lr=0.01)

# ---------- SCA loop ----------
obj_hist = []

for k in range(max_outer):
    # ----- (a) update phase phi -----
    phi_k = phi.detach().clone()
    for _ in range(inner_v):
        opt_phi.zero_grad()
        loss = -objective(v_complex(phi), v_complex(phi_k), x)
        loss.backward()
        opt_phi.step()

    # ----- (b) update routing x ------
    for _ in range(inner_x):
        opt_x.zero_grad()
        loss = -objective(v_complex(phi), v_complex(phi), x)  # v fixed
        loss.backward()
        opt_x.step()
        # keep x within [0,1]
        x.data.clamp_(0, 1)

    # convergence check
    obj_val = objective(v_complex(phi), v_complex(phi), x).item()
    obj_hist.append(obj_val)
    print(f"Iter {k:02d}  obj = {obj_val:.4f}")
    if k > 0 and abs(obj_hist[-1] - obj_hist[-2]) < tol:
        print(f"Converged at outer iter {k}\n")
        break

# ---------- Post-processing ----------
v_opt   = v_complex(phi).detach()
x_cont  = x.detach()
x_bin   = (x_cont > 0.5).float()

snr_fin = exact_SNR(v_opt).item()
cap_fin = np.log2(1 + snr_fin)
delay_fin = torch.sum(d * x_bin).item()

print("==================== FINAL RESULT ====================")
print(f"SNR            : {snr_fin:.2f}  ({10*np.log10(snr_fin):.2f} dB)")
print(f"Capacity       : {cap_fin:.2f} bit/s/Hz")
print(f"Total delay    : {delay_fin:.2f} ms")
print("Routing (binary):\n", x_bin.numpy())
print("======================================================")

# ---------- Convergence plot ----------
plt.figure()
plt.plot(obj_hist)
plt.xlabel('Outer iteration')
plt.ylabel('Objective')
plt.title('SCA convergence')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

