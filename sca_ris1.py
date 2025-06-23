import torch, torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Tham số --------------------
N_RIS   = 16        # số phần tử RIS
N_nodes = 5         # số nút mạng mặt đất
P_tx    = 10.0      # (W)
sigma2  = 1e-9      # (W) ~ -90 dBm
λ       = 0.1       # trọng số delay
max_iter, eps = 50, 1e-3
μ_flow  = 10.0      # hệ số phạt flow-constraint  ★

torch.manual_seed(0)

# -------------------- Kênh ngẫu nhiên --------------------
h_d = (torch.randn(N_RIS, dtype=torch.complex64) + 1j*torch.randn(N_RIS, dtype=torch.complex64))/np.sqrt(2)
h_r = (torch.randn(N_RIS, dtype=torch.complex64) + 1j*torch.randn(N_RIS, dtype=torch.complex64))/np.sqrt(2)

# delay giữa các nút (ms)
d = torch.rand(N_nodes, N_nodes)*9 + 1
d = (d + d.t())/2
d.fill_diagonal_(0)

s, t = 0, 4          # nguồn & đích

# -------------------- Hàm tiện ích --------------------
def compute_SNR(v_cplx):
    """SNR đúng (không xấp xỉ)."""
    h = (h_r.conj() @ (v_cplx * h_d))
    return (P_tx * torch.abs(h)**2) / sigma2          # scalar

def SNR_lin(v, v_k):
    """Xấp xỉ tuyến tính hóa SNR quanh v_k (SCA)."""
    g = h_r * h_d                                     # N_RIS vector
    A = torch.outer(g, g.conj())                      # N×N
    term1 = 2*torch.real(v_k.conj() @ (A @ v))
    term2 = torch.real(v_k.conj() @ (A @ v_k))
    return (P_tx / sigma2)*(term1 - term2)

def objective(v, v_k, x):
    """Hàm mục tiêu SCA: SNR_lin(v;v_k) - λ*delay - penalty(flow)."""
    delay = torch.sum(d * x)
    # ràng buộc lưu lượng: Σ_out - Σ_in = b
    flow = torch.sum(x, 1) - torch.sum(x, 0)
    b    = torch.zeros_like(flow); b[s], b[t] = 1, -1
    penalty = μ_flow * torch.sum((flow - b)**2)       # ★
    return SNR_lin(v, v_k) - λ*delay - penalty

# -------------------- Khởi tạo biến --------------------
# 1. Tham số pha (real) → complex weight
phi = nn.Parameter(2*np.pi*torch.rand(N_RIS))          # ★
def v_complex(phi): return torch.exp(1j*phi)           # twiddle function

# 2. Ma trận nối đường (continuous 0–1)
x   = nn.Parameter(torch.zeros(N_nodes, N_nodes))
# path tạm s-1-t
path = [s,1,t]
for i in range(len(path)-1): x.data[path[i], path[i+1]] = 1

# -------------------- Vòng lặp SCA --------------------
obj_hist = []
opt_phi  = torch.optim.Adam([phi], lr=0.02)
opt_x    = torch.optim.Adam([x],   lr=0.02)

for k in range(max_iter):
    phi_k = phi.detach().clone()                       # lưu v_k
    # ----- (a) cập nhật phi (v) -----
    for _ in range(100):
        opt_phi.zero_grad()
        loss = -objective(v_complex(phi), v_complex(phi_k), x)
        loss.backward()
        opt_phi.step()
    # ----- (b) cập nhật x ----------
    for _ in range(100):
        opt_x.zero_grad()
        loss = -objective(v_complex(phi), v_complex(phi), x)   # v cố định
        loss.backward()
        opt_x.step()
        x.data.clamp_(0,1)                              # giữ [0,1]

    # Theo dõi hội tụ
    obj_val = objective(v_complex(phi), v_complex(phi), x).item()
    obj_hist.append(obj_val)
    if k>0 and abs(obj_hist[-1]-obj_hist[-2]) < eps:
        print(f"Converged at iter {k}")
        break

# -------------------- Kết quả --------------------
x_bin  = (x.data>0.5).float()
print(f"Final SNR   : {compute_SNR(v_complex(phi)).item():.2f}")
print(f"Final Delay : {torch.sum(d * x_bin).item():.2f} ms")
print("Routing (binary):\n", x_bin.numpy())

# -------------------- Đồ thị hội tụ --------------------
plt.plot(obj_hist); plt.xlabel("Outer iteration"); plt.ylabel("Objective")
plt.title("SCA convergence"); plt.show()

