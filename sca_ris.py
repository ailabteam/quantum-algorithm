import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N_RIS = 16  # Number of RIS elements
N_nodes = 5  # Number of ground network nodes
P_tx = 10.0  # Satellite transmit power (W)
sigma2 = 1e-9  # Noise power (W, -90 dBm)
lambda_weight = 0.1  # Weight for delay in objective
max_iter = 50  # Maximum SCA iterations
epsilon = 1e-3  # Convergence threshold

# Generate random Rayleigh channels
h_d = (torch.randn(N_RIS, dtype=torch.complex64) + 1j * torch.randn(N_RIS, dtype=torch.complex64)) / np.sqrt(2)  # Satellite-to-RIS
h_r = (torch.randn(N_RIS, dtype=torch.complex64) + 1j * torch.randn(N_RIS, dtype=torch.complex64)) / np.sqrt(2)  # RIS-to-ground

# Generate random delay matrix for ground network
d = torch.rand(N_nodes, N_nodes) * 9 + 1  # Delays in [1, 10] ms
d = (d + d.t()) / 2  # Symmetric
d.fill_diagonal_(0)  # No delay from node to itself

# Source and destination nodes
s = 0  # Source (ground station receiving from RIS)
t = 4  # Destination

# Compute SNR
def compute_SNR(v, h_d, h_r, P_tx, sigma2):
    Theta = torch.diag(v)
    h = torch.matmul(h_r.conj().t(), torch.matmul(Theta, h_d))
    SNR = (P_tx * torch.abs(h)**2) / sigma2
    return SNR

# Approximate SNR for SCA
def approximate_SNR(v, v_k, h_d, h_r, P_tx, sigma2):
    A = torch.outer(h_r * h_d, (h_r * h_d).conj())
    term1 = 2 * torch.real(torch.matmul(v_k.conj().t(), torch.matmul(A, v)))
    term2 = torch.real(torch.matmul(v_k.conj().t(), torch.matmul(A, v_k)))
    return (P_tx / sigma2) * (term1 - term2)

# Objective function
def objective_function(v, x, h_d, h_r, P_tx, sigma2, d, lambda_weight):
    SNR_approx = approximate_SNR(v, v, h_d, h_r, P_tx, sigma2)
    delay = lambda_weight * torch.sum(d * x)
    return SNR_approx - delay

# Initialize
v = torch.exp(1j * 2 * np.pi * torch.rand(N_RIS))  # Random RIS phases
x = torch.zeros(N_nodes, N_nodes)  # Initialize routing
# Set initial path (e.g., s -> 1 -> t)
path = [s, 1, t]
for i in range(len(path)-1):
    x[path[i], path[i+1]] = 1.0
x = x.requires_grad_()

# Track objective values
objective_history = []

# SCA loop
for k in range(max_iter):
    v_k = v.clone().detach()  # Store current v
    x_k = x.clone().detach()  # Optimize v
    optimizer_v = torch.optim.Adam([v], lr=0.01)
    for _ in range(100):  # Inner optimization for v
        optimizer_v.zero_grad()
        loss = -objective_function(v, x_k, h_d, h_r, P_tx, sigma2, d, lambda_weight)
        loss.backward()
        optimizer_v.step()
        # Normalize phases
        v.data = v.data / torch.abs(v.data)  # |v_n| = 1
    
    # Optimize x
    optimizer_x = torch.optim.Adam([x], lr=0.01)
    for _ in range(100):  # Inner optimization for x
        optimizer_x.zero_grad()
        loss = -objective_function(v, x, h_d, h_r, P_tx, sigma2, d, lambda_weight)
        loss.backward()
        optimizer_x.step()
        # Constrain x to [0, 1]
        x.data = torch.clamp(x.data, 0, 1)
        # Enforce flow constraints
        flow = torch.sum(x, dim=1) - torch.sum(x, dim=0)
        flow[s] = 1.0
        flow[t] = -1.0
        flow[flow != 0] = 0.0  # Approximate flow balance

    # Compute objective
    obj = objective_function(v, x, h_d, h_r, P_tx, sigma2, d, lambda_weight)
    objective_history.append(obj.item())
    
    # Check convergence
    if k > 0 and abs(objective_history[-1] - objective_history[-2]) < epsilon:
        print(f"Converged at iteration {k}")
        break

# Threshold x for binary routing
x_binary = (x > 0.5).float()

# Results
final_SNR = compute_SNR(v, h_d, h_r, P_tx, sigma2).item()
final_delay = torch.sum(d * x_binary).item()
print(f"Final SNR: {final_SNR:.2f}")
print(f"Final Delay: {final_delay:.2f} ms")
print("Routing Matrix (Binary):\n", x_binary.numpy())

# Plot convergence
plt.plot(objective_history)
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.title("SCA Convergence")
plt.show()
