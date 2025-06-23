# t2_fixed.py  –  PennyLane 0.41 + PyTorch: XOR ≈ 100 %

import math
import pennylane as qml
import torch
from torch import nn

# 1) Dữ liệu XOR
X = torch.tensor([[0.,0.],
                  [0.,1.],
                  [1.,0.],
                  [1.,1.]], dtype=torch.float32)
Y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]], dtype=torch.float32)

# 2) Thiết bị
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

# 3) QNode – dùng parameter-shift để chắc gradient hiện hữu
@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs * math.pi, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 4) TorchLayer
weight_shapes = {"weights": (6, n_qubits, 3)}          # 6 layer × 3 tham số/qubit
qlayer = qml.qnn.TorchLayer(
    circuit,
    weight_shapes,
    init_method=lambda shape: torch.rand(shape) * 2*math.pi
)

# 5) Mạng lai
class HybridXOR(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = qlayer
        self.clf = nn.Linear(2, 1)

    def forward(self, x):
        x = self.q(x)          # [-1,1]
        return self.clf(x)     # logits cho BCEWithLogitsLoss

torch.manual_seed(0)
model = HybridXOR()

# 6) Huấn luyện
opt  = torch.optim.Adam(model.parameters(), lr=0.2)
loss = nn.BCEWithLogitsLoss()

for epoch in range(800):
    opt.zero_grad()
    logits = model(X)
    l = loss(logits, Y)
    l.backward()
    opt.step()

    if (epoch+1) % 80 == 0:
        with torch.no_grad():
            probs = torch.sigmoid(logits).squeeze().tolist()
        print(f"Epoch {epoch+1:4d} | Loss {l.item():.4f} | Probs {probs}")

# 7) Kiểm thử
with torch.no_grad():
    probs = torch.sigmoid(model(X)).squeeze()
    preds = torch.round(probs)
print("\nDự đoán cuối cùng:", preds.tolist())
print("Xác suất:", probs.tolist())

