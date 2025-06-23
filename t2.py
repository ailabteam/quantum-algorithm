# t1_fixed.py – PennyLane + PyTorch XOR 100 %

import pennylane as qml
import torch
from torch import nn

###############################
# 1) Cấu hình & dữ liệu XOR
###############################
torch.manual_seed(42)  # tái lập kết quả

X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
], dtype=torch.float32)

Y = torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.]
], dtype=torch.float32)

###############################
# 2) Thiết bị lượng tử
###############################
n_qubits = 2

# Chạy CPU mặc định; nếu có CUDA, đổi thành "lightning.gpu"
dev = qml.device("default.qubit", wires=n_qubits)

###############################
# 3) QNode: tên đối số đầu vào PHẢI là `inputs`
###############################
@qml.qnode(dev, interface="torch")
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

###############################
# 4) Chuyển thành lớp PyTorch
###############################
weight_shapes = {"weights": (4, n_qubits)}  # 4 block entangler
qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

###############################
# 5) Mạng lai Quantum + Classical
###############################
class HybridNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = qlayer          # output shape: (batch, 2)
        self.clf = nn.Linear(2, 1)

    def forward(self, x):
        x = self.q(x)            # giá trị nằm trong [-1, 1]
        x = self.clf(x)          # logits, KHÔNG sigmoid/tanh
        return x                 # BCEWithLogitsLoss sẽ lo sigmoid

model = HybridNN()

###############################
# 6) Huấn luyện
###############################
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(400):
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, Y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 40 == 0:
        print(f"Epoch {epoch+1:3d} | Loss = {loss.item():.4f}")

###############################
# 7) Kiểm thử & in kết quả
###############################
with torch.no_grad():
    logits = model(X)
    probs = torch.sigmoid(logits)
    preds = torch.round(probs).squeeze()

print("\nXOR dự đoán:", preds.tolist())
print("Xác suất   :", probs.squeeze().tolist())

