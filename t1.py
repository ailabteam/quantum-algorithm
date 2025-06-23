import pennylane as qml
import torch
from torch import nn

###############################
# 1) Dữ liệu XOR
###############################
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]], dtype=torch.float32)
Y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]], dtype=torch.float32)

###############################
# 2) Thiết bị và QNode
###############################
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def circuit(inputs, weights):
    """inputs -> dữ liệu, weights -> tham số huấn luyện"""
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

###############################
# 3) Chuyển thành Torch layer
###############################
weight_shapes = {"weights": (3, n_qubits)}      # 3 layer entangler
qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

###############################
# 4) Mạng lai Quantum–Classical
###############################
class HybridNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = qlayer                     # output: (batch, 2)
        self.clf = nn.Linear(2, 1)

    def forward(self, x):
        x = self.q(x)                       # giá trị trong [-1, 1]
        x = torch.tanh(self.clf(x))         # squash về [-1, 1]
        return x

model = HybridNN()
opt = torch.optim.Adam(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

###############################
# 5) Huấn luyện
###############################
for epoch in range(200):
    opt.zero_grad()
    preds = model(X)
    loss = loss_fn(preds, Y)
    loss.backward()
    opt.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | Loss = {loss.item():.4f}")

###############################
# 6) Kiểm thử nhanh
###############################
with torch.no_grad():
    logits = model(X)
    preds = torch.round((logits + 1) / 2).squeeze()
print("Dự đoán:", preds.tolist())

