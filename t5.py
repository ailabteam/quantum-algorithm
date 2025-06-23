# qmnist_torchlayer.py   (Python ≥3.9, PennyLane ≥0.41, PyTorch ≥2.2)

import math, torch
import pennylane as qml
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

#################### 1) DATA ####################
# Chỉ giữ ảnh '0' và '1', resize 4×4 → 16-D vector
preprocess = transforms.Compose([
    transforms.ToTensor(),              # [0,1], shape 1×28×28
    transforms.Resize((4, 4)),          # 1×4×4
    transforms.Lambda(lambda x: x.view(-1))  # 16-D
])

train_full = datasets.MNIST(root="data", train=True, download=True, transform=preprocess)
test_full  = datasets.MNIST(root="data", train=False, download=True, transform=preprocess)

train_idx = [i for i, (_, y) in enumerate(train_full) if y in (0, 1)]
test_idx  = [i for i, (_, y) in enumerate(test_full)  if y in (0, 1)]

train_set = Subset(train_full, train_idx)
test_set  = Subset(test_full,  test_idx)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=32)

#################### 2) QUANTUM LAYER ####################
n_qubits, n_layers = 4, 6
dev = qml.device("default.qubit", wires=n_qubits)      # or "lightning.gpu" for CUDA

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs * math.pi, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))                   # 1 scalar

weight_shapes = {"weights": (n_layers, n_qubits, 3)}

def init_0_2pi(tensor):          # Pennylane ghi trực tiếp vào tensor
    return tensor.uniform_(0.0, 2 * math.pi)

qlayer = qml.qnn.TorchLayer(circuit, weight_shapes, init_method=init_0_2pi)

#################### 3) HYBRID MODEL ####################
class QMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = qlayer           # -> shape (batch, 1)
        self.fc = nn.Linear(1, 1) # -> logit

    def forward(self, x):
        x = self.q(x)             # expectation in [-1,1]
        return self.fc(x).squeeze()

model = QMNIST()
optim = torch.optim.Adam(model.parameters(), lr=0.02)
criterion = nn.BCEWithLogitsLoss()

#################### 4) TRAIN ####################
for epoch in range(5):           # tăng lên 10–15 để đạt ~99 %
    model.train()
    for xb, yb in train_loader:
        optim.zero_grad()
        logits = model(xb)            # shape (batch,)
        loss = criterion(logits, yb.float())
        loss.backward()
        optim.step()
    print(f"Epoch {epoch+1}: loss {loss.item():.4f}")

#################### 5) TEST ####################
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        preds = torch.sigmoid(model(xb)) > 0.5
        correct += (preds.int() == yb).sum().item()
        total   += yb.size(0)
print(f"Test accuracy: {100*correct/total:.2f}%")

