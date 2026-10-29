import torch
from torch import nn
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

class MinhaRede(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MinhaRede, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden = self.relu(self.hidden(x))
        output = self.output(hidden)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)

net = MinhaRede(input_size=2, hidden_size=8, output_size=1).to(device)
entrada = torch.from_numpy(X).float().to(device)
pred = net(entrada)

print(net)
print("Dispositivo da entrada:", entrada.device)
print("Formato da entrada:", entrada.shape)
print("Formato da saída:", pred.shape)
