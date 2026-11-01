import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, tam_entrada, tam_feature, tam_saida):
        super().__init__()

        self.tam_entrada = tam_entrada
        self.tam_feature = tam_feature
        self.tam_saida = tam_saida

        self.rnn = nn.RNNCell(tam_entrada, tam_feature)
        self.linear = nn.Linear(tam_feature, tam_saida)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, nome):
        hidden = torch.zeros(
            1,
            self.tam_feature,
            device=nome.device
        )

        for letra in nome:
            letra = letra.unsqueeze(0)
            hidden = self.rnn(letra, hidden)

        saida = self.linear(hidden)
        saida = self.softmax(saida)

        return saida
