import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class HighWayMLPLayer(nn.Module):
    def __init__(self, inSize, outSize, dropout=0.05):
        super(HighWayMLPLayer, self).__init__()

        self.linear = nn.Linear(inSize, outSize)
        self.gate = nn.Linear(inSize, outSize)
        self.bn = nn.BatchNorm1d(outSize)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        linear = self.bn(self.linear(self.drop(x)))
        activation = nn.ReLU(linear)

        gate = F.sigmoid(self.bn(self.gate(self.drop(x))))

        return gate * activation + (1 - gate) * x

if __name__ == '__main__':
    net = HighWayMLPLayer(100, 100)
    print(net)
