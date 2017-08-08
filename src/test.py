import torch
import torch.nn as nn
import torch.nn.init as winit
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, inSize, hiddenSize, outSize, dropout=0.2):
        super(Net, self).__init__()
        self.linear = nn.Linear(inSize, hiddenSize)
        winit.constant(self.linear.weight, 1)
        winit.constant(self.linear.bias, 2)
        self.subnet = SubNet(hiddenSize, outSize)

    def forward(self, x):
        m = self.linear(x)
        o = self.subnet.forward(m)
        return o

class SubNet(nn.Module):
    def __init__(self, inSize, outSize, dropout=0.2):
        super(SubNet, self).__init__()
        self.linear = nn.Linear(inSize, outSize)
        winit.constant(self.linear.weight, 2)
        winit.constant(self.linear.bias, 5)

    def forward(self, x):
        return self.linear(x)


if __name__ == '__main__':
    n = Net(3, 2, 1)
    x = torch.Tensor([1.0, 1.0, 1.0])
    x = Variable(x.view(1, 3))
    o = n.forward(x)
    print(o)
