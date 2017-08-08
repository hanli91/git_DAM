import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

class DecomposableAttentionModel(nn.Module):
    def __init__(self, embedSize, hiddenSize, outSize, dropout=0.2):
        super(DecomposableAttentionModel, self).__init__()
        self.attend = AttentionLayer(embedSize, hiddenSize, dropout)
        self.compare = CompareLayer(embedSize, hiddenSize, dropout)
        self.aggre = AggregateLayer(hiddenSize, outSize, dropout)

    def forward(self, sent1_batch, sent2_batch):
        att1, att2 = self.attend(sent1_batch, sent2_batch)
        cmp1, cmp2 = self.compare(att1, att2, sent1_batch, sent2_batch)
        return self.aggre(cmp1, cmp2)


class AttentionLayer(nn.Module):
    def __init__(self, embedSize, hiddenSize, dropout=0.2):
        super(AttentionLayer, self).__init__()
        self.embedSize = embedSize
        self.hiddenSize = hiddenSize
        self.mlp_f = MLPLayer(embedSize, hiddenSize, dropout)


    def forward(self, sent1_batch, sent2_batch):
        len1 = sent1_batch.size(1)
        len2 = sent2_batch.size(1)

        f1 = self.mlp_f(sent1_batch.view(-1, self.embedSize))
        f2 = self.mlp_f(sent2_batch.view(-1, self.embedSize))

        f1 = f1.view(-1, len1, self.hiddenSize)
        f2 = f2.view(-1, len2, self.hiddenSize)

        e1 = torch.bmm(f1, torch.transpose(f2, 1, 2))
        w1 = F.softmax(e1.view(-1, len2)).view(-1, len1, len2)

        e2 = torch.transpose(e1.contiguous(), 1, 2).contiguous()
        w2 = F.softmax(e2.view(-1, len1)).view(-1, len2, len1)

        att1 = torch.bmm(w1, sent2_batch)
        att2 = torch.bmm(w2, sent1_batch)

        return att1, att2


class CompareLayer(nn.Module):
    def __init__(self, embedSize, hiddenSize, dropout=0.2):
        super(CompareLayer, self).__init__()
        self.hiddenSize = hiddenSize
        self.embedSize = embedSize
        self.mlp_g = MLPLayer(embedSize * 2, hiddenSize, dropout)

    def forward(self, att1_batch, att2_batch, sent1_batch, sent2_batch):
        merge1 = torch.cat((att1_batch, sent1_batch), 2)
        merge2 = torch.cat((att2_batch, sent2_batch), 2)
        len1 = sent1_batch.size(1)
        len2 = sent2_batch.size(1)

        cmp1 = self.mlp_g(merge1.view(-1, 2 * self.embedSize)).view(-1, len1, self.hiddenSize)
        cmp2 = self.mlp_g(merge2.view(-1, 2 * self.embedSize)).view(-1, len2, self.hiddenSize)

        return cmp1, cmp2


class AggregateLayer(nn.Module):
    def __init__(self, hiddenSize, outSize, dropout=0.2):
        super(AggregateLayer, self).__init__()
        self.mlp_h = MLPLayer(2 * hiddenSize, hiddenSize, dropout)
        self.final = nn.Linear(hiddenSize, outSize, bias=True)
        # self.log_prob = nn.LogSoftmax()

    def forward(self, cmp1, cmp2):
        sent1_sum = torch.squeeze(torch.sum(cmp1, 1), 1)
        sent2_sum = torch.squeeze(torch.sum(cmp2, 1), 1)

        combine = torch.cat((sent1_sum, sent2_sum), 1)
        h = self.mlp_h(combine)
        v = self.final(h)
        # return self.log_prob(v)
        return v


class MLPLayer(nn.Module):
    def __init__(self, inSize, outSize, dropout=0.2):
        super(MLPLayer, self).__init__()
        linear1 = nn.Linear(inSize, outSize)
        linear2 = nn.Linear(outSize, outSize)
        # init.normal(linear1.weight, 0, 0.01)
        # init.constant(linear1.bias, 0.01)
        # init.normal(linear2.weight, 0, 0.01)
        # init.constant(linear2.bias, 0.01)

        self.seq = nn.Sequential(nn.Dropout(dropout), linear1, nn.ReLU(),
                                 nn.Dropout(dropout), linear2, nn.ReLU())

    def forward(self, x):
        return self.seq(x)


if __name__ == '__main__':
    dam = DecomposableAttentionModel(300, 20, 1)
    print(dam)
