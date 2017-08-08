import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import copy
import time
import math

import numpy as np
from data_loader import load_data
from decomposable_attention import DecomposableAttentionModel as DAM


def append_zeros(sent, embedSize):
    maxlen = -1
    for i in range(len(sent)):
        if maxlen < len(sent[i]):
            maxlen = len(sent[i])

    for i in range(len(sent)):
        curlen = len(sent[i])
        while curlen < maxlen:
            sent[i].append(np.zeros(embedSize))
            curlen += 1


def get_batch(data, label, batch_idx, embedSize):
    raw = copy.deepcopy(data[batch_idx])

    sent1 = raw[0::2]
    sent2 = raw[1::2]

    append_zeros(sent1, embedSize)
    append_zeros(sent2, embedSize)

    sent1 = torch.from_numpy(np.asarray(sent1))
    sent2 = torch.from_numpy(np.asarray(sent2))
    label = torch.from_numpy(np.asarray(label[batch_idx]))
    return sent1.float(), sent2.float(), label

def get_learning_rate(epoch, lr):
    decay = 0
    if epoch != 1:
        decay = epoch - 1

    return lr * math.pow(0.7, decay)

def train(data, embedSize, hiddenSize, outSize):
    print('embedSize:', embedSize, 'hiddenSize:', hiddenSize)
    batch_train_data = data[0]
    batch_train_label = data[1]
    batch_valid_data = data[2]
    batch_valid_label = data[3]

    net = DAM(embedSize, hiddenSize, outSize, 0.05)
    net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)

    num_train_batches = len(batch_train_data)
    num_valid_batches = len(batch_valid_data)

    for epoch in range(30):
        print('======epoch ' + str(epoch + 1) + '======')
        cur_lr = get_learning_rate(epoch + 1, 0.001)
        optimizer = optim.Adagrad(net.parameters(), lr=cur_lr, weight_decay=1e-5)

        # train
        running_loss = 0.0
        total = 0
        correct = 0
        train_start = time.time()
        timer = 0
        tpSum = 0
        fpSum = 0
        fnSum = 0
        for i in range(num_train_batches):
            batch_start = time.time()
            optimizer.zero_grad()
            sent1_batch, sent2_batch, label =\
                get_batch(batch_train_data, batch_train_label, i, embedSize)
            # print(sent1_batch.size())
            # print(sent2_batch.size())
            sent1_batch = sent1_batch.cuda()
            sent2_batch = sent2_batch.cuda()
            label = label.cuda()
            output = net(Variable(sent1_batch), Variable(sent2_batch))
            loss = criterion(output, Variable(label))
            loss.backward()

            grad_norm = 0.
            para_norm = 0.
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    grad_norm += m.weight.grad.data.norm() ** 2
                    para_norm += m.weight.data.norm() ** 2
                    if m.bias:
                        grad_norm += m.bias.grad.data.norm() ** 2
                        para_norm += m.bias.data.norm() ** 2

            grad_norm ** 0.5
            para_norm ** 0.5

            max_grad_norm = 5
            shrinkage = max_grad_norm / grad_norm
            if shrinkage < 1 :
                for m in net.modules():
                    # print(m)
                    if isinstance(m, nn.Linear):
                        m.weight.grad.data = m.weight.grad.data * shrinkage
                        m.bias.grad.data = m.bias.grad.data * shrinkage

            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            batchCorrect = (predicted == label).sum()
            correct += batchCorrect

            tp = torch.dot(predicted.cpu(), label.cpu())
            fn = label.sum() - tp
            fp = predicted.cpu().sum() - tp
            tpSum += tp
            fpSum += fp
            fnSum += fn

            # tempLabelList = []
            # tempPredList = []
            # for value in label:
            #     tempLabelList.append(int(value))
            # for value in predicted:
            #     tempPredList.append(value)
            # print(tempLabelList)
            # print(tempPredList)

            running_loss += loss.data[0]
            batch_end = time.time()
            timer += (batch_end - batch_start)
            if (i + 1) % 20 == 0:
                prec = tpSum / (tpSum + fpSum)
                recall = tpSum / (tpSum + fnSum)
                f1 = 2 * prec * recall / (prec + recall)
                print('epoch: %d  batch: %d|%d  loss: %.6f  train acc: %.3f  prec: %.3f  recall: %.3f  f1: %.3f  runtime: %.2f' % (epoch + 1, i + 1, num_train_batches, running_loss / 5 / batchSize, correct / total, prec, recall, f1, timer))
                running_loss = 0.0
                timer = 0
        train_end = time.time()

        # valid
        valid_correct = 0
        valid_total = 0
        for i in range(num_valid_batches):
            sent1_batch, sent2_batch, label =\
                get_batch(batch_valid_data, batch_valid_label, i, embedSize)
            sent1_batch = sent1_batch.cuda()
            sent2_batch = sent2_batch.cuda()
            label = label.cuda()
            output = net(Variable(sent1_batch), Variable(sent2_batch))
            _, predicted = torch.max(output.data, 1)
            valid_total += label.size(0)
            valid_correct += (predicted == label).sum()
        print('train time: %.3f  valid acc: %.3f' % (train_end - train_start, valid_correct/ valid_total))

if __name__ == '__main__':
    trainfile = '../data/products/home/train_raw.txt'
    validfile = '../data/products/home/validation_raw.txt'
    wordfile = '../data/glove_word_list.txt'
    embedfile = '../data/glove_word_embed_vec.npy'
    batchSize = 64
    data = load_data(trainfile, validfile, wordfile, embedfile, batchSize)
    train(data, 300, 300, 2)
