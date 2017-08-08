import os
import numpy as np


def parse_file(infile, batchSize, words, embeds):
    embed_size = embeds.shape[1]
    fin = open(infile, 'r')
    lines = fin.readlines()
    batch_data = [[]]
    batch_label = [[]]
    count = 0
    for i, l in enumerate(lines):
        l = l.strip()
        if (i + 1) % 20000 == 0:
            print('loading: ' + str(i + 1))
        if (i + 1) % 3 == 0:
            batch_label[-1].append(int(l))
            if count % batchSize == 0:
                batch_data.append([])
                batch_label.append([])
        else:
            l = l.split()
            sent = []
            for word in l:
                if word in words:
                    sent.append(embeds[words[word]])
                else:
                    sent.append(np.zeros(embed_size))
            batch_data[-1].append(sent)
            count += 1
    batch_data.pop(-1)
    batch_label.pop(-1)

    # for i in range(len(batch_data)):
    #     maxlen = -1
    #     for j in range(len(batch_data[i])):
    #         if maxlen < len(batch_data[i][j]):
    #             maxlen = len(batch_data[i][j])
    #     print(maxlen)
    #     strlen = ''
    #     for j in range(len(batch_data[i])):
    #         curlen = len(batch_data[i][j])
    #         # while curlen < maxlen:
    #         #     batch_data[i][j].append(np.zeros(embed_size))
    #         #     curlen += 1
    #         strlen += str(curlen) + ' '
    #     print(strlen)
    # print(count)
    # print(len(batch_data))
    # print(len(batch_data[0]))
    # # lenstr = ''
    # # for i in range(len(batch_data)):
    # #     lenstr += str(len(batch_data[i])) + ' '
    # # print(lenstr)
    # print(len(batch_label))
    # print(len(batch_label[0]))
    # # lenstr = ''
    # # for i in range(len(batch_label)):
    # #     lenstr += str(len(batch_label[i])) + ' '
    # # print(lenstr)
    return batch_data, batch_label


def load_data(trainfile, validfile, wordfile, embedfile, batchSize):
    embeds = np.load(embedfile)
    print('Finish loading word embeds: ' + str(embeds.shape))

    words = {}
    fwin = open(wordfile, 'r')
    lines = fwin.readlines()
    for i, l in enumerate(lines):
        l = l.strip()
        words[l] = i
    print('Finish loading words: ' + str(len(words)))

    batch_train_data, batch_train_label = parse_file(
                        trainfile, batchSize, words, embeds)
    batch_valid_data, batch_valid_label = parse_file(
                        validfile, batchSize, words, embeds)

    return [batch_train_data, batch_train_label,
            batch_valid_data, batch_valid_label]

if __name__ == '__main__':
    trainfile = '../data/products/home/train_raw.txt'
    validfile = '../data/products/home/validation_raw.txt'
    wordfile = '../data/glove_word_list.txt'
    embedfile = '../data/glove_word_embed_vec.npy'
    batchSize = 60
    load_data(trainfile, validfile, wordfile, embedfile, batchSize)
