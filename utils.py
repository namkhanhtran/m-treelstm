"""
Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
A copy of the License is located at

http://www.apache.org/licenses/LICENSE-2.0

or in the "license" file accompanying this file. This file is distributed
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
express or implied. See the License for the specific language governing
permissions and limitations under the License.
"""
import os
import torch
from vocab import Vocab
import math
from tqdm import tqdm


def load_word_vectors(path):
    if os.path.isfile(path + '.pth') and os.path.isfile(path + '.vocab'):
        vectors = torch.load(path + '.pth')
        vocab = Vocab(filename=path + '.vocab')
        return vocab, vectors

    print("==> Preprocessing glove file!")
    count = sum(1 for _ in open(path + '.txt'))
    with open(path + '.txt', 'r') as f:
        contents = f.readline().rstrip('\n').split(' ')
        dim = len(contents[1:])
    words = [None] * count
    vectors = torch.zeros(count, dim)
    with open(path + '.txt', 'r') as f:
        idx = 0
        for line in tqdm(f):
            contents = line.rstrip('\n').split(' ')
            words[idx] = contents[0]
            vectors[idx] = torch.Tensor(list(map(float, contents[1:])))
            idx += 1
    with open(path + '.vocab', 'w') as f:
        for word in words:
            f.write(word + '\n')
    vocab = Vocab(filename=path + '.vocab')
    torch.save(vectors, path + '.pth')
    return vocab, vectors


def build_vocab(filenames, vocabfile, lowercase=False):
    vocab = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                tokens = line.rstrip('\n').split()
                vocab |= set(tokens)
    with open(vocabfile, 'w') as f:
        for token in sorted(vocab):
            f.write(token + '\n')


def map_label_to_target(label):
    target = torch.LongTensor(1)
    target[0] = int(label)
    return target


def map_float_label_to_target(label, num_classes=5):
    target = torch.zeros(1, num_classes)
    ceil = int(math.ceil(label))
    floor = int(math.floor(label))
    if ceil == floor:
        target[0][floor - 1] = 1
    else:
        target[0][floor - 1] = ceil - label
        target[0][ceil - 1] = label - floor
    return target


def count_params(model):
    print("__param count_")
    params = list(model.parameters())
    total_params = 0
    for p in params:
        if p.requires_grad:
            total_params += p.numel()
            print(p.size())
    print("total", total_params)
    print('______________')
