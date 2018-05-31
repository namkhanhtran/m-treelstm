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
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

import Constants as C
from tree import Tree


class NLIDataset(data.Dataset):
    """"""

    def __init__(self, path, vocab, num_classes, edge_vocab=None, treetype=""):
        super(NLIDataset, self).__init__()

        self.vocab = vocab
        self.edge_vocab = edge_vocab
        self.num_classes = num_classes

        self.lsentences = self.read_sentences(os.path.join(path, 'a%s.toks' % treetype))
        self.rsentences = self.read_sentences(os.path.join(path, 'b%s.toks' % treetype))

        self.ltrees = self.read_trees(os.path.join(path, 'a%s.parents' % treetype))
        self.rtrees = self.read_trees(os.path.join(path, 'b%s.parents' % treetype))

        self.ledges = self.read_edges(os.path.join(path, 'a%s.rels' % treetype))
        self.redges = self.read_edges(os.path.join(path, 'b%s.rels' % treetype))

        self.labels = self.read_labels(os.path.join(path, 'nli_score.txt'))

        self.size = self.labels.size(0)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        ltree = deepcopy(self.ltrees[index])
        rtree = deepcopy(self.rtrees[index])
        lsent = deepcopy(self.lsentences[index])
        rsent = deepcopy(self.rsentences[index])
        label = deepcopy(self.labels[index])

        ledge = deepcopy(self.ledges[index])
        redge = deepcopy(self.redges[index])

        return ltree, lsent, rtree, rsent, label, ledge, redge

    def read_sentences(self, filename):
        with open(filename, 'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convert_to_idx(line.split(), C.UNK_WORD)
        return torch.LongTensor(indices)

    def read_edges(self, filename):
        with open(filename, 'r') as f:
            edges = [self.read_edge(line) for line in tqdm(f.readlines())]
        return edges

    def read_edge(self, line):
        indices = self.edge_vocab.convert_to_idx(line.split(), C.UNK_WORD)
        return torch.LongTensor(indices)

    def read_trees(self, filename):
        with open(filename, 'r') as f:
            trees = [self.read_tree(line) for line in tqdm(f.readlines())]
        return trees

    def read_tree(self, line):
        parents = list(map(int, line.split()))
        trees = dict()
        root = None
        for i in range(1, len(parents) + 1):
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx - 1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    def read_labels(self, filename):
        label_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        with open(filename, 'r') as f:
            labels = list(
                map(lambda x: label_map[x.strip().lower()] if x.strip().lower() in label_map else 0, f.readlines()))
            labels = torch.Tensor(labels)
        return labels
