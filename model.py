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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ListModule(object):
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class FullTreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, edge_size, steps=37):
        """
        :param input_size: size of word vectors (default 300)
        :param hidden_size: LSTM hidden size
        :param edge_size: size of edge or relation vectors (default 100)
        :param steps: a number of edge types
        """
        super(FullTreeLSTM, self).__init__()
        self.input_size = input_size
        self.edge_size = edge_size
        self.hidden_size = hidden_size

        self.steps = steps
        # each transition matrix for each edge type
        self.e2h = ListModule(self, 'e2h_')
        for i in range(self.steps):
            self.e2h.append(nn.Linear(self.hidden_size, self.hidden_size, bias=False))

        # TreeLSTM gates
        self.ioux = nn.Linear(self.input_size, 3 * self.hidden_size)
        self.iouh = nn.Linear(self.hidden_size, 3 * self.hidden_size)

        self.fx = nn.Linear(self.input_size, self.hidden_size)
        self.fh = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, tree, inputs, edge_inputs=None):
        """"""
        # iterate over each nodes in the tree
        _ = [self.forward(tree.children[idx], inputs, edge_inputs) for idx in range(tree.num_children)]

        if tree.num_children == 0:  # leaf node
            child_c = Variable(inputs[0].data.new(1, self.hidden_size).fill_(0.))
            child_h = Variable(inputs[0].data.new(1, self.hidden_size).fill_(0.))
            child_w = [0]
        else:  # internal node
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

            child_w = list(map(lambda x: edge_inputs[x.idx].data[0], tree.children))

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h, child_w)

        return tree.state

    def node_forward(self, inputs, child_c, child_h, child_w):
        """"""
        child_h = child_h.unsqueeze(1)
        if child_w is None:
            child_h_sum = torch.sum(child_h, dim=0, keepdim=True)
        else:
            child_h_sum = None
            # each edge type, i.e. child_w[k], have a separate transition matrix self.e2h
            for k in range(child_h.size(0)):
                m = self.e2h[child_w[k]](child_h[k])
                if child_h_sum is None:
                    child_h_sum = m
                else:
                    child_h_sum = torch.cat((child_h_sum, m), 0)

            # sum over values
            child_h_sum = torch.sum(child_h_sum, dim=0, keepdim=True)

        # TreeLSTM gate computation
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h.squeeze(1)) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))

        return c, h


class OtherTreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, edge_size):
        """
        :param input_size: size of word vectors (default 300)
        :param hidden_size: LSTM hidden size
        :param edge_size: size of edge or relation vectors (default 100)
        """
        super(OtherTreeLSTM, self).__init__()
        self.input_size = input_size
        self.edge_size = edge_size
        self.hidden_size = hidden_size

        # use edge vectors for a sigmoid layer
        self.ex = nn.Linear(self.edge_size, self.hidden_size)

        # TreeLSTM gates
        self.ioux = nn.Linear(self.input_size, 3 * self.hidden_size)
        self.iouh = nn.Linear(self.hidden_size, 3 * self.hidden_size)

        self.fx = nn.Linear(self.input_size, self.hidden_size)
        self.fh = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, tree, inputs, edge_inputs=None):
        """"""
        # iterate over each nodes in the tree
        _ = [self.forward(tree.children[idx], inputs, edge_inputs) for idx in range(tree.num_children)]

        if tree.num_children == 0:
            child_c = Variable(inputs[0].data.new(1, self.hidden_size).fill_(0.))
            child_h = Variable(inputs[0].data.new(1, self.hidden_size).fill_(0.))
            if edge_inputs is not None:
                child_w = Variable(inputs[0].data.new(1, self.edge_size).fill_(0.))
            else:
                child_w = None
        else:
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
            if edge_inputs is not None:
                child_w = list(map(lambda x: edge_inputs[x.idx], tree.children))
                child_w = torch.stack(child_w)
            else:
                child_w = None

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h, child_w)

        return tree.state

    def node_forward(self, inputs, child_c, child_h, child_w):
        """"""

        if child_w is None:
            child_h_sum = torch.sum(child_h, dim=0, keepdim=True)
        else:
            # weight for each edge type
            ex = F.sigmoid(self.ex(child_w))
            child_h_sum = torch.sum(torch.mul(ex, child_h), dim=0, keepdim=True)

        # TreeLSTM gate computation
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))

        return c, h


class AddTreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, edge_size):
        """
        :param input_size: size of word vectors (default 300)
        :param hidden_size: LSTM hidden size
        :param edge_size: size of edge vectors (default 100)
        """
        super(AddTreeLSTM, self).__init__()
        self.input_size = input_size
        self.edge_size = edge_size
        self.hidden_size = hidden_size

        # TreeLSTM gates
        self.ioux = nn.Linear(self.input_size + self.edge_size, 3 * self.hidden_size)  # bigger size
        self.iouh = nn.Linear(self.hidden_size, 3 * self.hidden_size)

        self.fx = nn.Linear(self.input_size + self.edge_size, self.hidden_size)
        self.fh = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, tree, inputs, edge_inputs=None):
        """"""
        # iterate over child nodes
        _ = [self.forward(tree.children[idx], inputs, edge_inputs) for idx in range(tree.num_children)]

        if tree.num_children == 0:  # leaf node
            child_c = Variable(inputs[0].data.new(1, self.hidden_size).fill_(0.))
            child_h = Variable(inputs[0].data.new(1, self.hidden_size).fill_(0.))
        else:  # internal node
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        seqs = inputs[tree.idx] if edge_inputs is None else torch.cat((inputs[tree.idx], edge_inputs[tree.idx]), dim=0)

        tree.state = self.node_forward(seqs, child_c, child_h)

        return tree.state

    def node_forward(self, inputs, child_c, child_h):
        """"""
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)

        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))

        return c, h


class mTreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, edge_size):
        """

        :param input_size: size of word vectors (default 300)
        :param hidden_size: LSTM hidden size
        :param edge_size: size of edge vectors (default 100)
        """
        super(mTreeLSTM, self).__init__()
        self.input_size = input_size
        self.edge_size = edge_size
        self.hidden_size = hidden_size

        # transition matrices
        self.wh = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.wmx = nn.Linear(self.edge_size, self.hidden_size, bias=False)
        self.wmh = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # TreeLSTM gates
        self.ioux = nn.Linear(self.input_size, 3 * self.hidden_size)
        self.iouh = nn.Linear(self.hidden_size, 3 * self.hidden_size)

        self.fx = nn.Linear(self.input_size, self.hidden_size)
        self.fh = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, tree, inputs, edge_inputs=None):
        """"""
        # iterate over child nodes
        _ = [self.forward(tree.children[idx], inputs, edge_inputs) for idx in range(tree.num_children)]

        if tree.num_children == 0:  # leaf node
            child_c = Variable(inputs[0].data.new(1, self.hidden_size).fill_(0.))
            child_h = Variable(inputs[0].data.new(1, self.hidden_size).fill_(0.))
            child_w = Variable(inputs[0].data.new(1, self.edge_size).fill_(0.))
        else:  # internal node
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
            child_w = list(map(lambda x: edge_inputs[x.idx], tree.children))
            child_w = torch.stack(child_w)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h, child_w)

        return tree.state

    def node_forward(self, inputs, child_c, child_h, child_w):
        """"""
        child_h = child_h.unsqueeze(1)
        child_w = child_w.unsqueeze(1)
        if child_w is None:
            child_h_sum = torch.sum(child_h.squeeze(1), dim=0, keepdim=True)
        else:
            child_h_sum = None
            for k in range(child_h.size(0)):
                m = self.wmx(child_w[k]) * self.wmh(child_h[k])
                m_k = self.wh(m)
                if child_h_sum is None:
                    child_h_sum = m_k
                else:
                    child_h_sum = torch.cat((child_h_sum, m_k), 0)
            # sum over hidden states from child nodes
            child_h_sum = torch.sum(child_h_sum, dim=0, keepdim=True)

        # TreeLSTM gate computation
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h.squeeze(1)) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))

        return c, h


class ChildSumTreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        :param input_size: size of word vectors (default 300)
        :param hidden_size: LSTM hidden size
        """
        super(ChildSumTreeLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # TreeLSTM gates
        self.ioux = nn.Linear(self.input_size, 3 * self.hidden_size)
        self.iouh = nn.Linear(self.hidden_size, 3 * self.hidden_size)

        self.fx = nn.Linear(self.input_size, self.hidden_size)
        self.fh = nn.Linear(self.hidden_size, self.hidden_size)

    def node_forward(self, inputs, child_c, child_h):
        """"""

        # sum over hidden states of child nodes
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        # TreeLSTM gates computation
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )

        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))

        return c, h

    def forward(self, tree, inputs):
        """"""
        # iterate over child nodes
        _ = [self.forward(tree.children[idx], inputs) for idx in range(tree.num_children)]

        if tree.num_children == 0:  # leaf node
            child_c = Variable(inputs[0].data.new(1, self.hidden_size).fill_(0.))
            child_h = Variable(inputs[0].data.new(1, self.hidden_size).fill_(0.))
        else:  # internal node
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)  # tree.idx from 0

        return tree.state


class NLIClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=None):
        """"""
        super(NLIClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout

        self.fc1 = nn.Linear(4 * self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, lvec, rvec):
        """"""
        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(F.torch.add(lvec, -rvec))
        vec_dist = torch.cat((lvec, rvec, mult_dist, abs_dist), 1)

        if self.dropout is not None:
            out = F.relu(self.fc1(F.dropout(vec_dist, p=self.dropout)))
            out = F.log_softmax(self.fc2(out), dim=1)
        else:
            out = F.relu(self.fc1(vec_dist))
            out = F.log_softmax(self.fc2(out), dim=1)
        return out


class NLITreeLSTM(nn.Module):
    """"""

    def __init__(self, vocab_size, emb_size, mem_size, hidden_size, num_classes, freeze_embed, edge_vocab_size=100,
                 edge_size=20, model='add', dropout=None):
        """ """
        super(NLITreeLSTM, self).__init__()
        # word embedding
        self.embs = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        if freeze_embed:  # no update
            self.embs.weight.requires_grad = False

        # edge embedding
        self.edge_embs = nn.Embedding(edge_vocab_size, edge_size)
        self.model = model
        self.dropout = dropout if dropout != -1 else None

        if model == 'base':
            self.treelstm = ChildSumTreeLSTM(input_size=emb_size, hidden_size=mem_size)
        elif model == 'add':
            self.treelstm = AddTreeLSTM(input_size=emb_size, hidden_size=mem_size, edge_size=edge_size)
        elif model == 'multi':
            self.treelstm = mTreeLSTM(input_size=emb_size, hidden_size=mem_size, edge_size=edge_size)
        elif model == 'other':
            self.treelstm = OtherTreeLSTM(input_size=emb_size, hidden_size=mem_size, edge_size=edge_size)
        elif model == 'full':
            self.treelstm = FullTreeLSTM(input_size=emb_size, hidden_size=mem_size, edge_size=edge_size,
                                         steps=edge_vocab_size)
        else:
            self.treelstm = None
            print('Please select an appropriate model')

        self.classifier = NLIClassifier(input_size=mem_size, hidden_size=hidden_size, num_classes=num_classes,
                                        dropout=self.dropout)

    def forward(self, ltree, linputs, rtree, rinputs, ledges=None, redges=None):
        """"""
        # compute word vectors
        linputs = self.embs(linputs)
        rinputs = self.embs(rinputs)

        if ledges is not None and redges is not None:
            if self.model != 'full':
                ledges = self.edge_embs(ledges)
                redges = self.edge_embs(redges)
            lstate, lhidden = self.treelstm(ltree, linputs, ledges)
            rstate, rhidden = self.treelstm(rtree, rinputs, redges)
        else:
            lstate, lhidden = self.treelstm(ltree, linputs)
            rstate, rhidden = self.treelstm(rtree, rinputs)

        output = self.classifier(lstate, rstate)

        return output
