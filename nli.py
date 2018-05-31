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
import random
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import Constants as C

from vocab import Vocab
from metrics import Metrics
from utils import load_word_vectors, build_vocab

from dataset import NLIDataset
from trainer import Trainer

from model import NLITreeLSTM


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch TreeLSTM for NLI")
    parser.add_argument('--data', default='data/sick/',
                        help='path to dataset')
    parser.add_argument('--glove', default='data/glove/',
                        help='directory with GLOVE embeddings')
    parser.add_argument('--save', default='checkpoints/',
                        help='directory to save checkpoints in')
    parser.add_argument('--expname', type=str, default='test',
                        help='Name to identify experiment')

    parser.add_argument('--word_size', default=300, type=int,
                        help="word embedding size (default: 300)")
    parser.add_argument('--edge_size', default=100, type=int,
                        help="relation embedding size (default: 100)")
    parser.add_argument('--mem_size', default=150, type=int,
                        help="treeLSTM memory size (default: 150)")
    parser.add_argument('--hidden_size', default=50, type=int,
                        help="classifier hidden size (default: 50)")
    parser.add_argument('--num_classes', default=3, type=int,
                        help="number of classes (default: 3)")
    parser.add_argument('--freeze_embed', action='store_true', default=False,
                        help='Freeze word embeddings')

    parser.add_argument('--batch_size', default=25, type=int,
                        help="batchsize for optimizer updates (default: 25)")
    parser.add_argument('--epochs', default=10, type=int,
                        help="number of total epochs to run (default: 10)")
    parser.add_argument('--lr', default=1e-3, type=float,
                        help="initial learning rate (default: 0.001)")
    parser.add_argument('--weight_decay', default=1e-4, type=int,
                        help="weight decay (default: 0.0001)")
    parser.add_argument('--dropout', default=0.5, type=float,
                        help="use dropout (default: 0.5), -1: no dropout")
    parser.add_argument('--emblr', default=0.1, type=float,
                        help="initial embedding learning rate (default: 0.1)")
    parser.add_argument('--optim', default="adam",
                        help="optimizer (default: adam)")

    parser.add_argument('--seed', default=123, type=int,
                        help="random seed (default: 123)")
    parser.add_argument('--model', default="multi",
                        help="model type:add, multi, full, base, other")
    parser.add_argument('--treetype', default="dep",
                        help="types of parsed tree: dep, amr")

    return parser.parse_args()


def main():
    global args
    args = parse_args()
    # logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    train_dir = os.path.join(args.data, 'train/')
    dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'test/')

    treetype = ".amr" if args.treetype == 'amr' else ""

    # write unique words from all token files
    nli_vocab_file = os.path.join(args.data, 'nli%s.vocab' % treetype)
    if not os.path.isfile(nli_vocab_file):
        token_files_b = [os.path.join(split, 'b%s.toks' % treetype) for split in [train_dir, dev_dir, test_dir]]
        token_files_a = [os.path.join(split, 'a%s.toks' % treetype) for split in [train_dir, dev_dir, test_dir]]
        token_files = token_files_a + token_files_b
        nli_vocab_file = os.path.join(args.data, 'nli%s.vocab' % treetype)
        build_vocab(token_files, nli_vocab_file)

    vocab = Vocab(filename=nli_vocab_file, data=[C.PAD_WORD, C.UNK_WORD])

    nli_vocab_file = os.path.join(args.data, 'nli%s.edge.vocab' % treetype)
    if not os.path.isfile(nli_vocab_file):
        token_files_b = [os.path.join(split, 'b%s.rels' % treetype) for split in [train_dir, dev_dir, test_dir]]
        token_files_a = [os.path.join(split, 'a%s.rels' % treetype) for split in [train_dir, dev_dir, test_dir]]
        token_files = token_files_a + token_files_b
        nli_vocab_file = os.path.join(args.data, 'nli%s.edge.vocab' % treetype)
        build_vocab(token_files, nli_vocab_file)

    edge_vocab = Vocab(filename=nli_vocab_file)

    # load SICK dataset splits
    train_file = os.path.join(args.data, 'nli%s.train.pth' % treetype)
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = NLIDataset(train_dir, vocab, args.num_classes, edge_vocab, treetype=treetype)
        torch.save(train_dataset, train_file)
    logger.debug('==> Size of train data   : %d ' % len(train_dataset))
    dev_file = os.path.join(args.data, 'nli%s.dev.pth' % treetype)
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = NLIDataset(dev_dir, vocab, args.num_classes, edge_vocab, treetype=treetype)
        torch.save(dev_dataset, dev_file)
    logger.debug('==> Size of dev data     : %d ' % len(dev_dataset))
    test_file = os.path.join(args.data, 'nli%s.test.pth' % treetype)
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = NLIDataset(test_dir, vocab, args.num_classes, edge_vocab, treetype=treetype)
        torch.save(test_dataset, test_file)
    logger.debug('==> Size of test data    : %d ' % len(test_dataset))

    # initialize model, criterion, optimizer
    model = NLITreeLSTM(vocab_size=vocab.size(),
                        emb_size=args.word_size,
                        mem_size=args.mem_size,
                        hidden_size=args.hidden_size,
                        num_classes=args.num_classes,
                        freeze_embed=args.freeze_embed,
                        edge_vocab_size=edge_vocab.size(),
                        edge_size=args.edge_size,
                        model=args.model,
                        dropout=args.dropout)

    criterion = nn.NLLLoss()
    if args.cuda:
        model.cuda(), criterion.cuda()
    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                               weight_decay=args.weight_decay)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                  weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                              weight_decay=args.weight_decay)
    metrics = Metrics(args.num_classes)

    # word embedding
    emb_file = os.path.join(args.data, 'nli%s.embed.pth' % treetype)
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(os.path.join(args.glove, 'glove.840B.300d'))
        logger.debug('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
        emb = torch.Tensor(vocab.size(), glove_emb.size(1)).uniform_(-0.05, 0.05)
        # zero out the embeddings for padding and other special words
        for idx, item in enumerate([C.PAD_WORD, C.UNK_WORD]):
            emb[idx].zero_()
        for word in vocab.label_to_idx.keys():
            if glove_vocab.get_index(word):
                emb[vocab.get_index(word)] = glove_emb[glove_vocab.get_index(word)]
        torch.save(emb, emb_file)

    if args.cuda:
        emb = emb.cuda()
    model.embs.weight.data.copy_(emb)

    # train
    trainer = Trainer(args, model, criterion, optimizer)

    best = -float('inf')
    for epoch in range(args.epochs):
        _ = trainer.train(train_dataset)
        train_loss, train_pred = trainer.test(train_dataset)
        train_acc = metrics.eval_acc(train_pred, train_dataset.labels)
        print('==> Epoch {}, Train \tLoss: {}\tAccuracy: {}'.format(epoch + 1, train_loss, train_acc))

        dev_loss, dev_pred = trainer.test(dev_dataset)
        dev_acc = metrics.eval_acc(dev_pred, dev_dataset.labels)
        print('==> Epoch {}, Dev \tLoss: {}\tAccuracy: {}'.format(epoch + 1, dev_loss, dev_acc))

        if best < dev_acc:
            best = dev_acc
            test_loss, test_pred = trainer.test(test_dataset)
            test_acc = metrics.eval_acc(test_pred, test_dataset.labels)
            print('==> Epoch {}, Test \tLoss: {}\tAccuracy: {}'.format(epoch + 1, test_loss, test_acc))

            checkpoint = {
                'model': trainer.model.state_dict(),
                'optim': trainer.optimizer,
                'accuracy': test_acc,
                'predict': test_pred,
                'args': args,
                'epoch': epoch + 1
            }
            dataset = 'sick' if 'sick' in args.data else 'snli'
            filename = 'nli-{}-{}-{}-e{}-m{}-h{}-b{}-s{}'.format(dataset, args.treetype, args.model, args.edge_size,
                                                                 args.mem_size,
                                                                 args.hidden_size,
                                                                 args.batch_size,
                                                                 args.seed)
            logger.debug('==> New optimum found, checkpointing everything now...')
            torch.save(checkpoint, '%s.pt' % os.path.join(args.save, filename))


if __name__ == "__main__":
    main()
