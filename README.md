## Multiplicative Tree LSTM

An implementation of the mTreeLSTM architectures.

### Citation
Nam Khanh Tran, Weiwei Cheng. 
_Multiplicative Tree-Structured Long Short-Term Memory Networks for Semantic Representations_.
Proceedings of the 7th Joint Conference on Lexical and Computational Semantics (*SEM-18): 276-286, ACL.
New Orleans, USA, June 2018

### Requirements
- PyTorch (0.3.0)
- Python3 (3.6.1)
- Java8 (for Stanford Parsers)

### Usage

Download the following data:
- [Stanford Sentiment Treebank](http://nlp.stanford.edu/sentiment/index.html) (sentiment classification task)
- [SICK dataset](http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools) (semantic relatedness and NLI tasks)
- [SNLI dataset](https://nlp.stanford.edu/projects/snli/) (NLI task)
- [Glove word vectors](http://nlp.stanford.edu/projects/glove/) (Common Crawl 840B)

Preprocess:
- [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.shtml)
- [Stanford POS Tagger](http://nlp.stanford.edu/software/tagger.shtml)
- [Stanford NN Dependency Parser](http://nlp.stanford.edu/software/nndep.shtml)
- [AMR parser -- JAMR](https://github.com/jflanigan/jamr)

Or run the script ```fetch_and_preprocess.sh```, as described in
[https://github.com/stanfordnlp/treelstm](https://github.com/stanfordnlp/treelstm).

Or use the pre-processed sentences [here](https://drive.google.com/file/d/1upYIKkE95CT9w6H3IFmGMp0hpB0RJ4pn)

### Natural Language Inference

In this task, the model reads two sentences (a premise and a hypothesis), and outputs a judgement of `entailment`, 
`contradiction`, or `neutral`, reflecting the relationship between the meanings of the two sentences.

To train models for the NLI task on SICK dataset, run:

```
python nli.py --model <base|add|full|multi> --data data/sick --glove data/glove --word_size 300 --edge_size 100 
              --mem_size 150 --hidden_size 50 --batch_size 25 --optim adam --epochs 10 --num_classes 3
```

To train models for the NLI task on SNLI dataset, run:
```
python nli.py --model <base|add|full|multi> --data data/snli --glove data/glove --word_size 300 --edge_size 100 
              --mem_size 100 --hidden_size 200 --batch_size 128 --optim adam --epochs 10 --num_classes 3
```

where:

- `model`: TreeLSTM variant to train
- `data`: path to dataset
- `glove`: path to pre-trained word embeddings
- `edge_size`: size of relation embeddings
- `mem_size`: LSTM memory dimension
- `hidden_size`: size of the classifier layer
- `batch_size`: batch size
- `epochs`: the number of traning epochs

See the [paper](http://aclweb.org/anthology/S18-2032) for more details on these experiments.
