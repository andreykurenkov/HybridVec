from __future__ import print_function
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import collections
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchtext.vocab as vocab
import argparse
import shutil
from tqdm import tqdm
from model import Def2VecModel, Seq2SeqModel
from torch.autograd import Variable
from loader import *
import json
from seq2seq import EncoderRNN, DecoderRNN
from config import eval_config


def get_args():
    """
    Gets the run_name, run_comment, and epoch of the model being evaluated
    """
    parser = argparse.ArgumentParser(description="Process nmt.")
    parser.add_argument("run_title")
    parser.add_argument("run_name")
    parser.add_argument("run_comment")
    parser.add_argument("epoch")
    parser.add_argument("--train", required = True)
    parser.add_argument("--vocab_size", default=50000 )
    parser.add_argument("--num_layers", default=2 )
    parser.add_argument("--verbose", default=True )
    args = parser.parse_args()
    return (args.run_title, args.run_name, args.run_comment, args.epoch, args.train, args.vocab_size, args.num_layers, args.verbose)

def load_config():
    """
    Load in the right config file from desired model to evaluate
    """
    run_title, run_name, run_comment, epoch, train, vocab_size, num_layers,  verbose = get_args()
    name = run_name + '-' + run_comment
    path = "outputs/{}/logs/{}/config.json".format(str(run_title), name)
    config = None
    with open(path) as f:
        config = dict(json.load(f))
        config = eval_config(config, run_name, run_comment, epoch, verbose)
    return (config,name, vocab_size, num_layers, train)

def get_word(word, vocab_1):
    word = unicode(word, 'utf-8')
    return vocab_1.vectors[vocab_1.stoi[word]]

if __name__ == "__main__":
    #config, name, vocab_size, num_layers, train_flag = load_config()

    VOCAB_DIM = 100
    VOCAB_SOURCE = '6B'
    train_flag = True
    if train_flag:
        TRAIN_FILE = 'data/glove/train_glove.%s.%sd.txt'%(VOCAB_SOURCE,VOCAB_DIM)
        output_file = 'data/nmt/glove/glove-full500k-train.txt'
    else:
        TRAIN_FILE = 'data/glove/glove.%s.%sd.txt'%(config.vocab_source,config.vocab_dim)
        output_file = 'data/nmt/glove/glovefull500k_full.txt'


    vocab_1 = vocab.GloVe(name=VOCAB_SOURCE, dim=VOCAB_DIM)
    use_gpu = torch.cuda.is_available()
    print("Using GPU:", use_gpu)
    

    train_loader = get_data_loader(TRAIN_FILE,
                                 vocab_1,
                                 "concat_defs",
                                 VOCAB_DIM,
                                 batch_size = 64,
                                 num_workers = 8,
                                 shuffle=False,
                                 vocab_size=50000)


    with open(output_file,'a') as output:
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
            words, inputs, lengths, labels = data
            labels = Variable(labels)

            for idx, word in enumerate(words):
                glove_vecs = [str(x) for x in get_word(word, vocab_1)]
                vec_str = " ".join(glove_vecs)
                output.write('%s %s\n'%(words[idx],vec_str))
