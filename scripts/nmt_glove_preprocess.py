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
from torch.autograd import Variable

from hybridvec.loader import *
from hybridvec.models import BaselineModel
from hybridvec.models import Def2VecModel

import json
from hybridvec.config import eval_config

def get_args():
    """
    Gets the run_name, run_comment, and epoch of the model being evaluated
    """
    parser = argparse.ArgumentParser(description="Process nmt.")
    parser.add_argument('--glove_file', type=str, required = False,
                        default='data/glove/glove.6B.100d.txt',
                        help='Source glove file.')
    # parser.add_argument('num_k_keep', type=int, 
    #                     help='How many thousands of GloVe vectors to keep for NMT model.')
    parser.add_argument("run_name")
    parser.add_argument("run_comment")
    parser.add_argument("epoch")
    parser.add_argument("--verbose", default=True)
    parser.add_argument("--train", required = True)
    args = parser.parse_args()
    return (args.glove_file,  args.run_name, args.run_comment, args.epoch, args.verbose, args.train)

def load_config():
    """
    Load in the right config file from desired model to evaluate
    """
    glove_file, run_name, run_comment, epoch, verbose, train_data_flag = get_args()
    name = run_name + '-' + run_comment
    path = "outputs/def2vec/logs/{}/config.json".format(name)
    config = None
    with open(path) as f:
        config = dict(json.load(f))
        config = eval_config(config, run_name, run_comment, epoch, verbose)
    return (config,name, glove_file, train_data_flag)

def get_word(word):
    return vocab.vectors[vocab.stoi[word]]

if __name__ == "__main__":
    config, name, glove_file, train_data_flag = load_config()

    VOCAB_DIM = 100
    VOCAB_SOURCE = '6B'
    vocab = vocab.GloVe(name=VOCAB_SOURCE, dim=VOCAB_DIM)
    use_gpu = torch.cuda.is_available()
    print("Using GPU:", use_gpu)
    TRAIN_FILE = 'data/glove/train_glove.%s.%sd.txt'%(config.vocab_source,config.vocab_dim)
    FULL_FILE = 'data/glove/glove.%s.%sd.txt'%(config.vocab_source,config.vocab_dim)

    model = BaselineModel(vocab,
                             vocab_size = config.vocab_size,
                             embed_size = config.vocab_dim,
                             output_size = config.vocab_dim,
                             hidden_size = config.hidden_size,
                             use_packing = config.packing,
                             use_bidirection = config.use_bidirection,
                             use_attention = config.use_attention,
                             cell_type = config.cell_type,
                             use_cuda = use_gpu,
                             use_glove_init = config.use_glove_init)

    model.load_state_dict(torch.load(config.save_path), strict = True)

    if train_data_flag:
        data_loader = get_data_loader(TRAIN_FILE,
                                  vocab,
                                  INPUT_METHOD_ONE,
                                  VOCAB_DIM,
                                  batch_size = config.batch_size,
                                  num_workers = 8,
                                  shuffle=False,
                                  vocab_size = config.vocab_size)
        output_file = 'data/nmt/glove/glove_baseline_train.txt'

    else:
        data_loader = get_data_loader(FULL_FILE,
                                  vocab,
                                  INPUT_METHOD_ONE,
                                  VOCAB_DIM,
                                  batch_size = config.batch_size,
                                  num_workers = 8,
                                  shuffle=False,
                                  vocab_size = config.vocab_size)
        output_file = 'data/nmt/glove/glove_baseline_full.txt'

    if use_gpu:
        model = model.cuda()

    # shutil.copyfile(provided_file,output_file)
    with open(output_file,'a') as output:
        for i, data in tqdm(enumerate(data_loader, 0), total=len(data_loader)):
            words, inputs, lengths, labels = data
            labels = Variable(labels)

            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs, lengths).cpu().data.numpy()
            defn_embeds = model.defn_embed.cpu().data.numpy()
            for i,word in enumerate(words):
                vec_str = " ".join([str(x) for x in defn_embeds[i]])
                output.write('%s %s\n'%(words[i],vec_str))

