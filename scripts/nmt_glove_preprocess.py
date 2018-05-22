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
    parser.add_argument('--glove_file', type=str, required = False,
                        default='data/glove/glove.6B.100d.txt',
                        help='Source glove file.')
    parser.add_argument('num_k_keep', type=int, 
                        help='How many thousands of GloVe vectors to keep for NMT model.')
    parser.add_argument("run_title")
    parser.add_argument("run_name")
    parser.add_argument("run_comment")
    parser.add_argument("epoch")
    parser.add_argument("--vocab_size", default=50000 )
    parser.add_argument("--num_layers", default=2 )
    parser.add_argument("--verbose", default=True )
    args = parser.parse_args()
    return (args.glove_file, args.num_k_keep, args.run_title, args.run_name, args.run_comment, args.epoch, args.vocab_size, args.num_layers, args.verbose)

def load_config():
    """
    Load in the right config file from desired model to evaluate
    """
    glove_file, num_k_keep, run_title, run_name, run_comment, epoch, vocab_size, num_layers,  verbose = get_args()
    name = run_name + '-' + run_comment
    path = "outputs/{}/logs/{}/config.json".format(str(run_title), name)
    config = None
    with open(path) as f:
        config = dict(json.load(f))
        config = eval_config(config, run_name, run_comment, epoch, verbose)
    return (config,name, glove_file, num_k_keep, vocab_size, num_layers)

def get_word(word):
    return vocab.vectors[vocab.stoi[word]]

if __name__ == "__main__":
    config, name, glove_file, num_k_keep, vocab_size, num_layers = load_config()

    GLOVE_TOTAL_K = 400

    provided_file = 'data/nmt/glove/glove_%dk_provided.txt'%(num_k_keep)
    held_out_file = 'data/nmt/glove/glove_%dk_held_out.txt'%(GLOVE_TOTAL_K-num_k_keep)
    output_file = 'data/nmt/glove/glove_%dk_provided_filled.txt'%(num_k_keep)

    with open(glove_file,'r') as glove_f:
        glove_lines = glove_f.readlines()

    with open(provided_file,'w') as provided:
        for i in range(num_k_keep*1000):
            provided.write(glove_lines[i])
        # Include unk token
        provided.write(glove_lines[-1])

    with open(held_out_file,'w') as held_out:
        for i in range(num_k_keep*1000, len(glove_lines)-1):
            held_out.write(glove_lines[i])

    VOCAB_DIM = 100
    VOCAB_SOURCE = '6B'
    vocab = vocab.GloVe(name=VOCAB_SOURCE, dim=VOCAB_DIM)
    use_gpu = torch.cuda.is_available()
    print("Using GPU:", use_gpu)
    
    vocab_reduced = True if vocab_size < 400000 else False
    encoder = EncoderRNN(vocab_size = vocab_size,
                      max_len = 200, 
                      hidden_size = config.hidden_size, 
                      embed_size = config.vocab_dim,
                      input_dropout_p=config.dropout,
                      dropout_p=config.dropout,
                      n_layers=num_layers,
                      bidirectional=config.use_bidirection,
                      rnn_cell=config.cell_type.lower(),
                      variable_lengths=False,
                      embedding=None, #randomly initialized,
                      )

    decoder = DecoderRNN(vocab_size = vocab_size,
                      max_len = 200,
                      hidden_size = config.hidden_size,
                      n_layers=num_layers,
                      rnn_cell=config.cell_type.lower(),
                      bidirectional=config.use_bidirection,
                      input_dropout_p=config.dropout,
                      dropout_p=config.dropout,
                      use_attention=config.use_attention
                      )


    model = Seq2SeqModel(encoder = encoder,
                        decoder = decoder
                        )
    model.load_state_dict(torch.load(config.save_path), strict = True)

    train_loader = get_data_loader(TRAIN_FILE,
                                 vocab_1,
                                 config.input_method,
                                 config.vocab_dim,
                                 batch_size = config.batch_size,
                                 num_workers = config.num_workers,
                                 shuffle=False,
                                 vocab_size=vocab_size)
    if use_gpu:
        model = model.cuda()
    model.train(False)

    shutil.copyfile(provided_file,output_file)
    with open(output_file,'a') as output:
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
            words, inputs, lengths, labels = data
            labels = Variable(labels)

            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            (decoder_outputs, decoder_hidden, ret_dicts), encoder_hidden  = model(inputs, lengths)
            for idx, word in enumerate(words):
                vec_str = " ".join([str(x) for x in encoder_hidden.cpu().data[idx, :]])
                output.write('%s %s\n'%(words[idx],vec_str))
