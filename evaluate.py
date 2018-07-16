from __future__ import print_function
from tqdm import tqdm
import sys
import torch.nn.functional as F
import collections
import traceback
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, accuracy_score, recall_score, mean_squared_error
from time import time
from model import Def2VecModel
from baseline import BaselineModel
from torch.autograd import Variable
import torchtext.vocab as vocab
from tensorboardX import SummaryWriter
from pytorch_monitor import monitor_module, init_experiment
from loader import *
from config import eval_config
import json
import argparse
from train import calculate_loss

DEBUG_LOG = False


def get_args():
    """
    Gets the run_name, run_comment, and epoch of the model being evaluated
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name")
    parser.add_argument("run_comment")
    parser.add_argument("epoch")
    parser.add_argument("--verbose", default=True)
    args = parser.parse_args()
    return (args.run_name, args.run_comment, args.epoch, args.verbose)

def load_config():
    """
    Load in the right config file from desired model to evaluate
    """
    run_name, run_comment, epoch, verbose = get_args()
    name = run_name + '-' + run_comment
    path = "outputs/def2vec/logs/{}/config.json".format(name)
    config = None
    with open(path) as f:
        config = dict(json.load(f))
        config = eval_config(config, run_name, run_comment, epoch, verbose)
    return config

def get_word(word):
    return vocab.vectors[vocab.stoi[word]]

def closest(vec, n=10):
    """
    Find the closest words for a given vector
    """
    all_dists = [(w, torch.dist(vec, get_word(w))) for w in vocab.itos]
    return sorted(all_dists, key=lambda t: t[1])[:n]

def write_output(f, pred, inputs, words, vocab_size):
  #print (pred)  
  for w in range(len(words)):
    f.write(words[w] +"\n")
    definition_input = [(vocab.itos[i] if (i>0 and i<=vocab_size+1) else str(i)) for i in (inputs[w])]
    definition_input = "input definition: " + " ".join(definition_input)
    f.write(definition_input+ "\n")

    len_pred = min(len(inputs[w]), len(pred[w]))
    dfn_pred = [(vocab.itos[pred[w][i]] if (pred[w][i]>0 and pred[w][i]<=vocab_size+1) else str(pred[w][i])) for i in range(len_pred) ]
    dfn_pred = "predicted definition: " + " ".join(dfn_pred) + "\n"
    f.write(dfn_pred)
    f.write("\n\n")

if __name__ == "__main__":
    f = open('input-output-baseline.txt','w')
    config = load_config()
    TEST_FILE = 'data/glove/test_glove.%s.%sd.txt'%(config.vocab_source,config.vocab_dim)
    vocab = vocab.GloVe(name=config.vocab_source, dim=config.vocab_dim)
    use_gpu = torch.cuda.is_available()
    print("Using GPU:", use_gpu)

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

    test_loader = get_data_loader(TEST_FILE,
                                 vocab,
                                 config.input_method,
                                 config.vocab_dim,
                                 batch_size = config.batch_size,
                                 num_workers = config.num_workers,
                                 shuffle=False,
                                 vocab_size=config.vocab_size)

    if use_gpu:
        model = model.cuda()
    criterion = nn.NLLLoss() #use multi label loss across unigram bag of words model
    reg_criterion = nn.MSELoss(reduce=False)
    if config.glove_aux_loss: glove_criterion = nn.MSELoss(reduce=False)
    model.train(False)

    running_loss = 0.0
    n_batches = 0
    out_embeddings = {}
    pred_defns = {}
    out_defns = {}

    for i, data in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
        words, inputs, lengths, labels = data
        labels = Variable(labels)

        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs, lengths)

        w_indices = np.array([vocab.stoi[w] + 1 for w in words])
        w_indices[w_indices > config.vocab_size] = 0 #for vocab size 
        input_embeddings = Variable(model.embeddings.weight.data[w_indices])
        defn_embeddings = model.defn_embed
        
        if use_gpu:
            defn_embeddings = defn_embeddings.cuda()
            input_embeddings = input_embeddings.cuda()
        criterions = [criterion, reg_criterion, glove_criterion] if config.glove_aux_loss else [criterion, reg_criterion]

        loss = calculate_loss(inputs, outputs, labels, criterions, input_embeddings, defn_embeddings)

        running_loss += loss.data[0]

        n_batches += 1

        #write to fil
        word_preds = []

        #need to make preds matrix of indices for each word where we take top d and fill the rest w 0s 
        #output is currently a 64 x vocab size matrix of probabilities -- from each, take the top d 
        def_len = inputs.size()[1] #length of definitions for the batch 
        outputs_np = outputs.data.cpu().numpy()
        top_indices = np.argpartition(outputs_np, -1*def_len)[:,-1*def_len:]

        write_output(f, top_indices, inputs, words, config.vocab_size)

        for word, embed, inp in zip(words,
                                  model.defn_embed.data.cpu(),
                                  inputs.cpu()):
            out_embeddings[word] = embed.numpy()
            out_defns[word] = " ".join([vocab.itos[i - 1] for i in inp])

    f.close()
    print("L2 loss over entire set:", running_loss / n_batches)
    np.save("eval/out_embeddings.npy", out_embeddings)
    #np.save("eval/out_attns.npy", out_attns)
    np.save("eval/out_defns.npy", out_defns)