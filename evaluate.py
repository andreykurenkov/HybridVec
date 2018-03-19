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
from torch.autograd import Variable
import torchtext.vocab as vocab
from loader import *
from tensorboardX import SummaryWriter
from pytorch_monitor import monitor_module, init_experiment
import requests_cache


DEBUG_LOG = False

CONFIG = dict(
    # meta data
    title="def2vec",
    description="Translating definitions to word vectors",
    run_name='ablate_attn', # defaults to START_TIME-HOST_NAME
    run_comment='vanilla', # gets appended to run_name as RUN_NAME-RUN_COMMENT
    log_dir='logs',
    vocab_dim = 100,
    vocab_source = '6B',
    load_path = None,
    # hyperparams
    random_seed=42,
    learning_rate=.0001,
    max_epochs=15,
    batch_size=16,
    n_hidden=250,
    # logging params
    print_freq=1,
    write_embed_freq=100,
    eval_freq = 1000,
    save_path="./checkpoints/full_run-vanillavanilla/epoch_5/model_weights.torch",
    embedding_log_size = 10000,
    # data loading params
    num_workers = 8,
    packing=False,
    shuffle=True,
    # model configuration [for ablation/hyperparam experiments]
    weight_init="xavier",
    input_method=INPUT_METHOD_ONE,
    use_bidirection=True,
    use_attention=True,
    cell_type='GRU',
    hidden_size=150,
    embed_size=100,
    dropout=0.1,
    weight_decay=0.0,
)
TEST_FILE = 'data/glove/test_glove.%s.%sd.txt'%(CONFIG['vocab_source'],CONFIG['vocab_dim'])


def get_word(word):
    return vocab.vectors[vocab.stoi[word]]

def closest(vec, n=10):
    """
    Find the closest words for a given vector
    """
    all_dists = [(w, torch.dist(vec, get_word(w))) for w in vocab.itos]
    return sorted(all_dists, key=lambda t: t[1])[:n]

if __name__ == "__main__":

    vocab = vocab.GloVe(name=CONFIG['vocab_source'], dim=CONFIG['vocab_dim'])
    use_gpu = torch.cuda.is_available()
    print("Using GPU:", use_gpu)

    model = Def2VecModel(vocab,
                         embed_size = CONFIG['vocab_dim'],
                         output_size = CONFIG['vocab_dim'],
                         hidden_size = CONFIG['hidden_size'],
                         use_packing = CONFIG['packing'],
                         use_bidirection = CONFIG['use_bidirection'],
                         use_attention = CONFIG['use_attention'],
                         cell_type = CONFIG['cell_type'],
                         use_cuda = use_gpu)
    model.load_state_dict(torch.load(CONFIG['save_path']))
    test_loader = get_data_loader(TEST_FILE,
                                   vocab,
                                   CONFIG['input_method'],
                                   CONFIG['vocab_dim'],
                                   batch_size = CONFIG['batch_size'],
                                   num_workers = CONFIG['num_workers'],
                                   shuffle=False)

    if use_gpu:
        model = model.cuda()
    criterion = nn.MSELoss()
    model.train(False)

    running_loss = 0.0
    n_batches = 0
    out_embeddings = {}
    out_attns = {}
    out_defns = {}

    for i, data in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
        words, inputs, lengths, labels = data
        labels = Variable(labels)

        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs, attns = model(inputs, return_attn=True)
        loss = criterion(outputs, labels)

        running_loss += loss.data[0]
        n_batches += 1

        for word, embed, attn, inp in zip(words,
                                          outputs.data.cpu(),
                                          attns.data.cpu().squeeze(2),
                                          inputs.cpu()):
            out_embeddings[word] = embed.numpy()
            out_attns[word] = attn.numpy()
            out_defns[word] = " ".join([vocab.itos[i - 1] for i in inp])

    print("L2 loss:", running_loss / n_batches)
    np.save("eval/out_embeddings.npy", out_embeddings)
    np.save("eval/out_attns.npy", out_attns)
    np.save("eval/out_defns.npy", out_defns)
