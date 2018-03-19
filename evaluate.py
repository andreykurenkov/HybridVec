from __future__ import print_function
import tqdm
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
from tensorboardX import SummaryWriter
from pytorch_monitor import monitor_module, init_experiment
from loader import *

DEBUG_LOG = True

VOCAB_DIM = 100
VOCAB_SOURCE = '6B'
GLOVE_FILE = 'data/test_glove.%s.%sd.txt'%(VOCAB_SOURCE,VOCAB_DIM)

CONFIG = dict(
        title="def2vec",
        description="Translating definitions to word vectors",
        run_name='full_debug_run', # defaults to START_TIME-HOST_NAME
        run_comment='1', # gets appended to run_name as RUN_NAME-RUN_COMMENT
        log_dir='logs',
        random_seed=42,
        learning_rate=.0005,
        max_epochs=5,
        batch_size=16,
        n_hidden=150,
        print_freq=1,
        write_embed_freq=100,
        weight_decay=0,
        save_path="./data/checkpoints/full_run_big_batch-def_concatdef_concat/epoch_5/model_weights.torch"
)

def get_word(word):
    return vocab.vectors[vocab.stoi[word]]

def closest(vec, n=10):
    """
    Find the closest words for a given vector
    """
    all_dists = [(w, torch.dist(vec, get_word(w))) for w in vocab.itos]
    return sorted(all_dists, key=lambda t: t[1])[:n]

if __name__ == "__main__":

    vocab = vocab.GloVe(name=VOCAB_SOURCE, dim=VOCAB_DIM)
    use_gpu = torch.cuda.is_available()
    print("Using GPU:", use_gpu)

    model = Def2VecModel(vocab,
                         embed_size = VOCAB_DIM,
                         output_size = VOCAB_DIM,
                         hidden_size = CONFIG['n_hidden'],
                         use_cuda = use_gpu,
                         use_packing = False)
    model.load_state_dict(torch.load(CONFIG['save_path']))
    data_loader = get_data_loader(GLOVE_FILE,
                                  vocab,
                                  INPUT_METHOD_ALL_CONCAT,
                                  VOCAB_DIM,
                                  batch_size = CONFIG['batch_size'],
                                  num_workers = 8,
                                  shuffle=True)

    if use_gpu:
        model = model.cuda()
    criterion = nn.MSELoss()

    running_loss = 0.0
    n_batches = 0

    for i, data in tqdm.tqdm(enumerate(data_loader, 0), total=len(data_loader)):
        words, inputs, lengths, labels = data
        labels = Variable(labels)

        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.data[0]
        n_batches += 1

    print("L2 loss:", running_loss / n_batches)

    while True:

        print("== Give a definition.")
        definition = raw_input()

        inputs = torch.LongTensor(get_on_the_fly_input(definition, vocab))
        inputs = inputs.unsqueeze(0)
        if use_gpu:
            inputs = inputs.cuda()
        out_embedding = model(inputs).squeeze().data
        if use_gpu:
            out_embedding = out_embedding.cpu()

        # print("Closest words:")
        # print(closest(out_embedding))

        print("== Give a word.")
        actual_word = raw_input()
        print("Cosine similarity between embedding and GloVe word:",
              F.cosine_similarity(out_embedding.unsqueeze(0), get_word(actual_word).unsqueeze(0)).numpy())

