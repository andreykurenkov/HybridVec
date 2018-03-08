from __future__ import print_function
import tqdm
import sys
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
from loader import get_data_loader, DefinitionsDataset
from tensorboardX import SummaryWriter
from pytorch_monitor import monitor_module, init_experiment
import requests_cache


DEBUG_LOG = True
requests_cache.install_cache('cache')

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
        save_path="./model_weights.torch"
)

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
