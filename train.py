from __future__ import print_function
import sys
import os
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
import torch.nn.init as init
import requests_cache

DEBUG_LOG = True
requests_cache.install_cache('cache')

VOCAB_DIM = 100
VOCAB_SOURCE = '6B'
GLOVE_FILE = 'data/train_glove.%s.%sd.txt'%(VOCAB_SOURCE,VOCAB_DIM)

CONFIG = dict(
        title="def2vec",
        description="Translating definitions to word vectors",
        run_name='full_debug_run', # defaults to START_TIME-HOST_NAME
        run_comment='weight_init', # gets appended to run_name as RUN_NAME-RUN_COMMENT
        log_dir='logs',
        random_seed=42,
        learning_rate=.0005,
        max_epochs=5,
        batch_size=64,
        n_hidden=150,
        print_freq=1,
        write_embed_freq=100,
        weight_decay=0,
        save_path="model_weights.torch",
        load_path=None,
        weight_init="xavier",
        packing=False
)

def weights_init(m):
    if CONFIG['weight_init']=='xavier':
        if type(m) in [nn.Linear]:
            nn.init.xavier_normal(m.weight.data)
        elif type(m) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.xavier_normal(m.weight_hh_l0)
            nn.init.xavier_normal(m.weight_ih_l0)


if __name__ == "__main__":

    vocab = vocab.GloVe(name=VOCAB_SOURCE, dim=VOCAB_DIM)
    use_gpu = torch.cuda.is_available()
    print("Using GPU:", use_gpu)

    model = Def2VecModel(vocab,
                         embed_size = VOCAB_DIM,
                         output_size = VOCAB_DIM,
                         hidden_size = CONFIG['n_hidden'],
                         use_cuda = use_gpu,
                         use_packing = CONFIG['packing'])
    if CONFIG["load_path"] is None:
        model.apply(weights_init)
    else:
        model.load_state_dict(torch.load(CONFIG["load_path"]))

    data_loader = get_data_loader(GLOVE_FILE,
                                  vocab,
                                  VOCAB_DIM,
                                  batch_size = CONFIG['batch_size'],
                                  num_workers = 8,
                                  shuffle=True)

    if use_gpu:
        model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                             lr=CONFIG['learning_rate'],
                             weight_decay=CONFIG['weight_decay'])

    writer,conf = init_experiment(CONFIG)
    if DEBUG_LOG:
        monitor_module(model, writer)

    total_time = 0
    total_iter = 0

    EMBEDDING_SIZE = 7500
    embed_outs = None
    embed_labels = []

    for epoch in range(CONFIG['max_epochs']):

        running_loss = 0.0
        start = time()
        print("Epoch", epoch)

        for i, data in enumerate(data_loader, 0):
            words, inputs, lengths, labels = data
            labels = Variable(labels)

            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            writer.add_scalar('loss', loss.data[0], total_iter)
            if embed_outs is None:
                embed_outs = outputs.data
                embed_labels = words
            else:
                embed_outs = torch.cat([embed_outs, outputs.data])
                embed_labels += words
                num_outs = embed_outs.shape[0]
                if num_outs > EMBEDDING_SIZE:
                    diff = num_outs - EMBEDDING_SIZE
                    embed_outs = embed_outs[diff:]
                    embed_labels = embed_labels[diff:]

            if i % CONFIG['print_freq'] == (CONFIG['print_freq']-1):
                end = time()
                diff = end-start
                total_time+=diff
                print('Epoch: %d, batch: %d, loss: %.4f , time/iter: %.2fs, total time: %.2fs' %
                             (epoch + 1, i + 1,
                                running_loss / CONFIG['print_freq'],
                                diff/CONFIG['print_freq'],
                                total_time))
                start = end
                running_loss = 0.0
            if i % CONFIG['write_embed_freq'] == (CONFIG['write_embed_freq']-1):
                writer.add_embedding(embed_outs,
                                     metadata=embed_labels,
                                     global_step=total_iter)

            total_iter += 1

        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        if not os.path.exists("checkpoints/epoch_{}".format(epoch + 1)):
            os.mkdir("checkpoints/epoch_{}".format(epoch + 1))
        torch.save(model.state_dict(), "checkpoints/epoch_{}".format(epoch + 1) + "/" + CONFIG['save_path'])

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

    print('Finished Training')
