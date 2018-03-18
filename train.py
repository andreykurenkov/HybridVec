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
from model import Def2VecModel
from torch.autograd import Variable
import torchtext.vocab as vocab
from tensorboardX import SummaryWriter
from pytorch_monitor import monitor_module, init_experiment
from loader import *
import torch.nn.init as init
from tqdm import tqdm
from time import time

DEBUG_LOG = False

CONFIG = dict(
    # meta data
    title="def2vec",
    description="Translating definitions to word vectors",
    run_name='full_run_big_hidden', # defaults to START_TIME-HOST_NAME
    run_comment='def_concat', # gets appended to run_name as RUN_NAME-RUN_COMMENT
    log_dir='logs',
    vocab_dim = 100,
    vocab_source = '6B',
    load_path = None,
    # hyperparams
    random_seed=42,
    learning_rate=.0001,
    max_epochs=13,
    batch_size=128,
    n_hidden=250,
    # logging params
    print_freq=1,
    write_embed_freq=100,
    eval_freq = 1000,
    save_path="./model_weights.torch",
    embedding_log_size = 10000,
    # data loading params
    num_workers = 8,
    packing=True,
    shuffle=True,
    # model configuration [for ablation/hyperparam experiments]
    weight_init="xavier",
    input_method=INPUT_METHOD_ALL_CONCAT,
    use_bidirection=True,
    use_attention=False,
    cell_type='GRU',
    #use_batchnorm=True,
    hidden_size=150,
    embed_size=100,
    dropout=0.1,
    weight_decay=0.0,
)
TRAIN_FILE = 'data/glove/train_glove.%s.%sd.txt'%(CONFIG['vocab_source'],CONFIG['vocab_dim'])
VAL_FILE = 'data/glove/val_glove.%s.%sd.txt'%(CONFIG['vocab_source'],CONFIG['vocab_dim'])

def weights_init(m):
    """
    Initialize according to Xavier initialization or default initialization.
    """
    if CONFIG['weight_init'] == 'xavier':
        if type(m) in [nn.Linear]:
            nn.init.xavier_normal(m.weight.data)
        elif type(m) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.xavier_normal(m.weight_hh_l0)
            nn.init.xavier_normal(m.weight_ih_l0)


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

    if CONFIG["load_path"] is None:
        model.apply(weights_init)
    else:
        model.load_state_dict(torch.load(CONFIG["load_path"]))
    model.apply(weights_init)

    if use_gpu:
        model = model.cuda()

    train_loader = get_data_loader(TRAIN_FILE,
                                   vocab,
                                   CONFIG['input_method'],
                                   CONFIG['vocab_dim'],
                                   batch_size = CONFIG['batch_size'],
                                   num_workers = CONFIG['num_workers'],
                                   shuffle=CONFIG['shuffle'])
    val_loader = get_data_loader(VAL_FILE,
                                   vocab,
                                   CONFIG['input_method'],
                                   CONFIG['vocab_dim'],
                                   batch_size = CONFIG['batch_size'],
                                   num_workers = CONFIG['num_workers'],
                                   shuffle=CONFIG['shuffle'])


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=CONFIG['learning_rate'],
                           weight_decay=CONFIG['weight_decay'])

    writer, conf = init_experiment(CONFIG)
    if DEBUG_LOG:
        monitor_module(model, writer)

    total_time = 0
    total_iter = 0

    embed_outs = None
    embed_labels = []

    for epoch in range(CONFIG['max_epochs']):

        running_loss = 0.0
        start = time()
        print("Epoch", epoch)

        for i, data in enumerate(train_loader, 0):
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
                if num_outs > CONFIG['embedding_log_size']:
                    diff = num_outs - CONFIG['embedding_log_size']
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

            if i % CONFIG['eval_freq'] == (CONFIG['eval_freq'] - 1):

                val_loss = 0.0
                for data in tqdm(val_loader, total=len(val_loader)):
                    words, inputs, lengths, labels = data
                    labels = Variable(labels)
                    if use_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    outputs = model(inputs, lengths)
                    loss = criterion(outputs, labels)
                    val_loss += loss.data[0]
                writer.add_scalar('val_loss', val_loss / len(val_loader), total_iter)
                print('Epoch: %d, batch: %d, val loss: %.4f' %
                             (epoch + 1, i + 1, val_loss / len(val_loader)))

            total_iter += 1

        if not os.path.exists("checkpoints/{}".format(CONFIG['run_name'] + CONFIG['run_comment'])):
            os.mkdir("checkpoints/{}".format(CONFIG['run_name'] + CONFIG['run_comment']))
        if not os.path.exists("checkpoints/{}/epoch_{}".format(CONFIG['run_name'] + CONFIG['run_comment'], epoch + 1)):
            os.mkdir("checkpoints/{}/epoch_{}".format(CONFIG['run_name'] + CONFIG['run_comment'], epoch + 1))
        torch.save(model.state_dict(), "checkpoints/{}/epoch_{}".format(CONFIG['run_name'] + CONFIG['run_comment'], epoch + 1) + "/" + CONFIG['save_path'])

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

    print('Finished Training')
