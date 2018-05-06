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
from loader import *
import torch.nn.init as init
from tqdm import tqdm
from time import time
from config import train_config
from pytorch_monitor import monitor_module, init_experiment


DEBUG_LOG = False

config = train_config()

TRAIN_FILE = 'data/glove/train_glove.%s.%sd.txt'%(config.vocab_source,config.vocab_dim)
VAL_FILE = 'data/glove/val_glove.%s.%sd.txt'%(config.vocab_source, config.vocab_dim)

def weights_init(m):
    """
    Initialize according to Xavier initialization or default initialization.
    """
    if config.weight_init == 'xavier':
        if type(m) in [nn.Linear]:
            nn.init.xavier_normal(m.weight.data)
        elif type(m) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.xavier_normal(m.weight_hh_l0)
            nn.init.xavier_normal(m.weight_ih_l0)


if __name__ == "__main__":

    vocab = vocab.GloVe(name=config.vocab_source, dim=config.vocab_dim)
    use_gpu = torch.cuda.is_available()
    print("Using GPU:", use_gpu)

    model = Def2VecModel(vocab,
                         embed_size = config.vocab_dim,
                         output_size = config.vocab_dim,
                         hidden_size = config.hidden_size,
                         use_packing = config.packing,
                         use_bidirection = config.use_bidirection,
                         use_attention = config.use_attention,
                         cell_type = config.cell_type,
                         use_cuda = use_gpu)

    if config.load_path is None:
        model.apply(weights_init)
    else:
        model.load_state_dict(torch.load(config.load_path))
    model.apply(weights_init)

    if use_gpu:
        model = model.cuda()

    train_loader = get_data_loader(TRAIN_FILE,
                                   vocab,
                                   config.input_method,
                                   config.vocab_dim,
                                   batch_size = config.batch_size,
                                   num_workers = config.num_workers,
                                   shuffle=config.shuffle)
    val_loader = get_data_loader(VAL_FILE,
                                   vocab,
                                   config.input_method,
                                   config.vocab_dim,
                                   batch_size = config.batch_size,
                                   num_workers = config.num_workers,
                                   shuffle=config.shuffle)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=config.learning_rate,
                           weight_decay=config.weight_decay)



    writer, conf = init_experiment(config.__dict__) #pytorch-monitor needs a dict
    if DEBUG_LOG:
        monitor_module(model, writer)

    total_time = 0
    total_iter = 0

    embed_outs = None
    embed_labels = []

    for epoch in range(config.max_epochs):

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
                if num_outs > config.embedding_log_size:
                    diff = num_outs - config.embedding_log_size
                    embed_outs = embed_outs[diff:]
                    embed_labels = embed_labels[diff:]

            if i % config.print_freq == (config.print_freq-1):
                end = time()
                diff = end-start
                total_time+=diff
                print('Epoch: %d, batch: %d, loss: %.4f , time/iter: %.2fs, total time: %.2fs' %
                             (epoch + 1, i + 1,
                              running_loss / config.print_freq,
                              diff/config.print_freq,
                              total_time))
                start = end
                running_loss = 0.0

            if i % config.write_embed_freq == (config.write_embed_freq-1):
                writer.add_embedding(embed_outs,
                                     metadata=embed_labels,
                                     global_step=total_iter)

            if i % config.eval_freq == (config.eval_freq - 1):

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
        name = config.run_name + '-' + config.run_comment
        out_dir = "outputs/def2vec/checkpoints/{}".format(name)
        if not os.path.exists(out_dir):
            os.mkdirs(out_dir)
        out_path = "outputs/def2vec/checkpoints/{}/epoch_{}".format(name, epoch + 1)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        torch.save(model.state_dict(), out_path + "/" + config.save_path)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

    print('Finished Training')
