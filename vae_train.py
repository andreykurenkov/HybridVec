# import os
# import json
# import time
# import torch
# import argparse
# import numpy as np
# from multiprocessing import cpu_count
# from tensorboardX import SummaryWriter
# from torch.utils.data import DataLoader
# from collections import OrderedDict, defaultdict

import sys
import os
import collections
import traceback
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse

from sklearn.metrics import precision_score, accuracy_score, recall_score, mean_squared_error
from model import Def2VecModel, Seq2SeqModel
from torch.autograd import Variable
import torchtext.vocab as vocab
from tensorboardX import SummaryWriter
from loader import *
import torch.nn.init as init
from tqdm import tqdm
from time import time
from config import train_config
from pytorch_monitor import monitor_module, init_experiment
from datetime import datetime

from vae_utils import to_var, idx2word, expierment_name
from vae_model import SentenceVAE
DEBUG_LOG = False

def main(args):
    config = train_config()
    vocab_1 = vocab.GloVe(name=config.vocab_source, dim=config.vocab_dim)
    use_gpu = torch.cuda.is_available()
    print("Using GPU:", use_gpu)
    TRAIN_FILE = 'data/glove/train_glove.%s.%sd.txt'%(config.vocab_source,config.vocab_dim)



    model = SentenceVAE(vocab_size=config.vocab_size,
        sos_idx=config.vocab_size + 2,
        eos_idx=config.vocab_size + 1,
        pad_idx=config.vocab_size + 3,
        max_sequence_length=config.max_len,
        embedding_size=config.vocab_dim,
        rnn_type=config.cell_type.lower(),
        hidden_size=config.hidden_size,
        word_dropout=config.dropout,
        latent_size=config.hidden_size,
        num_layers=config.num_layers,
        bidirectional=config.use_bidirection
        )

    if torch.cuda.is_available():
        model = model.cuda()

    train_loader = get_data_loader(TRAIN_FILE,
                                   vocab_1,
                                   config.input_method,
                                   config.vocab_dim,
                                   batch_size = config.batch_size,
                                   num_workers = config.num_workers,
                                   shuffle=config.shuffle,
                                   vocab_size = config.vocab_size)

    # val_loader = get_data_loader(VAL_FILE,
    #                                vocab_1,
    #                                config.input_method,
    #                                config.vocab_dim,
    #                                batch_size = config.batch_size,
    #                                num_workers = config.num_workers,
    #                                shuffle=config.shuffle,
    #                                vocab_size = config.vocab_size)

    out_dir = "outputs/{}/checkpoints/{}".format(config.title, config.run_name + "-" + config.run_comment + "-" + str(config.exp_counter))
    while os.path.exists(out_dir):
        config.exp_counter += 1
        out_dir = "outputs/{}/checkpoints/{}".format(config.title, config.run_name + "-" + config.run_comment + "-{}".format(config.exp_counter))

    config.run_comment += "-{}".format(config.exp_counter) #add exp counter to run-comment so that other eval code doesnt change and for log changed in pytorch-monitor
    config.run_time = str(datetime.now())


    writer, conf = init_experiment(config.__dict__) #pytorch-monitor needs a dict
    print ("Running experiment named --- {}".format(config.run_name))

    if DEBUG_LOG:
        monitor_module(model, writer)

    total_time = 0
    total_iter = 0

    embed_outs = None
    embed_labels = []

    embed_dicts = {}


    def kl_anneal_function(anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0)

    NLL = torch.nn.NLLLoss(size_average=False, ignore_index=config.vocab_size + 3)
    def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0):

        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).data[0]].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))
        
        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k, x0)

        return NLL_loss, KL_loss, KL_weight

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    for epoch in range(config.max_epochs):
        running_loss = 0.0
        start = time()
        print("Epoch", epoch)

        for iteration, data in enumerate(train_loader, 0):
            words, inputs, lengths, labels = data
            batch_size = inputs.shape[0]
            inputs, labels, lengths = Variable(inputs), Variable(labels), Variable(torch.FloatTensor(lengths))

            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()


            # Forward pass
            logp, mean, logv, z = model(inputs, lengths)

            # loss calculation
            NLL_loss, KL_loss, KL_weight = loss_fn(logp, inputs, #trying to recreate what went in
                lengths, mean, logv, args.anneal_function, step, args.k, args.x0)

            loss = (NLL_loss + KL_weight * KL_loss)/batch_size

            # backward + optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            step += 1


            writer.add_scalar("%s/Loss"%split.upper(), loss.data[0], epoch*len(train_loader) + iteration)
            writer.add_scalar("%s/NLL Loss"%split.upper(), NLL_loss.data[0]/batch_size, epoch*len(train_loader) + iteration)
            writer.add_scalar("%s/KL Loss"%split.upper(), KL_loss.data[0]/batch_size, epoch*len(train_loader) + iteration)
            writer.add_scalar("%s/KL Weight"%split.upper(), KL_weight, epoch*len(train_loader) + iteration)


            if embed_outs is None:
                embed_outs = z.data.cpu()
                embed_labels = words
            else:
                embed_outs = torch.cat([embed_outs, z.data.cpu()])
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
                             (epoch + 1, iteration + 1,
                              running_loss / config.print_freq,
                              diff/config.print_freq,
                              total_time))
                start = end
                running_loss = 0.0

            if i % config.write_embed_freq == (config.write_embed_freq-1):
                writer.add_embedding(embed_outs,
                                     metadata=embed_labels,
                                     global_step=total_iter)
            total_iter += 1
        out_dir = "outputs/{}/checkpoints/{}".format(config.title, config.run_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = "outputs/{}/checkpoints/{}/epoch_{}".format(config.title, config.run_name, epoch + 1)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        print ("saving model")
    torch.save(model.state_dict(), out_path + "/" + config.save_path)
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

    print('Finished Training')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.5)

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-v','--print_every', type=int, default=50)
    parser.add_argument('-tb','--tensorboard_logging', action='store_true')
    parser.add_argument('-log','--logdir', type=str, default='logs')
    parser.add_argument('-bin','--save_model_path', type=str, default='bin')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.anneal_function in ['logistic', 'linear']

    main(args)
