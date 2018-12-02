import os
import numpy as np

import argparse
import torch
import torch._utils

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import torchtext.vocab as vocab
from torch.autograd import Variable

from hybridvec.loader import *
from hybridvec.config import *
from hybridvec.models import *

from tqdm import tqdm
from time import time

from tensorboardX import SummaryWriter

DEBUG_LOG = False

def weights_init_xavier(m):
    """
    Initialize according to Xavier initialization or default initialization.
    """
    if type(m) in [nn.Linear]:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.xavier_normal(m.weight_hh_l0)
        nn.init.xavier_normal(m.weight_ih_l0)

if __name__ == "__main__":

    use_gpu = torch.cuda.is_available()
    print("Using GPU:", use_gpu)

    # continue from last training
    config = load_config()

    # pymonitor change things, not a good idea. I have to restore the one I passed in.
    #run_name = config.run_name
    #writer, conf = init_experiment(config.__dict__)
    #config.run_name = run_name

    config.use_glove_init = False
    save_config(config)
    vocab = vocab.GloVe(name=config.vocab_source, dim=config.vocab_dim)
    print ('vocab dim', config.vocab_dim)

    TRAIN_FILE = 'data/glove/train_glove.%s.%sd.txt' % (config.vocab_source, config.vocab_dim)
    VAL_FILE = 'data/glove/val_glove.%s.%sd.txt' % (config.vocab_source, config.vocab_dim)

    model_type = config.model_type
    model_path = get_model_path(config)
    if model_type == 'baseline': 
        model = BaselineModel(vocab, 
                            config = config, 
                            use_cuda = use_gpu)

    elif model_type == 'seq2seq':
        encoder = EncoderRNN(config = config,
                            variable_lengths = False, 
                            embedding = None)
        decoder = DecoderRNN(config = config)
        model = Seq2seq(encoder = encoder, 
                        decoder=decoder)

    if model_path is None or not os.path.exists(model_path):
        model.apply(weights_init_xavier)
    else:
        model.load_state_dict(torch.load(model_path))

    if use_gpu:
        model = model.cuda()

    train_loader = get_data_loader(
                            TRAIN_FILE,
                            vocab,
                            config.input_method,
                            config.vocab_dim,
                            batch_size = config.batch_size,
                            num_workers = config.num_workers,
                            shuffle=config.shuffle,
                            vocab_size = config.vocab_size)

    val_loader = get_data_loader(
                            VAL_FILE,
                            vocab,
                            config.input_method,
                            config.vocab_dim,
                            batch_size = config.batch_size,
                            num_workers = config.num_workers,
                            shuffle=config.shuffle,
                            vocab_size = config.vocab_size)

    optimizer = optim.Adam(model.parameters(),
                            lr = config.learning_rate,
                            weight_decay = config.weight_decay)

    #if DEBUG_LOG:
    #    monitor_module(model, writer)

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
            
            loss_object, loss_val = model.calculate_loss(inputs, outputs, labels, words)
            loss_object.backward()
            optimizer.step()

            # print statistics
            running_loss += loss_val
            if embed_outs is None:
                embed_outs = model.get_def_embeddings(outputs)
                embed_labels = words
            else:
                embed_outs = torch.cat([embed_outs, model.get_def_embeddings(outputs)])
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
                    (epoch + 1, i + 1, running_loss / config.print_freq, diff/config.print_freq, total_time))
                start = end
                running_loss = 0.0

            if i % config.eval_freq == (config.eval_freq - 1):
                val_loss = 0.0
                for data in tqdm(val_loader, total=len(val_loader)):
                    words, inputs, lengths, labels = data
                    labels = Variable(labels)

                    if use_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    optimizer.zero_grad()
                    outputs = model(inputs, lengths)

                    loss_object, loss_val = model.calculate_loss(inputs, outputs, labels, words)
                    val_loss += loss_val
                
                print('Epoch: %d, batch: %d, val loss: %.4f' %
                             (epoch + 1, i + 1, val_loss / len(val_loader)))
            # increase iteration
            total_iter += 1

        config.load_epoch = epoch + 1

        #out_dir = "outputs/def2vec/checkpoints/{}-{}".format(config.run_name, config.run_commennt)
        #if not os.path.exists(out_dir):
        #    os.makedirs(out_dir)
        #out_path = "outputs/def2vec/checkpoints/{}-{}/epoch_{}".format(config.run_name, config.run_commennt, epoch + 1)
        #if not os.path.exists(out_path):
        #    os.makedirs(out_path)
        torch.save(model.state_dict(), get_model_path(config))

    print('Finished Training')
