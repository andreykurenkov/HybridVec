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
from hybridvec.config import train_config
from hybridvec.models import Seq2seq, BaselineModel, EncoderRNN, DecoderRNN

from tqdm import tqdm
from time import time
from pytorch_monitor import monitor_module, init_experiment

from tensorboardX import SummaryWriter

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

def get_model_type():
  """
  Argument at command line for model type, either 'baseline' for baseline model 
  or 's2s' for seq2seq model
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("model_type")
  args = parser.parse_args()
  return args.model_type 

if __name__ == "__main__":
    vocab = vocab.GloVe(name=config.vocab_source, 
                        dim=config.vocab_dim)
    use_gpu = torch.cuda.is_available()

    print("Using GPU:", use_gpu)
    print ('vocab dim', config.vocab_dim)

    parser = argparse.ArgumentParser()
    parser.add_argument("run_name")
    parser.add_argument("run_comment")
    parser.add_argument("epoch")
    args = parser.parse_args()
    model_type, run_comment, epoch = args.run_name, args.run_comment, args.epoch
    name = model_type + '-' + run_comment
    config.load_path = "outputs/def2vec/checkpoints/{}/epoch_{}/model_weights.torch".format(name, epoch)

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
                                   shuffle=config.shuffle,
                                   vocab_size = config.vocab_size)
    val_loader = get_data_loader(VAL_FILE,
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
            
            loss_object, loss_val = model.calculate_loss(
                                                    inputs, 
                                                    outputs, 
                                                    labels, 
                                                    words)
            loss_object.backward()
            optimizer.step()

            # print statistics
            running_loss += loss_val
            writer.add_scalar('loss', loss_val, total_iter)
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

                    optimizer.zero_grad()
                    outputs = model(inputs, lengths)

                    loss_object, loss_val = model.calculate_loss(inputs, outputs, labels, words)
                    val_loss += loss_val
                
                writer.add_scalar('val_loss', val_loss / len(val_loader), total_iter)
                print('Epoch: %d, batch: %d, val loss: %.4f' %
                             (epoch + 1, i + 1, val_loss / len(val_loader)))
            # increase iteration
            total_iter += 1
        
        name = config.run_name + '-' + config.run_comment
        out_dir = "outputs/def2vec/checkpoints/{}".format(config.run_name)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        out_path = "outputs/def2vec/checkpoints/{}/epoch_{}".format(config.run_name, epoch + 1)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        torch.save(model.state_dict(), out_path + "/" + config.save_path)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

    print('Finished Training')
