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
from baseline import BaselineModel
from torch.autograd import Variable
import torchtext.vocab as vocab
from tensorboardX import SummaryWriter
from loader import *
import torch.nn.init as init
from tqdm import tqdm
from time import time
from config import train_config
from pytorch_monitor import monitor_module, init_experiment

DEBUG_LOG = True

config = train_config()
# total_avg_main_loss = 0
# total_avg_regular_loss = 0
# total_avg_glove_loss = 0

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

def calculate_loss(inputs, outputs, labels, criterions, input_embeddings, defn_embeddings):
    loss = 0
    count = 0
    # global total_avg_main_loss 
    # global total_avg_regular_loss
    # global total_avg_glove_loss 
    criterion, reg_criterion = criterions[0], criterions[1]
    for word_idx in range(list(inputs.size())[1]):
        label = Variable(inputs[:,word_idx])
        if use_gpu:
            label = label.cuda()
        loss+= criterion(outputs, label)
        count+=1.0
    loss/=count

    # total_avg_main_loss += loss 
    reg_loss = reg_criterion(defn_embeddings, input_embeddings)
    #sum the square differences and average across the batch
    reg_loss = torch.sum(reg_loss, 1)
    reg_loss = torch.mean(reg_loss)
    reg_loss *= config.reg_weight 
    reg_loss /= defn_embeddings.size()[0] 

    loss += reg_loss
    # total_avg_regular_loss += reg_loss
    if config.glove_aux_loss: #add regression on original glove labels into loss 
      glove_criterion = criterions[2]
      glove_loss = glove_criterion(defn_embeddings, labels)
      #sum the square differences and average across the batch
      glove_loss = torch.sum(glove_loss, 1)
      glove_loss = torch.mean(glove_loss)

      glove_loss *= config.glove_aux_weight
      glove_loss /= defn_embeddings.size()[0]
      loss += glove_loss
      # total_avg_glove_loss += glove_loss

    #calcualte embed magnitudes, debug 
    # print_embed_magnitudes(input_embeddings, defn_embeddings, labels)
    return loss 

def print_embed_magnitudes(input_embeddings, defn_embeddings, labels):
  input_norm = torch.norm(input_embeddings, 2, 1)
  def_norm = torch.norm(defn_embeddings, 2, 1)
  glove_norm = torch.norm(labels, 2, 1)

  print('these are norms')
  print('input norm')
  print(torch.mean(input_norm))
  # print(input_norm)
  print('def norm')
  print(torch.mean(def_norm))
  # print(def_norm)
  print('glove norm')
  print(torch.mean(glove_norm))
  # print(glove_norm)

  print('these are some embeddings')
  print(input_embeddings[0])
  print(defn_embeddings[0])
  print(labels[0])

if __name__ == "__main__":
    vocab = vocab.GloVe(name=config.vocab_source, dim=config.vocab_dim)
    use_gpu = torch.cuda.is_available()
    #use_gpu = False
    print("Using GPU:", use_gpu)
    print ('vocab dim', config.vocab_dim)
    
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


    criterion = nn.NLLLoss() #use multi label loss across unigram bag of words model
    reg_criterion = nn.MSELoss(reduce=False)
    if config.glove_aux_loss: glove_criterion = nn.MSELoss(reduce=False)
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
        # if epoch != 0:
        #   print('Running average losses after epoch '+ str(epoch) + ' were: ')
        #   print('Total avg main: ', str(total_avg_main_loss/(epoch)))
        #   print('Total avg regular: ', str(total_avg_regular_loss/(epoch)))
        #   print('Total avg glovee: ', str(total_avg_glove_loss/(epoch)))

        running_loss = 0.0
        start = time()
        print("Epoch", epoch)
        for i, data in enumerate(train_loader, 0):
            words, inputs, lengths, labels = data
            labels = Variable(labels)
            # if i % 1 == 0 and i != 0: 
            #   print ('after 1 batch, runnning averages per batch are')
            #   print('Total avg main: ', str(total_avg_main_loss/(i)))
            #   print('Total avg regular: ', str(total_avg_regular_loss/(i)))
            #   print('Total avg glovee: ', str(total_avg_glove_loss/(i)))

            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs, lengths)

            w_indices = np.array([vocab.stoi[w] + 1 for w in words])
            w_indices[w_indices > config.vocab_size] = 0 #for vocab size 
            input_embeddings = Variable(model.embeddings.weight.data[w_indices])
            defn_embeddings = model.defn_embed

            if use_gpu:
                defn_embeddings = defn_embeddings.cuda()
                input_embeddings = input_embeddings.cuda()
                labels = labels.cuda()
            
            criterions = [criterion, reg_criterion, glove_criterion] if config.glove_aux_loss else [criterion, reg_criterion]
            loss = calculate_loss(inputs, outputs, labels, criterions, input_embeddings, defn_embeddings)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            writer.add_scalar('loss', loss.data[0], total_iter)
            if embed_outs is None:
                embed_outs = model.defn_embed.data.cpu()
                embed_labels = words
            else:
                embed_outs = torch.cat([embed_outs, model.defn_embed.data.cpu()])
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

                    optimizer.zero_grad()
                    outputs = model(inputs, lengths)

                    w_indices = np.array([vocab.stoi[w] + 1 for w in words])
                    w_indices[w_indices > config.vocab_size] = 0 #for vocab size 
                    input_embeddings = Variable(model.embeddings.weight.data[w_indices])
                    defn_embeddings = model.defn_embed

                    if use_gpu:
                        defn_embeddings = defn_embeddings.cuda()
                        input_embeddings = input_embeddings.cuda()

                    loss = calculate_loss(inputs, outputs, criterion, reg_criterion, input_embeddings, defn_embeddings)
                    val_loss += loss.data[0]
                writer.add_scalar('val_loss', val_loss / len(val_loader), total_iter)
                print('Epoch: %d, batch: %d, val loss: %.4f' %
                             (epoch + 1, i + 1, val_loss / len(val_loader)))

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