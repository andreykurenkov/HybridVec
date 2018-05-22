from __future__ import print_function
import os
from eval_scripts import evaluate_on_all
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
from model import Def2VecModel, Seq2SeqModel
from torch.autograd import Variable
import torchtext.vocab as vocab
from tensorboardX import SummaryWriter
from pytorch_monitor import monitor_module, init_experiment
from loader import *
from config import eval_config
import json
import argparse
from seq2seq import EncoderRNN, DecoderRNN
from collections import OrderedDict
from nltk.corpus import wordnet as wn



def get_args():
    """
    Gets the run_name, run_comment, and epoch of the model being evaluated
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("run_title")
    parser.add_argument("run_name")
    parser.add_argument("run_comment")
    parser.add_argument("epoch")
    parser.add_argument("--verbose", default=True)
    args = parser.parse_args()
    return (args.run_title, args.run_name, args.run_comment, args.epoch, args.verbose)
def load_config():
    """
    Load in the right config file from desired model to evaluate
    """
    run_title, run_name, run_comment, epoch, verbose = get_args()
    name = run_name + '-' + run_comment
    path = "outputs/{}/logs/{}/config.json".format(str(run_title), name)
    config = None
    with open(path) as f:
        config = dict(json.load(f))
        config = eval_config(config, run_name, run_comment, epoch, verbose)
    return (config,name)

def create_data():
  in_glove = open("data/glove/glove.6B.100d.txt", "r")

  out_glove = open("data/glove/5k_glove.6B.100d.txt", "w")

  count = 0
  for line in in_glove:
    if count == 5000: break
    out_glove.write(line)
    count += 1

  in_glove.close()
  out_glove.close()

def get_embeddings():
  config, name = load_config()

  TRAIN_FILE = 'data/glove/5k_glove.%s.%sd.txt'%(config.vocab_source,config.vocab_dim)
  if not os.path.exists(TRAIN_FILE):
    create_data()

  vocab_1 = vocab.GloVe(name=config.vocab_source, dim=config.vocab_dim)
  use_gpu = torch.cuda.is_available()
  print("Using GPU:", use_gpu)

  vocab_size = 50000
  vocab_reduced = True if vocab_size < 400000 else False
  encoder = EncoderRNN(vocab_size = vocab_size,
                      max_len = 200, 
                      hidden_size = config.hidden_size, 
                      embed_size = config.vocab_dim,
                      input_dropout_p=config.dropout,
                      dropout_p=config.dropout,
                      n_layers=2,
                      bidirectional=config.use_bidirection,
                      rnn_cell=config.cell_type.lower(),
                      variable_lengths=False,
                      embedding=None, #randomly initialized,
                      )

  decoder = DecoderRNN(vocab_size = vocab_size,
                      max_len = 200,
                      hidden_size = config.hidden_size,
                      n_layers=2,
                      rnn_cell=config.cell_type.lower(),
                      bidirectional=config.use_bidirection,
                      input_dropout_p=config.dropout,
                      dropout_p=config.dropout,
                      use_attention=config.use_attention
                      )


  model = Seq2SeqModel(encoder = encoder,
                      decoder = decoder
                      )
  model.load_state_dict(torch.load(config.save_path), strict = True)

  train_loader = get_data_loader(TRAIN_FILE,
                                 vocab_1,
                                 config.input_method,
                                 config.vocab_dim,
                                 batch_size = config.batch_size,
                                 num_workers = config.num_workers,
                                 shuffle=False,
                                 vocab_size=vocab_size)
  if use_gpu:
      model = model.cuda()
  model.train(False)

  out_embeddings = {}


  for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
      words, inputs, lengths, labels = data
      labels = Variable(labels)

      if use_gpu:
          inputs = inputs.cuda()
          labels = labels.cuda()

      (decoder_outputs, decoder_hidden, ret_dicts), encoder_hidden  = model(inputs, lengths)
      for idx, word in enumerate(words):
        out_embeddings[word] = encoder_hidden.cpu().data[idx, :]

  #np.save("eval/name-output_embeddings.npy".format(name), out_embeddings)
  return out_embeddings


def load_embeddings():
   config, name = load_config()
   a = np.load("./outputs/{}/embeddings/{}/out_embeddings.npy".format(config.title, name)).item()
   return a

def glove_embedding():
  vocab_glove = vocab.GloVe(name="6B", dim=100)
  mapping = {}
  for index, w in enumerate(vocab_glove.itos):
    if index > 5000: break
    mapping[w] = vocab_glove.vectors[index]
  return mapping


def cosine_similarity(x,y):
    return np.dot(x,y)/(np.sqrt(np.dot(x,x)) * np.sqrt(np.dot(y,y)))

def evaluate_antonyms(embeddings):
    #embedding is dictionary of word to embedding 
    running_total = 0
    count = 0 
    for word in embeddings.keys():
        embed = embeddings[word]

        #make sure the antonym can be indexed into
        antonym = None 
        if len(wordnet.synsets(word)) > 0 and len(wordnet.synsets(word)[0].lemmas()) > 0 and len(wordnet.synsets(word)[0].lemmas()[0].antonyms()) > 0:
            antonym = wordnet.synsets(word)[0].lemmas()[0].antonyms()[0].name()
        else: continue 

        if antonym is not None and antonym in embeddings:
            antonym_embed = embeddings[antonym]
            total = cosine_similarity(embed, antonym_embed)
            running_total += total 
            count += 1 

        else: continue

    running_total /= count 
    print('average cosine similarity loss is: ', running_total)


def main():
   glove_emb = glove_embedding()
   glove_dist = evaluate_antonyms(glove_emb)
   print ("The average distance of antonyms on GloVe embeddings is {}".format(glove_dist))

   s2s_emb = get_embeddings()
   s2s_dist = evaluate_antonyms(glove_emb)
   print ("The average distance of antonyms on HybridVec embeddings is {}".format(s2s_dist))



if __name__ == "__main__":
    main()
