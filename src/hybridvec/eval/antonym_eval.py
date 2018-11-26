from __future__ import print_function
from eval_scripts import evaluate_on_all
from tqdm import tqdm
import os, sys
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
from baseline import BaselineModel
from nltk.corpus import wordnet

# # runs over all the words in glove and returns embeddings for each one
# def get_embeddings():
#   #check if there is a local file first


#   # if not run the model on all the glove files and print the scores

def get_args():
    """
    Gets the run_name, run_comment, and epoch of the model being evaluated
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type")
    parser.add_argument("run_name")
    parser.add_argument("run_comment")
    parser.add_argument("epoch")
    parser.add_argument("--verbose", default=True)
    args = parser.parse_args()
    return (args.model_type, args.run_name, args.run_comment, args.epoch, args.verbose)

def load_config():
    """
    Load in the right config file from desired model to evaluate
    """
    model_type, run_name, run_comment, epoch, verbose = get_args()
    name = run_name + '-' + run_comment
    if model_type == "s2s": 
      run_title = "seq2seq"
    else:
      run_title = "def2vec"
    path = "outputs/{}/logs/{}/config.json".format(run_title, name)
    config = None
    with open(path) as f:
        config = dict(json.load(f))
        config = eval_config(config, run_name, run_comment, epoch, verbose)
    return (config, name, model_type)

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
  config, name, model_type = load_config()
  TRAIN_FILE = 'data/glove/5k_glove.%s.%sd.txt'%(config.vocab_source,config.vocab_dim)
  if not os.path.exists(TRAIN_FILE):
    create_data()
  vocab_1 = vocab.GloVe(name=config.vocab_source, dim=config.vocab_dim)
  use_gpu = torch.cuda.is_available()

  if model_type == 'baseline': model = BaselineModel(vocab, config=config, use_cuda = use_gpu)
  elif model_type == 's2s': model = Seq2SeqModel(config)

  model.load_state_dict(torch.load(config.save_path), strict = True)

  train_loader = get_data_loader(TRAIN_FILE,
                   vocab_1,
                   config.input_method,
                   config.vocab_dim,
                   batch_size = config.batch_size,
                   num_workers = config.num_workers,
                   shuffle=False,
                   vocab_size=config.vocab_size)
  if use_gpu:
      model = model.cuda()
  model.train(False)

  running_loss = 0.0
  n_batches = 0
  out_embeddings = {}
  pred_defns = {}
  out_defns = {}

  for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
      words, inputs, lengths, labels = data
      labels = Variable(labels)

      if use_gpu:
          inputs = inputs.cuda()
          labels = labels.cuda()

      outputs = model(inputs, lengths)
      for idx, word in enumerate(words):
        out_embeddings[word] = model.get_def_embeddings()[idx, :]

  # out_dir = "outputs/def2vec/checkpoints/{}".format(name)
  #       if not os.path.exists(out_dir):
  #           os.makedirs(out_dir)
  return out_embeddings


def load_embeddings():
  a = np.load('./eval/out_embeddings.npy').item()
  return a

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

def glove_embedding():
  vocab_glove = vocab.GloVe(name="6B", dim=100)
  mapping = {}
  for index, w in enumerate(vocab_glove.itos):
    if index > 5000: break 
    mapping[w] = vocab_glove.vectors[index]
  return mapping

def main():
  embeddings = get_embeddings()
  #embeddings = load_embeddings()
  glove_embeddings = glove_embedding()
  print('average coefficient for embeddings')
  evaluate_antonyms(embeddings)
  print('average coefficient for glove')
  evaluate_antonyms(glove_embeddings)


if __name__ == "__main__":
  main()