from __future__ import print_function
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
from baseline import BaselineModel

# # runs over all the words in glove and returns embeddings for each one
# def get_embeddings():
# 	#check if there is a local file first


# 	# if not run the model on all the glove files and print the scores

def get_args():
    """
    Gets the run_name, run_comment, and epoch of the model being evaluated
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name")
    parser.add_argument("run_comment")
    parser.add_argument("epoch")
    parser.add_argument("--verbose", default=True)
    args = parser.parse_args()
    return (args.run_name, args.run_comment, args.epoch, args.verbose)
def load_config():
    """
    Load in the right config file from desired model to evaluate
    """
    run_name, run_comment, epoch, verbose = get_args()
    name = run_name + '-' + run_comment
    path = "outputs/def2vec/logs/{}/config.json".format(name)
    config = None
    with open(path) as f:
        config = dict(json.load(f))
        config = eval_config(config, run_name, run_comment, epoch, verbose)
    return (config,name)

def get_embeddings():
  config, name = load_config()
  TRAIN_FILE = 'data/glove/train_glove.%s.%sd.txt'%(config.vocab_source,config.vocab_dim)
  vocab_1 = vocab.GloVe(name=config.vocab_source, dim=config.vocab_dim)
  use_gpu = torch.cuda.is_available()
  print("Using GPU:", use_gpu)

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
      	out_embeddings[word] = model.defn_embed.cpu().data[idx, :]

  # out_dir = "outputs/def2vec/checkpoints/{}".format(name)
  #       if not os.path.exists(out_dir):
  #           os.makedirs(out_dir)
  np.save("./outputs/def2vec/checkpoints/{}/output_embeddings.npy".format(name), out_embeddings)
  return out_embeddings


def load_embeddings():
	a = np.load('./eval/out_embeddings.npy').item()
	return a

def main():
	embeddings = get_embeddings()
	#embeddings = load_embeddings()
	evaluate_on_all(embeddings)




if __name__ == "__main__":
	main()