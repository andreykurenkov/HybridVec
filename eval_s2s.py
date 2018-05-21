from __future__ import print_function
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

DEBUG_LOG = False


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
    return (args.run_name, args.run_comment, args.epoch, args.verbose)
def load_config():
    """
    Load in the right config file from desired model to evaluate
    """
    run_title, run_name, run_comment, epoch, verbose = get_args()
    name = run_name + '-' + run_comment
    path = "outputs/{}/logs/{}/config.json".format(run_title, name)
    config = None
    with open(path) as f:
        config = dict(json.load(f))
        config = eval_config(config, run_name, run_comment, epoch, verbose)
    return config

def get_word(word):
    return vocab.vectors[vocab.stoi[word]]

def closest(vec, n=10):
    """
    Find the closest words for a given vector
    """
    all_dists = [(w, torch.dist(vec, get_word(w))) for w in vocab.itos]
    return sorted(all_dists, key=lambda t: t[1])[:n]

def write_output(f, pred, inputs, words, vocab_size):
  #print (pred)  
  for w in range(len(words)):
    f.write(words[w] +"\n")
    definition_input = [( vocab.itos[i - 1] if (i>0 and i<=vocab_size+1) else str(i)) for i in (inputs[w])]
    definition_input = "input definition: " + " ".join(definition_input)
    f.write(definition_input+ "\n")

    len_pred = min(len(inputs[w]), len(pred[w]))
    dfn_pred = [(vocab.itos[pred[w][i] - 1] if (pred[w][i]>0 and pred[w][i]<=vocab_size+1) else str(pred[w][i])) for i in range(len_pred) ]
    dfn_pred = "predicted definition: " + " ".join(dfn_pred) + "\n"
    f.write(dfn_pred)
    f.write("\n\n")




def get_loss_nll(acc_loss, norm_term):
        if isinstance(acc_loss, int):
            return 0
        # total loss for all batches
        loss = acc_loss.data
        loss /= norm_term
        loss =  (Variable(loss).data)[0]
        #print (type(loss))
        return loss

if __name__ == "__main__":
  f = open('input-output.txt','w')
  config = load_config()
  TEST_FILE = 'data/glove/test_glove.%s.%sd.txt'%(config.vocab_source,config.vocab_dim)
  vocab = vocab.GloVe(name=config.vocab_source, dim=config.vocab_dim)
  use_gpu = torch.cuda.is_available()
  print("Using GPU:", use_gpu)

  vocab_size = 50000
  vocab_reduced = True if vocab_size < 400000 else False

  embedding = nn.Embedding(vocab_size+3, config.vocab_dim, padding_idx=0) #+2 for the start and end symbol and +1 for unk token
  embedding.weight.data[0,:] = 0
  if config.use_glove:
      embedding.weight.data[1:vocab_size,:].copy_(vocab.vectors[:vocab_size-1,:])

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
                      embedding=embedding, #randomly initialized,
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




  # encoder_dict, decoder_dict = load_dicts(torch.load(config.save_path))
  # encoder.load_state_dict(encoder_dict)
  # decoder.load_state_dict(decoder_dict)

  model = Seq2SeqModel(encoder = encoder,
                      decoder = decoder
                      )
  print (model.state_dict().keys())
  model.load_state_dict(torch.load(config.save_path), strict = True)

  test_loader = get_data_loader(TEST_FILE,
                                 vocab,
                                 config.input_method,
                                 config.vocab_dim,
                                 batch_size = config.batch_size,
                                 num_workers = config.num_workers,
                                 shuffle=False,
                                 vocab_size=vocab_size)

  if use_gpu:
      model = model.cuda()
  criterion = nn.NLLLoss()
  model.train(False)

  running_loss = 0.0
  n_batches = 0
  out_embeddings = {}
  pred_defns = {}
  out_defns = {}


  for i, data in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
      words, inputs, lengths, labels = data
      labels = Variable(labels)

      if use_gpu:
          inputs = inputs.cuda()
          labels = labels.cuda()

      (decoder_outputs, decoder_hidden, ret_dicts), encoder_hidden  = model(inputs, lengths)
      #print (i, "out of data")
      acc_loss = 0
      norm_term = 0


      for step, step_output in enumerate(decoder_outputs):
          batch_size = inputs.shape[0]
          if step > (inputs.shape[1] -1): continue
          labeled_vals = Variable((inputs).long()[:, step])
          labeled_vals.requires_grad = False
          pred = step_output.contiguous().view(batch_size, -1)
          acc_loss += criterion(pred, labeled_vals)
          norm_term += 1


      batch_loss = get_loss_nll(acc_loss, norm_term)
      running_loss += batch_loss
      n_batches += 1


      #write to fil
      word_preds = []
      for soft in decoder_outputs:
          vals, idx = soft.max(1)
          word_preds.append(idx.unsqueeze(1))
      preds = torch.cat(word_preds, dim=1).cpu().data
      write_output(f, preds, inputs, words, vocab_size)


      for word, embed, inp in zip(words,
                                  encoder_hidden.data.cpu(),
                                  inputs.cpu()):
          out_embeddings[word] = embed.numpy()
          out_defns[word] = " ".join([vocab.itos[i - 1] for i in inp])

  f.close()
  print("L2 loss:", running_loss / n_batches)
  np.save("eval/out_embeddings.npy", out_embeddings)
  #np.save("eval/out_attns.npy", out_attns)
  np.save("eval/out_defns.npy", out_defns)
