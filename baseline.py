from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext.vocab as vocab
from torch.autograd import Variable
import numpy as np


class BaselineModel(nn.Module):

  def __init__(self,
               vocab,
               output_size=100,
               hidden_size=150,
               embed_size=100,
               num_layers=2,
               dropout=0.0,
               use_bidirection=True,
               use_attention=True,
               cell_type='LSTM',
               use_cuda=True,
               use_packing=False,
               max_length=784):
    super(BaselineModel, self).__init__()
    self.use_packing = use_packing
    self.use_cuda = use_cuda
    self.vocab_size = len(vocab.stoi)
    self.embeddings = nn.Embedding(self.vocab_size + 1, embed_size, padding_idx=0)
    #no longer copying glove 
    # self.embeddings.weight.data[1:,:].copy_(vocab.vectors) #no longer copying glove, randomly initialize weights
    # self.embeddings.weight.data[0:,:] = 0 #regularizing against input embeddings doesn't make sense if these are all zero -- esp since the gradient on these is 0 bc we regularize against them
    self.embed_size = embed_size
    self.num_layers = num_layers
    # needs to be the same as number of words used in the definitions, so same as embedding size 
    self.output_size = self.vocab_size
    self.hidden_size = int(embed_size/2) if use_attention else embed_size
    self.use_attention = use_attention
    self.use_bidirection = use_bidirection
    self.cell_type = cell_type
    if cell_type == 'GRU':
        self.cell = nn.GRU(embed_size,
                           self.hidden_size,
                           num_layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=use_bidirection)
    elif cell_type == 'LSTM':
        self.cell = nn.LSTM(embed_size,
                           self.hidden_size,
                           num_layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=use_bidirection)
    elif cell_type == 'RNN':
        self.cell = nn.RNN(embed_size,
                           self.hidden_size,
                           num_layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=use_bidirection)
    else:
        self.baseline = nn.Linear(self.embed_size, self.hidden_size)
    if use_attention:
        self.attn = nn.Linear((2 if use_bidirection else 1) * self.hidden_size, 1)
        self.attn_softmax = nn.Softmax(dim=1)

    #apply last matrix operation to compute dotproduct log likelihood of hidden representation with 
    #each word in definitional vocabulary 
    self.output_layer = nn.Linear((2 if use_bidirection else 1) * self.hidden_size, self.output_size)

  def forward(self, inputs, lengths = None, return_attn = False):
    inputs = Variable(inputs)
    batch_size, input_size = inputs.shape

    embed = self.embeddings(inputs.view(-1, input_size)).view(batch_size, input_size, -1)

    if self.use_packing:
      embed = nn.utils.rnn.pack_padded_sequence(embed, lengths, batch_first=True)
    h0 = Variable(torch.zeros(self.num_layers * (2 if self.use_bidirection else 1),
                              batch_size, self.hidden_size))
    c0 = Variable(torch.zeros(self.num_layers * (2 if self.use_bidirection else 1),
                              batch_size, self.hidden_size))
    if self.use_cuda:
      h0 = h0.cuda()
      c0 = c0.cuda()
    if self.cell_type == 'LSTM':
        cell_outputs, _ = self.cell(embed, (h0, c0))
    elif self.cell_type:
        cell_outputs, _ = self.cell(embed, h0)
    else:
        cell_outputs = self.baseline(embed)
    if self.use_packing:
      cell_outputs, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(
                                        cell_outputs, batch_first=True)
    if self.use_attention:
        logits = self.attn(cell_outputs)
        softmax = self.attn_softmax(logits)
        self.defn_embed = torch.sum(softmax * cell_outputs, dim=1)
    else:
        self.defn_embed = torch.mean(cell_outputs, dim=1)
    likelihood_scores = self.output_layer(self.defn_embed) #calculate score for definition embedding --> to max unigram prob for words in definition
    softmax = nn.LogSoftmax(dim = 1)
    probability_unigram = softmax(likelihood_scores)

    if return_attn:
        return probability_unigram, softmax
    else:
        return probability_unigram
