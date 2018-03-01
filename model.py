from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext.vocab as vocab
import numpy as np
from torch.autograd import Variable


class Def2VecModel(nn.Module):

  def __init__(self, 
               vocab, 
               output_size=300, 
               hidden_size=128, 
               embed_size=300, 
               num_layers=2,
               use_cuda=False,
               use_packing=False):
    super(Def2VecModel, self).__init__()
    self.use_packing = use_packing
    self.use_cuda = use_cuda
    self.vocab_size = len(vocab.stoi)
    self.embeddings = nn.Embedding(self.vocab_size, embed_size, padding_idx=0)
    self.embeddings.weight.data.copy_(vocab.vectors)
    self.embed_size = embed_size
    self.num_layers = num_layers
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
    self.output_layer = nn.Linear(hidden_size, output_size)

  def forward(self, inputs, lengths):
    inputs = Variable(inputs)
    batch_size, input_size = inputs.shape
    embed = self.embeddings(inputs.view(-1, input_size)).view(batch_size, input_size, -1)
    if self.use_packing:
      embed = nn.utils.rnn.pack_padded_sequence(embed, lengths, batch_first=True)
    h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
    if self.use_cuda:
      h0 = h0.cuda()
    gru_outputs, _ = self.gru(embed, h0)
    if self.use_packing:
      gru_outputs, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(
                                        gru_outputs, batch_first=True)
    our_embedding = self.output_layer(torch.mean(gru_outputs, dim=1))
    return our_embedding
