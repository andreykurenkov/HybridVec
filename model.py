from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext.vocab as vocab
import numpy as np
from torch.autograd import Variable


class Def2VecModel(nn.Module):

  def __init__(self, vocab_size=100000, output_size=300, hidden_size=128, num_layers=2):
    super(Def2VecModel, self).__init__()
    embed_size = hidden_size
    self.embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=0)
    self.embed_size = embed_size
    self.num_layers = num_layers
    self.vocab_size = vocab_size
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.gru = nn.GRU(embed_size, hidden_size, num_layers)
    self.output_layer = nn.Linear(hidden_size, output_size)

  def forward(self, defn):
    inputs = Variable(defn)
    batch_size, input_size = inputs.shape
    embed = self.embeddings(inputs.view(-1, input_size)).view(batch_size, input_size, -1)
    h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
    gru_outputs, _ = self.gru(embed, h0)
    our_embedding = self.output_layer(torch.mean(gru_outputs, dim=1))
    return our_embedding
