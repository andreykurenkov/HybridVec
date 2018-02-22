from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable


class Def2VecModel(nn.Module):

  def __init__(self, input_size, output_size, hidden_size, num_layers):
    super(Def2VecModel, self).__init__()
    embed_size = hidden_size
    self.embed_size = embed_size
    self.num_layers = num_layers
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.gru = nn.GRU(embed_size, hidden_size, num_layers)
    self.embeddings = nn.Embedding(input_size, embed_size, padding_idx=0)
    self.output_layer = nn.Linear(hidden_size, output_size)

  def forward(self, defn):
    inputs = Variable(defn)
    h0 = Variable(torch.zeros(self.num_layers, self.hidden_size))
    gru_outputs, _ = self.gru(inputs, h0)
    our_embedding = self.output_layer(torch.mean(gru_outputs, axis=1))
    return our_embedding

  def loss(self, defn, glove_embedding):
    our_embedding = forward(defn)
    return nn.MSELoss(our_embedding, glove_embedding)
