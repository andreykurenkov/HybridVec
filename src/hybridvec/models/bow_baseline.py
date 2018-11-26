
import numpy as np
import torch
import torch.nn as nn
import torchtext.vocab as vocab
from torch.autograd import Variable

class BaselineModel(nn.Module):

  def __init__(self,
               vocab,
               config,
               use_cuda=True,
               max_length=784,
               ):
    super(BaselineModel, self).__init__()

    self.use_packing = config.packing
    self.use_cuda = use_cuda
    self.vocab = vocab
    self.vocab_size = config.vocab_size
    self.embed_size = config.vocab_dim
    self.embeddings = nn.Embedding(self.vocab_size + 1, self.embed_size, padding_idx=0)
    self.reg_weight = config.reg_weight
    self.glove_aux_loss = config.glove_aux_loss
    self.glove_aux_weight = config.glove_aux_weight
    #no longer copying glove, randomly initialize weights
    if config.use_glove_init:
      self.embeddings.weight.data[1:,:].copy_(vocab.vectors[:self.vocab_size, :]) 
    self.embeddings.weight.data[0,:] = 0 #set to 0 for unk 
    self.num_layers = config.num_layers

    # needs to be the same as number of words used in the definitions, so same as vocab size 
    self.output_size = self.vocab_size
    self.use_bidirection = config.use_bidirection
    self.hidden_size = int(self.embed_size/2) if self.use_bidirection else self.embed_size
    self.use_attention = config.use_attention
    self.cell_type = config.cell_type
    self.dropout = config.dropout
    if self.cell_type == 'GRU':
        self.cell = nn.GRU(self.embed_size,
                           self.hidden_size,
                           self.num_layers,
                           batch_first=True,
                           dropout=self.dropout,
                           bidirectional=self.use_bidirection)
    elif self.cell_type == 'LSTM':
        self.cell = nn.LSTM(self.embed_size,
                           self.hidden_size,
                           self.num_layers,
                           batch_first=True,
                           dropout=self.dropout,
                           bidirectional=self.use_bidirection)
    elif self.cell_type == 'RNN':
        self.cell = nn.RNN(self.embed_size,
                           self.hidden_size,
                           self.num_layers,
                           batch_first=True,
                           dropout=self.dropout,
                           bidirectional=self.use_bidirection)
    else:
        self.baseline = nn.Linear(self.embed_size, self.hidden_size)
    if self.use_attention:
        self.attn = nn.Linear((2 if self.use_bidirection else 1) * self.hidden_size, 1)
        self.attn_softmax = nn.Softmax(dim=1)
    #apply last matrix operation to compute dotproduct log likelihood of hidden representation with 
    #each word in definitional vocabulary 
    self.output_layer = nn.Linear((2 if self.use_bidirection else 1) * self.hidden_size, self.output_size)

    #initialize loss criterions
    criterion = nn.NLLLoss() #use multi label loss across unigram bag of words model
    reg_criterion = nn.MSELoss(reduce=False)
    if config.glove_aux_loss: glove_criterion = nn.MSELoss(reduce=False)
    self.criterions = [criterion, reg_criterion, glove_criterion] if config.glove_aux_loss else [criterion, reg_criterion]


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

  def calculate_loss(self, inputs, outputs, labels, words):

    #get input embeddings and definition embeddings for this batch of words
    w_indices = np.array([self.vocab.stoi[w] + 1 for w in words])
    w_indices[w_indices > self.vocab_size] = 0 #for vocab size 
    input_embeddings = Variable(self.embeddings.weight.data[w_indices])
    defn_embeddings=self.defn_embed

    #cuda shenanigans if need be 
    if self.use_cuda:
        defn_embeddings = defn_embeddings.cuda()
        input_embeddings = input_embeddings.cuda()

    loss = 0
    count = 0

    criterion, reg_criterion = self.criterions[0], self.criterions[1]
    for word_idx in range(list(inputs.size())[1]):
        label = Variable(inputs[:,word_idx])
        if self.use_cuda:
            label = label.cuda()
        loss+= criterion(outputs, label)
        count+=1.0
    loss/=count

    reg_loss = reg_criterion(defn_embeddings, input_embeddings)
    #sum the square differences and average across the batch
    reg_loss = torch.sum(reg_loss, 1)
    reg_loss = torch.mean(reg_loss)
    reg_loss *= self.reg_weight 
    reg_loss /= defn_embeddings.size()[0] 

    loss += reg_loss
    if self.glove_aux_loss: #add regression on original glove labels into loss 
      glove_criterion = self.criterions[2]
      glove_loss = glove_criterion(defn_embeddings, labels)
      #sum the square differences and average across the batch
      glove_loss = torch.sum(glove_loss, 1)
      glove_loss = torch.mean(glove_loss)

      glove_loss *= self.glove_aux_weight
      glove_loss /= defn_embeddings.size()[0]
      loss += glove_loss

    return loss, loss.data[0]

  def get_def_embeddings(self, output=None):
    return self.defn_embed.data.cpu()
