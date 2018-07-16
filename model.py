from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torchtext.vocab as vocab
from torch.autograd import Variable
import numpy as np
from seq2seq import EncoderRNN, DecoderRNN



class Seq2SeqModel(nn.Module):
    def __init__(self, config, decode_function=F.log_softmax):
        super(Seq2SeqModel, self).__init__()
        vocab_reduced = True if config.vocab_size < 400000 else False
        encoder = EncoderRNN(vocab_size = config.vocab_size,
                          max_len = config.max_len, 
                          hidden_size = config.hidden_size, 
                          embed_size = config.vocab_dim,
                          input_dropout_p=config.dropout,
                          dropout_p=config.dropout,
                          n_layers=self.num_layers,
                          bidirectional=config.use_bidirection,
                          rnn_cell=config.cell_type.lower(),
                          variable_lengths=False,
                          embedding=None, #randomly initialized,
                          )

        decoder = DecoderRNN(vocab_size = config.vocab_size,
                          max_len = config.max_len,
                          hidden_size = config.hidden_size,
                          n_layers= config.num_layers,
                          rnn_cell=config.cell_type.lower(),
                          bidirectional=config.use_bidirection,
                          input_dropout_p=config.dropout,
                          dropout_p=config.dropout,
                          use_attention=config.use_attention
                          )
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_hidden = None

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
        #self.encoder_hidden = encoder_hidden[self.encoder.n_layers - 1]
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=F.log_softmax,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result, encoder_hidden[self.encoder.n_layers - 1]

    
    def get_loss_nll(self, acc_loss, norm_term):
        if isinstance(acc_loss, int):
            return 0
        # total loss for all batches
        loss = acc_loss.data
        loss /= norm_term
        loss =  (Variable(loss).data)[0]
        #print (type(loss))
        return loss


    def calculate_loss(self, output):
      (decoder_outputs, decoder_hidden, ret_dicts), encoder_hidden  = output

      criterion = nn.NLLLoss()
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


      if type(acc_loss) is int:
          raise ValueError("No loss to back propagate.")

      batch_loss = get_loss_nll(acc_loss, norm_term)

      return acc_loss, batch_loss
      # print statistics

    def get_def_embeddings(self, output):
      (decoder_outputs, decoder_hidden, ret_dicts), encoder_hidden  = output
      return encoder_hidden.data.cpu()


class Def2VecModel(nn.Module):

  def __init__(self,
               vocab,
               output_size=100,
               hidden_size=150,
               embed_size=100,
               num_layers=2,
               dropout=0.0,
               use_bidirection=True,
               use_attention=True,
               cell_type='GRU',
               use_cuda=True,
               use_packing=False,
               max_length=784):
    super(Def2VecModel, self).__init__()
    self.use_packing = use_packing
    self.use_cuda = use_cuda
    self.vocab_size = len(vocab.stoi)
    self.embeddings = nn.Embedding(self.vocab_size + 1, embed_size, padding_idx=0)
    self.embeddings.weight.data[1:,:].copy_(vocab.vectors)
    self.embeddings.weight.data[0:,:] = 0
    self.embed_size = embed_size
    self.num_layers = num_layers
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.use_attention = use_attention
    self.use_bidirection = use_bidirection
    self.cell_type = cell_type
    if cell_type == 'GRU':
        self.cell = nn.GRU(embed_size,
                           hidden_size,
                           num_layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=use_bidirection)
    elif cell_type == 'LSTM':
        self.cell = nn.LSTM(embed_size,
                           hidden_size,
                           num_layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=use_bidirection)
    elif cell_type == 'RNN':
        self.cell = nn.RNN(embed_size,
                           hidden_size,
                           num_layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=use_bidirection)
    else:
        self.baseline = nn.Linear(embed_size, hidden_size)
    if use_attention:
        self.attn = nn.Linear((2 if use_bidirection else 1) * hidden_size, 1)
        self.attn_softmax = nn.Softmax(dim=1)
    self.output_layer = nn.Linear((2 if use_bidirection else 1) * hidden_size, output_size)

  def forward(self, inputs, lengths = None, return_attn = False):
    inputs = Variable(inputs)
    batch_size, input_size = inputs.shape
    embed = self.embeddings(inputs.view(-1, input_size)).view(batch_size, input_size, -1)
    if self.use_packing:
      embed = nn.utils.rnn.pack_padded_sequence(embed, lengths, batch_first=True)
    h0 = Variable(torch.zeros(self.num_layers * (2 if self.use_bidirection else 1),
                              batch_size, self.hidden_size))
    if self.use_cuda:
      h0 = h0.cuda()
    if self.cell_type:
        cell_outputs, _ = self.cell(embed, h0)
    else:
        cell_outputs = self.baseline(embed)
    if self.use_packing:
      cell_outputs, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(
                                        cell_outputs, batch_first=True)
    if self.use_attention:
        logits = self.attn(cell_outputs)
        softmax = self.attn_softmax(logits)
        mean = torch.sum(softmax * cell_outputs, dim=1)
    else:
        mean = torch.mean(cell_outputs, dim=1)
    our_embedding = self.output_layer(mean)
    if return_attn:
        return our_embedding, softmax
    else:
        return our_embedding
