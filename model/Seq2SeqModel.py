import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN
from .seq2seq import Seq2seq

class Seq2SeqModel(Seq2seq):
    """ sequence-to-sequence architecture with specific encoder and decoder.
    """
    def __init__(self, config, decode_function=F.log_softmax):
        super(Seq2SeqModel, self).__init__(
            encoder = EncoderRNN(vocab_size = config.vocab_size,
                          max_len = config.max_len, 
                          hidden_size = config.hidden_size, 
                          embed_size = config.vocab_dim,
                          input_dropout_p=config.dropout,
                          dropout_p=config.dropout,
                          n_layers=config.num_layers,
                          bidirectional=config.use_bidirection,
                          rnn_cell=config.cell_type.lower(),
                          variable_lengths=False,
                          embedding=None, #randomly initialized,
                        ),
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
        )
                          
        self.encoder_hidden = None
