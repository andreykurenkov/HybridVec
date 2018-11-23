""" A base class for RNN. """
import torch.nn as nn


class BaseRNN(nn.Module):
    r"""
    Applies a multi-layer RNN to an input sequence.
    Note:
        Do not use this class directly, use one of the sub classes.
    Args (pass by config):
        vocab_size (int): size of the vocabulary
        max_len (int): maximum allowed length for the sequence to be processed
        hidden_size (int): number of features in the hidden state `h`
        input_dropout_p (float): dropout probability for the input sequence
        dropout_p (float): dropout probability for the output sequence
        n_layers (int): number of recurrent layers
        rnn_cell (str): type of RNN cell (Eg. 'LSTM' , 'GRU')

    Inputs: ``*args``, ``**kwargs``
        - ``*args``: variable length argument list.
        - ``**kwargs``: arbitrary keyword arguments.

    Attributes:
        SYM_MASK: masking symbol
        SYM_EOS: end-of-sequence symbol
    """
    SYM_MASK = "MASK"
    SYM_EOS = "EOS"

    def __init__(self, config):
        super(BaseRNN, self).__init__()
        self.vocab_size = config.vocab_size
        self.max_len = config.max_len
        self.hidden_size = config.hidden_size
        self.n_layers = config.n_layers
        self.bidirectional = config.use_bidirection,

        # input/output dropout probability
        self.input_dropout_p = config.dropout
        self.dropout_p = config.dropout

        # input dropout hook
        self.input_dropout = nn.Dropout(p = config.dropout)

        if config.rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif config.rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(config.rnn_cell))

    def forward(self, *args, **kwargs):
        raise NotImplementedError()
