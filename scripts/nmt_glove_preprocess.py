from __future__ import print_function
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import collections
import torch
import torch.optim as optim
import torch.nn as nn

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import numpy as np
import torchtext.vocab as vocab
import argparse
import shutil
from tqdm import tqdm
from torch.autograd import Variable

from hybridvec.loader import *
from hybridvec.models import *

import json
from hybridvec.config import *

VOCAB_DIM = 100
VOCAB_SOURCE = '6B'

if __name__ == "__main__":

    config = load_config()
    vocab = vocab.GloVe(name=VOCAB_SOURCE, dim=VOCAB_DIM)
    use_gpu = torch.cuda.is_available()
    print("Using GPU:", use_gpu)
    TRAIN_FILE = 'data/glove/train_glove.%s.%sd.txt'%(config.vocab_source,config.vocab_dim)
    FULL_FILE = 'data/glove/glove.%s.%sd.txt'%(config.vocab_source,config.vocab_dim)

    if config.model_type == 'baseline':
        model = BaselineModel(vocab, config=config, use_cuda = use_gpu)

    elif config.model_type == 'seq2seq':
        encoder = EncoderRNN(config = config,
                            variable_lengths = False,
                            embedding = None)

        decoder = DecoderRNN(config = config)
        model = Seq2seq(encoder = encoder,
                        decoder=decoder)

    model_path = get_model_path(config)
    model.load_state_dict(torch.load(model_path))

    if config.train_data_flag:
        data_loader = get_data_loader(TRAIN_FILE,
                                  vocab,
                                  INPUT_METHOD_ONE,
                                  VOCAB_DIM,
                                  batch_size = config.batch_size,
                                  num_workers = 8,
                                  shuffle=False,
                                  vocab_size = config.vocab_size)
        output_file = 'data/nmt/glove/glove_baseline_train.txt'

    else:
        data_loader = get_data_loader(FULL_FILE,
                                  vocab,
                                  INPUT_METHOD_ONE,
                                  VOCAB_DIM,
                                  batch_size = config.batch_size,
                                  num_workers = 8,
                                  shuffle=False,
                                  vocab_size = config.vocab_size)
        output_file = 'data/nmt/glove/glove_baseline_full.txt'

    if use_gpu:
        model = model.cuda()

    # shutil.copyfile(provided_file,output_file)
    with open(output_file,'a') as output:
        for i, data in tqdm(enumerate(data_loader, 0), total=len(data_loader)):
            words, inputs, lengths, labels = data
            labels = Variable(labels)

            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs, lengths)    #.cpu().data.numpy()
            defn_embeds = model.get_def_embeddings(outputs)  #.cpu().data.numpy()
            for i,word in enumerate(words):
                vec_str = " ".join([str(x) for x in defn_embeds[i]])
                output.write('%s %s\n'%(words[i],vec_str))

