from __future__ import print_function
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import collections
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchtext.vocab as vocab
import argparse
import shutil
from tqdm import tqdm
from model import Def2VecModel, Seq2SeqModel
from torch.autograd import Variable
from loader import *
import json
from seq2seq import EncoderRNN, DecoderRNN
from config import eval_config



VOCAB_DIM = 100
VOCAB_SOURCE = '6B'
vocab_1 = vocab.GloVe(name=VOCAB_SOURCE, dim=VOCAB_DIM)
embeddings_file = 'data/nmt/glove/glove_s2s_train.txt'



with open(embeddings_file,'r') as output:
	embeds = output.readlines()