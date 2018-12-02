from tqdm import tqdm
import sys
import torch

import numpy as np
from time import time

from torch.autograd import Variable
import torchtext.vocab as vocab

from hybridvec.loader import *
from hybridvec.config import *
from hybridvec.models import *
#from hybridvec.eval import evaluate_on_all

from web.embeddings import fetch_GloVe, load_embedding
from web.datasets.utils import _get_dataset_dir
from web.evaluate import evaluate_on_all

import logging

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

#   # if not run the model on all the glove files and print the scores
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

def get_embeddings():
  config = load_config(eval=True)

  model_type = config.model_type

  TRAIN_FILE = 'data/glove/train_glove.%s.%sd.txt'%(config.vocab_source,config.vocab_dim)
  vocab_1 = vocab.GloVe(name=config.vocab_source, dim=config.vocab_dim)
  use_gpu = torch.cuda.is_available()
  print("Using GPU:", use_gpu)

  if model_type == 'baseline': 
      model = BaselineModel(vocab_1, 
                           config = config, 
                           use_cuda = use_gpu)

  elif model_type == 'seq2seq':
      encoder = EncoderRNN(config = config,
                           variable_lengths = False, 
                           embedding = None)
      decoder = DecoderRNN(config = config)
      model = Seq2seq(encoder = encoder, 
                           decoder=decoder)

  model.load_state_dict(torch.load(get_model_path(config)), strict = True)

  train_loader = get_data_loader(TRAIN_FILE,
                                 vocab_1,
                                 config.input_method,
                                 config.vocab_dim,
                                 batch_size = config.batch_size,
                                 num_workers = 0, #config.num_workers,
                                 shuffle=False,
                                 vocab_size=config.vocab_size)
  if use_gpu:
      model = model.cuda()

  model.train(False)
  out_embeddings = {}

  for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
      words, inputs, lengths, labels = data

      if use_gpu:
          inputs = inputs.cuda()

      outputs = model(inputs, lengths)
      for idx, word in enumerate(words):
        out_embeddings[word] = model.get_def_embeddings(outputs)[idx, :]

  return out_embeddings

def load_embeddings():
  a = np.load('./eval/out_embeddings.npy').item()
  return a

def main():
  embeddings = get_embeddings()
  results = evaluate_on_all(embeddings)
  out_fname = "results.csv"
  logger.info("Saving results...")
  print(results)
  results.to_csv(out_fname)


if __name__ == "__main__":
  main()
