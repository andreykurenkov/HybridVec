import sys
import traceback
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, accuracy_score, recall_score, mean_squared_error
from timeit import default_timer as timer
from model import Def2VecModel
from torch.autograd import Variable
import torchtext.vocab as vocab
from pytorch_monitor import monitor_module, init_experiment
from loader import get_data_loader, DefinitionsDataset
import requests_cache

requests_cache.install_cache('wordnet_cache')

VOCAB_DIM = 300
VOCAB_SOURCE = '6B'
GLOVE_FILE = 'data/glove.%s.%dd.txt'%(VOCAB_SOURCE, VOCAB_DIM)

# an example config dict
CONFIG = dict(
    title="An Experiment",
    description="Testing out a NN",
    log_dir='logs',
#     run_name='custom run name', # defaults to START_TIME-HOST_NAME
#     run_comment='custom run comment' # gets appended to run_name as RUN_NAME-RUN_COMMENT

    # hyperparams
    random_seed=42,
    learning_rate=.001,
    max_epochs=5,
    batch_size=1,

    # model config
    n_hidden=128,
)

if __name__ == "__main__":
  vocab = vocab.GloVe(name=VOCAB_SOURCE, dim=VOCAB_DIM)
  use_gpu = torch.cuda.is_available()
  print("Using GPU:", use_gpu)

  model = Def2VecModel(vocab,
                       embed_size = VOCAB_DIM, 
                       hidden_size = CONFIG['n_hidden'],
                       use_cuda = use_gpu)
  data_loader = get_data_loader(GLOVE_FILE, 
                                vocab, 
                                batch_size = 16,
                                num_workers = 8)

  if use_gpu:
    model = model.cuda()
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), 
                         lr=CONFIG['learning_rate'], 
                         weight_decay=0)
  
  # setup the experiment
  writer,config = init_experiment(CONFIG)
  monitor_module(model, writer)

  for epoch in range(CONFIG['max_epochs']):  # loop over the dataset multiple times

    running_loss = 0.0
    start = timer()
    for i, data in enumerate(data_loader, 0):
      # get the inputs
      inputs, input_lengths, labels = data
      print(inputs.size())
      print(len(input_lengths))
      print(labels.size())
      labels = Variable(labels)
      if use_gpu:
        inputs = inputs.cuda()
        labels = labels.cuda()

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs, input_lengths)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.data[0]
      if i % 10 == 9:    # print every 10 mini-batches
        end = timer()
        print('[%d, %5d] loss: %.4f , perf: %.4f' %
               (epoch + 1, i + 1, running_loss / 10, (end-start)/10))
        start = end
        running_loss = 0.0

  print('Finished Training')
