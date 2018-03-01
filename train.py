import sys
import traceback
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, accuracy_score, recall_score, mean_squared_error
from time import time
from model import Def2VecModel
from torch.autograd import Variable
import torchtext.vocab as vocab
from loader import get_data_loader, DefinitionsDataset
from tensorboardX import SummaryWriter


VOCAB_DIM = 100
VOCAB_SOURCE = '6B'
GLOVE_FILE = 'data/glove.%s.%dd.shuffled.txt'%(VOCAB_SOURCE, VOCAB_DIM)

CONFIG = dict(
    title="An Experiment",
    description="Testing out a NN",
    log_dir='logs',

    # hyperparams
    random_seed=42,
    learning_rate=.001,
    max_epochs=5,
    batch_size=32,

    # model config
    n_hidden=150,
    print_freq=10,
)

if __name__ == "__main__":
  vocab = vocab.GloVe(name=VOCAB_SOURCE, dim=VOCAB_DIM)
  use_gpu = torch.cuda.is_available()
  print("Using GPU:", use_gpu)

  model = Def2VecModel(vocab,
                       embed_size = VOCAB_DIM, 
                       output_size = VOCAB_DIM, 
                       hidden_size = CONFIG['n_hidden'],
                       use_cuda = use_gpu)
  data_loader = get_data_loader(GLOVE_FILE, 
                                vocab, 
                                batch_size = CONFIG['batch_size'],
                                num_workers = 16)

  if use_gpu:
    model = model.cuda()
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), 
                         lr=CONFIG['learning_rate'], 
                         weight_decay=0)
  
  # setup the experiment
  writer = SummaryWriter()

  total_time = 0
  total_iter = 0
  for epoch in range(CONFIG['max_epochs']):  # loop over the dataset multiple times

    running_loss = 0.0
    start = time()
    for i, data in enumerate(data_loader, 0):
      # get the inputs
      inputs, input_lengths, labels = data
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
      writer.add_scalar('loss', loss.data[0], total_iter)
      if i % CONFIG['print_freq'] == (CONFIG['print_freq']-1):    # print every 10 mini-batches
        writer.add_embedding(outputs.data, 
                           metadata=inputs, 
                           global_step=total_iter)
        end = time()
        diff = end-start
        total_time+=diff
        print('[%d, %5d] loss: %.4f , time/iter: %.2fs, total time: %.2fs' %
               (epoch + 1, i + 1, 
                running_loss / CONFIG['print_freq'], 
                diff/CONFIG['print_freq'],
                total_time))

        start = end
        running_loss = 0.0
      total_iter+=1
  writer.export_scalars_to_json("./all_scalars.json")
  writer.close()

  print('Finished Training')
