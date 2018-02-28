import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import scipy
from scipy import stats
from sklearn.metrics import precision_score, accuracy_score, recall_score, mean_squared_error
from model import Def2VecModel
from torch.autograd import Variable
import torchtext.vocab as vocab

from loader import get_data_loader, DefinitionsDataset

VOCAB_DIM = 300
VOCAB_SOURCE = '6B'
GLOVE_FILE = 'data/glove.%s.%dd.txt'%(VOCAB_SOURCE, VOCAB_DIM)

if __name__ == "__main__":
  vocab = vocab.GloVe(name=VOCAB_SOURCE, dim=VOCAB_DIM)
  use_gpu = torch.cuda.is_available()
  print("Using GPU:", use_gpu)

  model = Def2VecModel(vocab, embed_size = VOCAB_DIM, use_cuda = use_gpu)
  dataloader = get_data_loader(GLOVE_FILE, vocab)

  if use_gpu:
    model = model.cuda()
  opt = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)

  defn, embed = next(iter(dataloader))
  gt = Variable(embed)
  if use_gpu:
    defn = defn.cuda()
    gt = gt.cuda()
  pred = model(defn)
  loss = nn.MSELoss()

  print(loss(pred, gt))


