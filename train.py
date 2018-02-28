import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import scipy
from scipy import stats
from sklearn.metrics import precision_score, accuracy_score, recall_score, mean_squared_error
from model import Def2VecModel
from torch.autograd import Variable

from loader import get_data_loader, DefinitionsDataset


if __name__ == "__main__":

  use_gpu = torch.cuda.is_available()
  print("Using GPU:", use_gpu)

  model = Def2VecModel(vocab_size=100000, output_size=300, hidden_size=128, num_layers=2)
  dataloader = get_data_loader()

  if use_gpu:
    model = model.cuda()
  opt = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)

  defn, embed = next(iter(dataloader))
  pred = model(defn)
  loss = nn.MSELoss()

  print(loss(pred, Variable(embed)))


