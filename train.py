import numpy as np
import torch
import torch.optim as optim
import scipy
from scipy import stats
from sklearn.metrics import precision_score, accuracy_score, recall_score, mean_squared_error
from model import Def2VecModel

N_LAYERS = 2
N_HIDDEN = 128
LR = 0.0001

use_gpu = torch.cuda.is_available()
print("Using GPU:", use_gpu)


if __name__ == "__main__":

  model = Def2VecModel(5, 10, N_HIDDEN, N_LAYERS)
  if use_gpu:
    model = model.cuda()
  opt = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)

  for epoch in range(1000):
    pass