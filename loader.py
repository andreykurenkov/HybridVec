import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DefinitionsDataset(Dataset):

  def __init__(self):
    pass

  def __len__(self):
    pass

  def __getitem__(self, idx):
    """
    Return (definition, embedding)"""
    pass


if __name__ == "__main__":

  dataset = DefinitionsDataset()
  dataloader = DataLoader(dataset, num_workers=4, shuffle=True)

