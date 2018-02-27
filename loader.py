import numpy as np
from definitions import get_wordnet_definition
import linecache
import torch
from torch.utils.data import Dataset, DataLoader

GLOVE_LEN = 400000
GLOVE_FILE = 'data/glove.6B.300d.txt'

def get_glove_pair(idx):
  line = linecache.getline(GLOVE_FILE, idx+1)
  splitLine = line.split()
  word = splitLine[0]
  embedding = np.array([float(val) for val in splitLine[1:]])
  return (word, embedding)

class DefinitionsDataset(Dataset):

  def __init__(self):
    self.idx_offset = 0

  def __len__(self):
    return GLOVE_LEN

  def __getitem__(self, idx):
    """
    Return (definition, embedding)
    """
    word,embedding = get_glove_pair(idx + self.idx_offset)
    definition = None
    definition = get_wordnet_definition(word)
    while definition is None:
      self.idx_offset += 1
      word,embedding = get_glove_pair(idx + self.idx_offset)
      definition = get_wordnet_definition(word)
    return (definition , embedding)

if __name__ == "__main__":
  dataset = DefinitionsDataset()
  dataloader = DataLoader(dataset, num_workers=4, shuffle=True)

