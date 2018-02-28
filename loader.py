import numpy as np
import string
from definitions import get_wordnet_definition
import linecache
import torch
import torchtext.vocab as vocab
from torch.utils.data import Dataset, DataLoader


GLOVE_LEN = 200
GLOVE_FILE = 'data/glove.sample.300d.txt'


class DefinitionsDataset(Dataset):

  def __init__(self, glove_file, glove_len):
    self.glove_file = glove_file
    self.glove_len = glove_len
    self.idx_offset = 0
    self.glove = vocab.GloVe(name='6B', dim=100)

  def __len__(self):
    return self.glove_len

  def __getitem__(self, idx):
    """
    Return (definition, embedding)
    """
    word,embedding = self.get_glove_pair(idx + self.idx_offset)
    definition = None
    while definition is None:
      self.idx_offset += 1
      word,embedding = self.get_glove_pair(idx + self.idx_offset)
      definition = get_wordnet_definition(word)
      try:
        definition = definition[list(definition.keys())[0]][0]
        exclude = set(string.punctuation)
        definition = [self.glove.stoi["".join([c for c in word.lower() if c not in exclude])] for word in definition.split()]
      except:
        definition = None
    return (np.array(definition), embedding.astype(np.float32))

  def get_glove_pair(self, idx):
    line = linecache.getline(self.glove_file, idx + 1)
    splitLine = line.split()
    word = splitLine[0]
    embedding = np.array([float(val) for val in splitLine[1:]])
    return (word, embedding)


def get_data_loader():
  dataset = DefinitionsDataset("data/glove.sample.300d.txt", 200)
  return DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)

