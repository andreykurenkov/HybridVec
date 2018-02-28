import numpy as np
import traceback
import string
import linecache
import torch
from definitions import get_wordnet_definition
from torch.utils.data import Dataset, DataLoader

PUNC = set(string.punctuation)
def clean_str(string):
    return "".join([c for c in string.lower() if c not in PUNC])

class DefinitionsDataset(Dataset):

  def __init__(self, vocab_file, vocab):
    self.vocab_file = vocab_file
    self.vocab_len = len(vocab.stoi)
    self.vocab = vocab
    self.idx_offset = 0

  def __len__(self):
    return self.vocab_len

  def __getitem__(self, idx):
    """
    Return (definition, embedding)
    """
    word,embedding = self.get_vocab_pair(idx + self.idx_offset)
    definition = None
    while definition is None:
      self.idx_offset += 1
      word,embedding = self.get_vocab_pair(idx + self.idx_offset)
      definition = get_wordnet_definition(word)
      if definition is None:
          continue
      try:
        definition = definition[list(definition.keys())[0]][0]
        words = [clean_str(word) for word in definition.split()]
        definition = []
        for i,word in enumerate(words):
            if word in self.vocab.stoi:
                definition.append(self.vocab.stoi[word])
            else:
                definition.append(0)
      except Exception as e:
        print('Error in lookup')
        traceback.print_exc()
        definition = None
    return (np.array(definition), embedding.astype(np.float32))

  def get_vocab_pair(self, idx):
    word = None
    while word is None:
        line = linecache.getline(self.vocab_file, idx + self.idx_offset + 1)
        splitLine = line.split()
        if len(line) == 0:
            self.idx_offset += 1
            continue
        try:
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
        except Exception as e:
            print(e)
            print(splitLine)
            self.idx_offset += 1
    return (word, embedding)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.
    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
    """
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = torch.from_numpy(seq[:end])
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs = np.stack(trg_seqs, axis = 0)

    return src_seqs, src_lengths, trg_seqs


def get_data_loader(vocab_file, vocab, batch_size=8, num_workers=1):
  dataset = DefinitionsDataset(vocab_file, vocab)
  return DataLoader(dataset, 
                    batch_size=batch_size, 
                    num_workers=num_workers, 
                    collate_fn=collate_fn,
                    shuffle=True)
