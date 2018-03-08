import numpy as np
import traceback
import string
import torch
from definitions import get_a_definition
from torch.utils.data import Dataset, DataLoader


PUNC = set(string.punctuation)
def clean_str(string):
    return "".join([c for c in string.lower() if c not in PUNC])

class DefinitionsDataset(Dataset):

  def __init__(self, vocab_file, glove, shuffle, embedding_size):
    self.vocab_lines = open(vocab_file, "r").readlines()
    self.glove = glove
    self.embedding_size = embedding_size
    if shuffle:
      np.random.shuffle(self.vocab_lines)

  def __len__(self):
    return len(self.vocab_lines)

  def get_idx_info(idx):
    line = self.vocab_lines[idx]
    split_line = line.split()
    word = split_line[0]
    definition = get_a_definition(word)
    embedding = np.array([float(val) for val in split_line[1:]])
    return word, definition, embedding

  def __getitem__(self, idx):
    word, definition, embedding = self.get_idx_info(idx)
    if self.shuffle:
        while not definition:
            idx = random.randomint(len(self))
            word, definition, embedding = self.get_idx_info(idx)
    words = [clean_str(word) for word in definition.split()]
    definition = [self.glove.stoi[w] if w in self.glove.stoi else 0 for w in words]
    return (word, np.array(definition).astype(np.float32), embedding.astype(np.float32))

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
    word, src_seqs, trg_seqs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs = torch.from_numpy(np.stack(trg_seqs, axis = 0))

    return word, src_seqs, src_lengths, trg_seqs


def get_data_loader(vocab_file, vocab, embedding_size, batch_size=8, num_workers=1, shuffle=False):
  dataset = DefinitionsDataset(vocab_file, vocab, shuffle, embedding_size)
  return DataLoader(dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                    shuffle=shuffle)
