import numpy as np
import traceback
import random
import torch
import time
from tqdm import tqdm
from definitions import *
from torch.utils.data import Dataset, DataLoader

INPUT_METHOD_ALL_CONCAT = "concat_defs"
INPUT_METHOD_ONE = "random_def"
INPUT_METHOD_ONE_CORRUPT = "random_def_concat"
INPUT_METHOD_RANDOM = "random_words"

class DefinitionsDataset(Dataset):

    def __init__(self, vocab_file, glove, input_method, shuffle, embedding_size):
        with open(vocab_file, "r") as f:
            self.vocab_lines = f.readlines()
        self.glove = glove
        self.embedding_size = embedding_size
        self.input_method = input_method
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.vocab_lines)

    def __len__(self):
        return len(self.vocab_lines)

    def get_idx_info(self, idx):
        line = self.vocab_lines[idx]
        split_line = line.split()
        word = split_line[0]
        if self.input_method == INPUT_METHOD_ONE:
		definition = get_a_definition(word)
        elif self.input_method == INPUT_METHOD_ALL_CONCAT:
		definition = get_definitions_concat(word)
        embedding = np.array([float(val) for val in split_line[1:]])
        return word, definition, embedding

    def __getitem__(self, idx):
        word, definition, embedding = self.get_idx_info(idx)
        if self.shuffle:
            while definition is None:
                idx = random.randint(0,len(self.vocab_lines)-1)
                word, definition, embedding = self.get_idx_info(idx)
        words = [clean_str(w) for w in definition.split()]
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
    data.sort(key=lambda x: len(x[1]), reverse=True)

    # seperate source and target sequences
    word, src_seqs, trg_seqs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs = torch.from_numpy(np.stack(trg_seqs, axis = 0))

    return word, src_seqs, src_lengths, trg_seqs


def get_data_loader(vocab_file, vocab, input_method, embedding_size, batch_size=8, num_workers=1, shuffle=False):
    dataset = DefinitionsDataset(vocab_file, vocab, input_method, shuffle, embedding_size)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      collate_fn=collate_fn,
                      shuffle=shuffle)

def get_on_the_fly_input(definition, glove):
    words = [clean_str(word) for word in definition.split()]
    definition = [glove.stoi[w] if w in glove.stoi else 0 for w in words]
    return np.array(definition).astype(np.float32)

def fill_cache(vocab_file, 
               source='wordnik', 
               filter_source=None, 
               sleep_time=0,
               print_defs=True):
    """
    A function to call a particular definition source for all words in a given
    source dataset. 

    Args:
        vocab_file: The GloVe formatted file to iterate over words for
        source: The definition source to cache
                Can be wordnik, glosbe, or wiki
        filter_source: A definition source to check against before 
                     querying source. Eg if one is faster to 
                     access that another, this will limit requests
                     to specified source to just those not defined
                     with the filter_with source.
        sleep_time: amount of time to sleep between calls, 
                    in order not to trigger rate limits
        print_defs: whether to print defs
    """
    with open(vocab_file, "r") as f:
        lines = f.readlines()
    pbar = tqdm(lines)
    no_def_count = 0
    filtered_count = 0
    no_def_words = []
    for line in pbar:
        word = line.split()[0]
        no_def_words.append(word)
        pbar.set_description("Processing %s" % word)
        if filter_source == 'wordnik':
            defs = get_wordnik_definitions(word)
        if filter_source is not None:
            if defs:
                filtered_count+=1
                continue
        defs = None
        while defs is None:
            if source == 'wordnik':
                defs = get_wordnik_definitions(word)
            elif source == 'glosbe':
                defs = get_glosbe_definitions(word)
            elif source == 'wiki':
                defs = get_wiki_summary(word)
            if defs is None:
                #hit rate limit, sleep it off
                time.sleep(100*sleep_time)
        if print_defs:
            print(defs)
        if not defs:
            no_def_count+=1
        if sleep_time:
            time.sleep(sleep_time)
    print(no_def_words)
    print('Total words without filtered %d'%filtered_count)
    print('Total words without def %d'%no_def_count)
