from __future__ import print_function
import tqdm
import collections
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchtext.vocab as vocab
import argparse
import shutil
from model import Def2VecModel
from torch.autograd import Variable
from loader import *

CONFIG = dict(
        title="def2vec",
        description="Translating definitions to word vectors",
        log_dir='logs',
        random_seed=42,
        learning_rate=.0005,
        max_epochs=5,
        batch_size=16,
        n_hidden=150,
        print_freq=1,
        write_embed_freq=100,
        weight_decay=0,
        save_path="./data/checkpoints/full_run_big_batch-def_concatdef_concat/epoch_5/model_weights.torch"
)

def get_word(word):
    return vocab.vectors[vocab.stoi[word]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process nmt.')
    parser.add_argument('held_out', type=int, 
                        help='How many k are held out.')
    args = parser.parse_args()

    provided_file = 'data/nmt/glove/glove_%dk_provided.txt'%(args.held_out)
    held_out_file = 'data/nmt/glove/glove_%dk_held_out.txt'%(args.held_out)
    output_file = 'data/nmt/glove/glove_%dk_combined.txt'%(args.held_out)

    VOCAB_DIM = 100
    VOCAB_SOURCE = '6B'
    vocab = vocab.GloVe(name=VOCAB_SOURCE, dim=VOCAB_DIM)
    use_gpu = torch.cuda.is_available()
    print("Using GPU:", use_gpu)

    model = Def2VecModel(vocab,
                         embed_size = VOCAB_DIM,
                         output_size = VOCAB_DIM,
                         hidden_size = CONFIG['n_hidden'],
                         use_cuda = use_gpu,
                         use_packing = False)
    model.load_state_dict(torch.load(CONFIG['save_path']))
    data_loader = get_data_loader(held_out_file,
                                  vocab,
                                  INPUT_METHOD_ALL_CONCAT,
                                  VOCAB_DIM,
                                  batch_size = CONFIG['batch_size'],
                                  num_workers = 8,
                                  shuffle=True)

    if use_gpu:
        model = model.cuda()

    shutil.copyfile(provided_file,output_file)
    with open(output_file,'a') as output:
        for i, data in tqdm.tqdm(enumerate(data_loader, 0), total=len(data_loader)):
            words, inputs, lengths, labels = data
            labels = Variable(labels)

            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            for i,word in enumerate(words):
                vec_str = " ".join([str(x) for x in outputs[i]])
                output.write('%s %s\n'%(words[i],vec_str))
