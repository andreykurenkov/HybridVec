from __future__ import print_function
import collections
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchtext.vocab as vocab
import argparse
import shutil
from tqdm import tqdm
from model import Def2VecModel
from torch.autograd import Variable
from loader import *
from tensorboardX import SummaryWriter
from pytorch_monitor import monitor_module, init_experiment

CONFIG = dict(
        title="def2vec",
        description="Translating definitions to word vectors",
        run_name='embeddings', 
        log_dir='outputs/def2vec/embedding_viz',
        random_seed=42,
        learning_rate=.0005,
        max_epochs=5,
        batch_size=16,
        n_hidden=150,
        print_freq=1,
        write_embed_freq=100,
        weight_decay=0,
        save_path="checkpoints/model_weights.torch",
        num_workers = 8,
        packing=True,
        shuffle=True,
)

def is_outlier(points, thresh=0.75):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def get_word(word):
    return vocab.vectors[vocab.stoi[word]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize glove.')
    parser.add_argument('--glove_file', type=str, required = False,
                        default='data/glove/glove.6B.100d.txt',
                        help='Source glove file.')
    parser.add_argument('--num_iter', type=int, required = False,
                        default=3000,
                        help='How many words to visualize')
    args = parser.parse_args()

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
                         use_packing = True)
    model.load_state_dict(torch.load(CONFIG['save_path']))
    data_loader = get_data_loader(args.glove_file,
                                  vocab,
                                  INPUT_METHOD_ONE,
                                  VOCAB_DIM,
                                  batch_size = CONFIG['batch_size'],
                                  num_workers = 8,
                                  shuffle=False)

    if use_gpu:
        model = model.cuda()

    writer, conf = init_experiment(CONFIG)

    embed_meta = []
    embed_vecs = []
    words_set = set()
    for i, data in tqdm(enumerate(data_loader, 0), 
                        total=args.num_iter):
        if i == args.num_iter:
            break
        words, inputs, lengths, labels = data
        labels = Variable(labels)

        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs, lengths)
        embed_labels = words
        outputs = outputs.cpu().data.numpy()

        for i,word in enumerate(words):
            word_out = outputs[i][np.newaxis,:]
            if word!='<unk>' and abs(np.sum(word_out))>0.0000000001:
                words_set.add(word)
                embed_meta.append('def2vec')#{'word':word, 
                                  #'label': 'def2vec'})
                if len(embed_vecs)==0:
                    embed_vecs = word_out
                else:
                    embed_vecs = np.concatenate([embed_vecs,word_out],axis=0)

    glove_embed_vecs = []
    with open(args.glove_file,'r') as glove_f:
        glove_lines = glove_f.readlines()[:args.num_iter*CONFIG['batch_size']]
        for i, line in tqdm(enumerate(glove_lines, 0), 
                            total=len(glove_lines)):
            split = line.split() 
            if split[0] in words_set:
                embed_meta.append('GloVe')#{#'word':split[0], 
                                  # 'label': 'GloVe'})
                glove_embed_vecs.append(np.array(split[1:]))
    embed_vecs = np.concatenate([embed_vecs,np.array(glove_embed_vecs)],axis=0)
    outliers = is_outlier(embed_vecs.astype(float))
    non_outlier_idx = np.where(outliers==False)[0]
    print('Pre filter len: %d'%len(embed_vecs))
    embed_vecs = embed_vecs[non_outlier_idx]
    print('Post filter len: %d'%len(embed_vecs))
    embed_meta = [embed_meta[i] for i in non_outlier_idx]
    writer.add_embedding(torch.FloatTensor(embed_vecs),
                         metadata=embed_meta,
                         global_step=0,
                         tag='embeddings')
