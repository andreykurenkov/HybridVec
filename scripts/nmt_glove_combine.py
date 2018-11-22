import numpy as np
import torch.nn as nn
import torchtext.vocab as vocab
import argparse
import shutil
from tqdm import tqdm
from hybridvec.models import Def2VecModel
from torch.autograd import Variable
from hybridvec.loader import *

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
        save_path="checkpoints/model_weights.torch"
)

def get_word(word):
    return vocab.vectors[vocab.stoi[word]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process nmt.')
    parser.add_argument('glove_provided', type=str, 
                        help='Source glove file.')
    parser.add_argument('glove_held_out', type=str, 
                        help='Source held out glove file.')
    parser.add_argument('glove_out_file', type=str, 
                        help='File to write to.')
    args = parser.parse_args()

    GLOVE_TOTAL_K = 400

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
    data_loader = get_data_loader(args.glove_held_out,
                                  vocab,
                                  INPUT_METHOD_ONE,
                                  VOCAB_DIM,
                                  batch_size = CONFIG['batch_size'],
                                  num_workers = 8,
                                  shuffle=False)

    if use_gpu:
        model = model.cuda()

    shutil.copyfile(args.glove_provided,args.glove_out_file)
    with open(args.glove_out_file,'a') as output:
        for i, data in tqdm(enumerate(data_loader, 0), total=len(data_loader)):
            words, inputs, lengths, labels = data
            labels = Variable(labels)

            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs, lengths).cpu().data.numpy()
            for i,word in enumerate(words):
                vec_str = " ".join([str(x) for x in outputs[i]])
                output.write('%s %s\n'%(words[i],vec_str))
