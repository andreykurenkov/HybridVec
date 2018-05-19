from __future__ import print_function
import sys
import os
import collections
import traceback
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, accuracy_score, recall_score, mean_squared_error
from model import Def2VecModel, Seq2SeqModel
from torch.autograd import Variable
import torchtext.vocab as vocab
from tensorboardX import SummaryWriter
from loader import *
import torch.nn.init as init
from tqdm import tqdm
from time import time
from config import train_config
from pytorch_monitor import monitor_module, init_experiment
from seq2seq import EncoderRNN, DecoderRNN


DEBUG_LOG = False

config = train_config()

TRAIN_FILE = 'data/glove/train_glove.%s.%sd.txt'%(config.vocab_source,config.vocab_dim)
VAL_FILE = 'data/glove/val_glove.%s.%sd.txt'%(config.vocab_source, config.vocab_dim)

def weights_init(m):
    """
    Initialize according to Xavier initialization or default initialization.
    """
    if config.weight_init == 'xavier':
        if type(m) in [nn.Linear]:
            nn.init.xavier_normal(m.weight.data)
        elif type(m) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.xavier_normal(m.weight_hh_l0)
            nn.init.xavier_normal(m.weight_ih_l0)

def get_loss_nll(acc_loss, norm_term):
        if isinstance(acc_loss, int):
            return 0
        # total loss for all batches
        loss = acc_loss.data
        loss /= norm_term
        loss =  (Variable(loss).data)[0]
        #print (type(loss))
        return loss



if __name__ == "__main__":
    vocab = vocab.GloVe(name=config.vocab_source, dim=config.vocab_dim)
    use_gpu = torch.cuda.is_available()
    print("Using GPU:", use_gpu)
    #vocab_size = len(vocab.stoi)
    #reduced vocab_size
    vocab_size = 50000
    vocab_reduced = True if vocab_size < 400000 else False
    encoder = EncoderRNN(vocab_size = vocab_size,
                        vocab = vocab,
                        max_len = 100, 
                        hidden_size = config.hidden_size, 
                        embed_size = config.vocab_dim,
                        input_dropout_p=config.dropout,
                        dropout_p=config.dropout,
                        n_layers=2,
                        bidirectional=config.use_bidirection,
                        rnn_cell=config.cell_type.lower(),
                        variable_lengths=False,
                        embedding=None, #randomly initialized,
                        update_embedding=False,
                        )

    decoder = DecoderRNN(vocab_size = vocab_size,
                        max_len = 100,
                        hidden_size = config.hidden_size,
                        n_layers=2,
                        rnn_cell=config.cell_type.lower(),
                        bidirectional=config.use_bidirection,
                        input_dropout_p=config.dropout,
                        dropout_p=config.dropout,
                        use_attention=config.use_attention
                        )

    model = Seq2SeqModel(encoder = encoder,
                        decoder = decoder
                        )

    if config.load_path is None:
        model.apply(weights_init)
    else:
        model.load_state_dict(torch.load(config.load_path))
    model.apply(weights_init)

    if use_gpu:
        model = model.cuda()

    train_loader = get_data_loader(TRAIN_FILE,
                                   vocab,
                                   config.input_method,
                                   config.vocab_dim,
                                   batch_size = config.batch_size,
                                   num_workers = config.num_workers,
                                   shuffle=config.shuffle,
                                   vocab_size = vocab_size)

    val_loader = get_data_loader(VAL_FILE,
                                   vocab,
                                   config.input_method,
                                   config.vocab_dim,
                                   batch_size = config.batch_size,
                                   num_workers = config.num_workers,
                                   shuffle=config.shuffle,
                                   vocab_size = vocab_size)


    criterion = nn.NLLLoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),
                           lr=config.learning_rate,
                           weight_decay=config.weight_decay)

    writer, conf = init_experiment(config.__dict__) #pytorch-monitor needs a dict

    if DEBUG_LOG:
        monitor_module(model, writer)

    total_time = 0
    total_iter = 0

    embed_outs = None
    embed_labels = []

    for epoch in range(config.max_epochs):

        running_loss = 0.0
        start = time()
        print("Epoch", epoch)


        for i, data in enumerate(train_loader, 0):
            words, inputs, lengths, labels = data
            labels = Variable(labels)

            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            (decoder_outputs, decoder_hidden, ret_dicts), encoder_hidden  = model(inputs, lengths)
            #print (i, "out of data")
            acc_loss = 0
            norm_term = 0

            for step, step_output in enumerate(decoder_outputs):
                batch_size = inputs.shape[0]
                if step > (inputs.shape[1] -1): continue
                labeled_vals = Variable((inputs).long()[:, step])
                labeled_vals.requires_grad = False
                pred = step_output.contiguous().view(batch_size, -1)
                acc_loss += criterion(pred, labeled_vals)
                norm_term += 1


            if type(acc_loss) is int:
                raise ValueError("No loss to back propagate.")
            glove_loss = 0
            glove_loss += criterion2(Variable(encoder_hidden.data[:, :100]), labels)
            total_loss = sum ([acc_loss, glove_loss])
            batch_loss = get_loss_nll(acc_loss, norm_term)
    
            total_loss.backward()
            optimizer.step()
            running_loss += batch_loss + glove_loss.cpu().data[0]
            
            writer.add_scalar('loss', batch_loss, total_iter)
            if embed_outs is None:
                embed_outs = encoder_hidden.data.cpu()
                embed_labels = words
            else:
                embed_outs = torch.cat([embed_outs, encoder_hidden.data.cpu()])
                embed_labels += words
                num_outs = embed_outs.shape[0]
                if num_outs > config.embedding_log_size:
                    diff = num_outs - config.embedding_log_size
                    embed_outs = embed_outs[diff:]
                    embed_labels = embed_labels[diff:]
            
            del acc_loss, encoder_hidden
            if i % config.print_freq == (config.print_freq-1):
                end = time()
                diff = end-start
                total_time+=diff
                print('Epoch: %d, batch: %d, loss: %.4f , time/iter: %.2fs, total time: %.2fs' %
                             (epoch + 1, i + 1,
                              running_loss / config.print_freq,
                              diff/config.print_freq,
                              total_time))
                start = end
                running_loss = 0.0

            # if i % config.write_embed_freq == (config.write_embed_freq-1):
            #     writer.add_embedding(embed_outs,
            #                          metadata=embed_labels,
            #                          global_step=total_iter)

            if i % config.eval_freq == (config.eval_freq - 1):
                print ("happening")
                val_loss = 0.0
                for data in tqdm(val_loader, total=len(val_loader)):
                    words, inputs, lengths, labels = data
                    labels = Variable(labels)
                    if use_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    (decoder_outputs, decoder_hidden, ret_dicts), encoder_hidden = model(inputs, lengths)
                    acc_loss = 0
                    norm_term = 0

                    for step, step_output in enumerate(decoder_outputs):
                        batch_size = inputs.shape[0]
                        if step > (inputs.shape[1] -1): continue
                        labeled_vals = Variable((inputs).long()[:, step])
                        labeled_vals.requires_grad = False
                        acc_loss += criterion(step_output.contiguous().view(batch_size, -1), labeled_vals)
                        norm_term += 1


                    if type(acc_loss) is int:
                        raise ValueError("No loss to back propagate.")
                    glove_loss = 0
                    glove_loss += criterion2(Variable(encoder_hidden.data[:, :100]), labels)
                    total_loss = sum ([acc_loss, glove_loss])
                    batch_loss = get_loss_nll(acc_loss, norm_term)
     
                    val_loss += batch_loss + glove_loss.cpu().data[0]

                writer.add_scalar('val_loss', val_loss / len(val_loader), total_iter)
                del acc_loss, encoder_hidden
                print('Epoch: %d, batch: %d, val loss: %.4f' %
                             (epoch + 1, i + 1, val_loss / len(val_loader)))

            total_iter += 1

        out_dir = "outputs/def2vec/checkpoints/{}".format(config.run_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = "outputs/def2vec/checkpoints/{}/epoch_{}".format(config.run_name, epoch + 1)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        print ("saving")
        torch.save(model.state_dict(), out_path + "/" + config.save_path)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

    print('Finished Training')
