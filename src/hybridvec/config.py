import os
import json
import argparse
from .loader import *


class base_config(object):
    def __init__ (self):
        self.title="def2vec"
        self.description="Creating new word embeddings using seq2seq"
        self.model_type='seq2seq'
        self.run_name='seq2seq'
        self.run_comment='test-aneesh' # gets appended to run_name as RUN_NAME-RUN_COMMENT
        self.log_dir='outputs/def2vec/logs'
        self.vocab_dim = 100
        self.vocab_source = '6B'
        self.vocab_size = 50000
        self.load_path = None
        self.load_epoch = 0
        # hyperparams
        self.random_seed=42
        self.learning_rate=.0001
        self.max_epochs=15
        self.batch_size=64
        self.n_hidden=250
        # logging params
        self.print_freq=1
        self.write_embed_freq= 100
        self.eval_freq = 1000
        self.save_path="model_weights.torch"
        self.embedding_log_size = 10000

        # data loading params
        self.num_workers = 8
        self.packing=True
        self.shuffle=True

        # model configuration [for ablation/hyperparam experiments]
        self.weight_init="xavier"
        self.input_method=INPUT_METHOD_ALL_CONCAT
        self.use_bidirection=False
        self.use_attention=True
        self.cell_type='GRU'

        #use_batchnorm=True,
        self.hidden_size=150
        self.embed_size=100
        self.dropout=0.1
        self.weight_decay=0.0
        self.use_glove_init = True
        self.glove_aux_loss = True
        self.num_layers = 2
        self.max_len = 100
        self.reg_weight = 0.01
        self.glove_aux_weight = 0.01


def train_config():
    return base_config()

#creates a config based on a dictionary config loaded in from the model being evaluated
def eval_config(d, run_name, run_comment, epoch, verbose):
    e = base_config()

    #update base
    for k in d:
        setattr(e, k, d[k])

    e.run_name=run_name 
    e.run_comment=run_comment
    e.log_dir='logs'
    e.batch_size = 16
    e.dropout = 0
    name = run_name + '-' + run_comment #+ "-" + run_comment
    e.save_path = "outputs/def2vec/checkpoints/{}/epoch_{}/model_weights.torch".format(name, epoch)
    e.packing = False
    e.input_method=INPUT_METHOD_ONE
    if verbose:
            print ("Evaluation model will be loaded from {}".format(e.save_path))

    return e


def get_cfg_from_args():
    """
    Gets configuration from command line
    """
    cfg = base_config().__dict__
    parser = argparse.ArgumentParser(description="Hybrydvec configurations")
    for key in cfg.keys():
        parser.add_argument("--{}".format(key), default=cfg[key])
    args = parser.parse_args()
    return args

# use the saved config first, if not exist, generate from command line.
def load_config():
    """
    Load in the right config file from desired model to evaluate
    """
    config = base_config()
    dict_cfg = vars(get_cfg_from_args())
    for k in dict_cfg:
        setattr(config, k, dict_cfg[k])

    # follow the current convention
    model_path = "outputs/{}".format(config.title)
    log_path = model_path + '/logs'

    # config exist based on model and comment, use the saved one
    config_path = log_path + "/{}-{}/config.json".format(config.run_name, config.run_comment)
    try:
        if os.path.exists(config_path):
            with open(config_path) as f:
                dict_cfg = dict(json.load(f))
                for k in dict_cfg:
                    setattr(config, k, dict_cfg[k])
    except:
        print("no config.json found, will create one instead")
    return config

# call after init_experiment to save a copy of current config
def save_config(config):
    # config exist based on model and comment, use the saved one
    model_path = "outputs/{}".format(config.title)
    log_path = model_path + '/logs'
    # a bit awkward since pytorch-monitor is going to change our config!
    config_path = log_path + "/{}/config.json".format(config.run_name)
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f)


# get current log
def get_log_path(config):
    return "outputs/{}/logs".format(config.run_name)

#get checkpoint path
def get_checkpoint_path(config):
    return "outputs/{}/checkpoints".format(config.run_name)

# get the last saved model path
def get_model_path(config):
    return "outputs/{}/checkpoints/{}-{}/epoch_{}/model_weights.torch".format(
        config.title, config.run_name, config.run_comment, config.load_epoch)