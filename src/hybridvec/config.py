import os
import json
import argparse
from .loader import *

""" Please do not initialize a None for any field, otherwise the type info will not be available, 
"""
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
        self.load_path = 'None'
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

        #misc not configuration related but uses the same arg parsing together with config
        self.train_data_flag = True


def train_config():
    return base_config()

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
def load_config(eval = False):
    """
    Load in the right config file from desired model to evaluate
    """
    config = base_config()
    # eval specific config
    if eval == True:
        config.batch_size = 16
        config.dropout = 0
        config.packing = False
        config.input_method=INPUT_METHOD_ONE

    dict_cfg = vars(get_cfg_from_args())
    for k in dict_cfg:
        dataT = type(getattr(config,k))
        setattr(config, k, dataT(dict_cfg[k]))

    # follow the current convention
    model_path = "outputs/{}-{}-{}".format(config.model_type, config.run_name, config.vocab_dim)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # used for training, all configuration from command line
    if eval == False:
        return config

    # config exist based on model and comment, use the saved one. called by evals
    config_path = model_path + "/config.json"
    try:
        if os.path.exists(config_path):
            load_epoch = config.load_epoch
            with open(config_path) as f:
                json_cfg = dict(json.load(f))
                # use dict_cfg, because of the nasty pymonitor put something new in
                for k in dict_cfg:
                    dataT = type(getattr(config,k))
                    setattr(config, k, dataT(json_cfg[k]))

            #restore things that is from cmd line
            config.load_epoch = load_epoch
    except:
        print("no config.json found, will create one instead")
    return config

# call after init_experiment to save a copy of current config
def save_config(config):
    # config exist based on model and comment, use the saved one
    model_path = "outputs/{}-{}-{}".format(config.model_type, config.run_name, config.vocab_dim)
    
    # a bit awkward since pytorch-monitor is going to change our config!
    config_path = model_path + "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f)

# get the last saved model path
def get_model_path(config):
    model_path = "outputs/{}-{}-{}".format(config.model_type, config.run_name, config.vocab_dim)
    return model_path + "/epoch_{}/model_weights.torch".format(config.load_epoch)