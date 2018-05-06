from loader import *


class base_config(object):
    def __init__ (self):
        self.title="def2vec"
        self.description="Translating definitions to word vectors"
        self.run_name='full_run_big'
        self.run_comment='def_concat' # gets appended to run_name as RUN_NAME-RUN_COMMENT
        self.log_dir='outputs/def2vec/logs'
        self.vocab_dim = 100
        self.vocab_source = '6B'
        self.load_path = None
        # hyperparams
        self.random_seed=42
        self.learning_rate=.0001
        self.max_epochs=15
        self.batch_size=64
        self.n_hidden=250
        # logging params
        self.print_freq=1
        self.write_embed_freq=100
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
        self.use_bidirection=True
        self.use_attention=True
        self.cell_type='GRU'
        #use_batchnorm=True,
        self.hidden_size=150
        self.embed_size=100
        self.dropout=0.1
        self.weight_decay=0.0


def train_config():
    return base_config()



def eval_config():
    e = base_config()
    e.run_name='ablate_attn', # defaults to START_TIME-HOST_NAME
    e.run_comment='vanilla',
    e.log_dir='logs'
    e.batch_size = 16
    e.save_path="./checkpoints/full_run-vanillavanilla/epoch_5/model_weights.torch"
    e.packing = False
    e.input_method=INPUT_METHOD_ONE
    return e