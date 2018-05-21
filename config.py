from loader import *


class base_config(object):
    def __init__ (self):
        self.title="seq2seq"
        self.description="Creating new word embeddings using seq2seq"
        self.run_name='check-add-embedding'
        self.run_comment='base' # gets appended to run_name as RUN_NAME-RUN_COMMENT
        self.log_dir='outputs/{}/logs'.format(self.title)
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
        self.use_attention=False
        self.cell_type='GRU'
        #use_batchnorm=True,
        self.hidden_size=150
        self.embed_size=100
        self.dropout=0.3
        self.weight_decay=0.0
        self.use_glove = True
        self.glove_weight = 1
        self.glove_loss = True

def train_config():
    return base_config()

#creates a config based on a dictionary config loaded in from the model being evaluated
def eval_config(d, run_name, run_comment, epoch, verbose):
    e = base_config()
    #update base
    for k in d:
        setattr(e, k, d[k])

    e.run_name=run_name, 
    e.run_comment=run_comment,
    e.log_dir='logs'
    e.batch_size = 16
    name = run_name + '-' + run_comment
    e.save_path="outputs/{}/checkpoints/{}/epoch_{}/model_weights.torch".format(e.title, name, epoch)
    e.packing = False
    e.input_method=INPUT_METHOD_ONE

    if verbose:
            print ("Evaluation model will be loaded from {}".format(e.save_path))

    return e
