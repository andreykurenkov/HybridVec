class Config(object):
    title="def2vec",
    description="Translating definitions to word vectors"
    run_name='full_run_big'
    run_comment='def_concat' # gets appended to run_name as RUN_NAME-RUN_COMMENT
    log_dir='outputs/def2vec/logs'
    vocab_dim = 100
    vocab_source = '6B'
    load_path = None
    # hyperparams
    random_seed=42
    learning_rate=.0001
    max_epochs=15
    batch_size=64
    n_hidden=250
    # logging params
    print_freq=1
    write_embed_freq=100
    eval_freq = 1000
    save_path="model_weights.torch"
    embedding_log_size = 10000
    # data loading params
    num_workers = 8
    packing=True
    shuffle=True
    # model configuration [for ablation/hyperparam experiments]
    weight_init="xavier"
   # input_method=INPUT_METHOD_ALL_CONCAT
    use_bidirection=True
    use_attention=True
    cell_type='GRU'
    #use_batchnorm=True,
    hidden_size=150
    embed_size=100
    dropout=0.1
    weight_decay=0.0