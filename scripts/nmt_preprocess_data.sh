python libs/OpenNMT-py/preprocess.py -train_src data/nmt/translation_data/$1-$2-train.$4.txt \
                                     -train_tgt data/nmt/translation_data/$1-$3-train.$4.txt \
                                     -valid_src data/nmt/translation_data/$1-$2-val.$5.txt   \
                                     -valid_tgt data/nmt/translation_data/$1-$3-val.$5.txt   \
                                     -save_data data/nmt/processed/$1
