python libs/OpenNMT-py/tools/embeddings_to_torch.py -emb_file data/nmt/glove/glove_$1.txt   \
                                                    -dict_file data/nmt/processed/$2.vocab.pt  \
                                                    -output_file data/nmt/embeddings/embeddings_$1 
