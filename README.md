Hybridvec
Repo for CS 224N project

# Environment:  
## Linux:  
Note: besides stuff in requrements.txt, should also install PyTorch (0.4) and Torchnet (https://github.com/pytorch/tnt)  
Assumes use of Python 2.7.  

## Windows: 
Install all latest tools and packages.  
Python 3.6  
PyTorch 0.5  

# Usage:
## Training 
Things in config.py are all able to be changed by command line options.  
python ./scripts/train.py --model_type baseline --run_name baseline --run_comment lstm300 --vocab_dim 300 --max_epoch 20 --print_freq=100 --cell_type LSTM  

## Eval 
A trained model type and run name together with a epoch is needed to load a trained model.  
python ./scripts/intrinsic_eval.py --model_type baseline --run_name baseline --load_epoch 20  

## Third party tools  

