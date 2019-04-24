import os
import torch
import numpy as np
from attention.helper import del_save
from attention.iteration import trainIter
from attention.preprocessing import prep_data
from attention.model import EncoderRNN, LuongAttnDecoderRNN

USE_CUDA = True

# feel free to delete pretrained model and cleaned dataset
#######################################################################################################
# # remove embeddings
# del_save('save/vocab.pt')
# # remove cleaned dataset
# del_save('save/pairs.npy')
# # remove pretrained models
# del_save('save/encoder.pt')
# del_save('save/decoder.pt')
# # remove loss_array
# del_save('save/loss_train.npy')
# del_save('save/loss_test.npy')


#######################################################################################################
# Data preparation, hyperparameters are embedding_size, min/max_length of sequences

hidden_size = 50 # embedding_dim, modify preprocessing.embeds accordingly if you want to change this
# data preparation
data_dir = '../data/pop'
MAX_LENGTH = 80 # max length to trim
# load/instantiate lang and pairs
if not ( os.path.exists('save/vocab.pt') and os.path.exists('save/pairs.npy') ):
	lang, pairs = prep_data(data_dir, MAX_LENGTH)
	torch.save(lang, 'save/vocab.pt')
	np.save('save/pairs.npy', pairs)
else:
	lang = torch.load('save/vocab.pt')
	pairs = np.load('save/pairs.npy')
print("<====================| Dataset Summary |====================>")
print("<===| Loaded {:d} sentence pairs | {:d} unique words |===>\n".format(len(pairs), lang.n_words))

#######################################################################################################
# Instantiate encoder/decoder, hyperparameter: dropout, n_layers, attention type, optimizers(given in trainIter)

n_layers = 2 # number of gru layers
dropout=0.1 # dropout rate
attention_model = 'general' # attention type
# instantiate or load encoder/decoder
if not os.path.exists('save/encoder.pt'):
	encoder = EncoderRNN(lang, hidden_size, n_layers)
else: encoder = torch.load('save/encoder.pt')
if not os.path.exists('save/decoder.pt'):
	decoder = LuongAttnDecoderRNN(attention_model, lang, hidden_size, n_layers, dropout)
else: decoder = torch.load('save/decoder.pt')
# enable train mode
encoder.train()
decoder.train()
# Use GPU
if USE_CUDA:
	    encoder = encoder.cuda()
	    decoder = decoder.cuda()

######################################################################################################
# Iteration, corresponding hyperparameters are given below

batch_size = 100
n_epochs = 80
clip = 5.0
encoder_learning_rate = 0.001
decoder_learning_rate = 0.003
test_size = 0.3 # train/validation ratio
random_state = 666 # random_state of train/test split
# let the fun begin
print('<====================| Training Summary |====================>')
print('<===| Epochs:{:d} | BatchSize:{:d} | ClipValue:{:.1f} | EncoderLR:{:.1e} | DecoderLR:{:.1e} | ValidationSize:{:.2f} |===>\n'
	.format(n_epochs, batch_size, clip, encoder_learning_rate, decoder_learning_rate, test_size))
# train model
trainIter(n_epochs, batch_size, lang, pairs, encoder, decoder,
	clip, encoder_learning_rate, decoder_learning_rate, test_size, random_state)


