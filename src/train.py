import os
import torch
import numpy as np
from attention.iteration import trainIter
from attention.preprocessing import prep_data
from attention.model import EncoderRNN, LuongAttnDecoderRNN

USE_CUDA = True

#######################################################################################################
# # remove embeddings
# os.remove('save/vocab.pt') 
# os.remove('save/pairs.npy')
# # remove weights
# os.remove('save/encoder.pt')
# os.remove('save/decoder.pt')
# remove loss_array
# os.remove('save/loss_train.npy')
# os.remove('save/loss_test.npy')

#######################################################################################################
# Training purpose

hidden_size = 50 # embedding_dim, modify preprocessing.embeds accordingly if you want to change this
# data preparation
data_dir = '../data/pop'
MAX_LENGTH = 20 # max length to trim
# load/instantiate lang and pairs
if not ( os.path.exists('save/vocab.pt') and os.path.exists('save/pairs.npy') ):
	lang, pairs = prep_data(data_dir, MAX_LENGTH)
	torch.save(lang, 'save/vocab.pt')
	np.save('save/pairs.npy', pairs)
else:
	lang = torch.load('save/vocab.pt')
	pairs = np.load('save/pairs.npy')

# prepare for encoder/decoder
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

# Training parameter tuning
batch_size = 50
n_epochs = 1
clip = 10.0
learning_rate = 0.0001
decoder_learning_ratio = 3.0
# let the fun begin
loss_train_array, loss_test_array = trainIter(n_epochs, batch_size, lang, pairs, encoder, decoder, learning_rate, decoder_learning_ratio)
# save model
torch.save(encoder, 'save/encoder.pt')
torch.save(decoder, 'save/decoder.pt')
# concatenate losses
if os.path.exists('save/loss_train.npy'):
	loss_train_total =  np.load('save/loss_train.npy').tolist() + loss_train_array
if os.path.exists('save/loss_test.npy'):
	loss_test_total =  np.load('save/loss_test.npy').tolist() + loss_test_array
# save losses
np.save('save/loss_train.npy', loss_train_total)
np.save('save/loss_test.npy', loss_test_total)

