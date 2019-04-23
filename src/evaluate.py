import os
import torch
import numpy as np
from attention.visualization import visualize, loss_plot


# load weights
lang = torch.load('save/vocab.pt')
encoder = torch.load('save/encoder.pt')
decoder = torch.load('save/decoder.pt')
# load losses
loss_train = np.load('save/loss_train.npy')
loss_test = np.load('save/loss_test.npy')

# plot loss
loss_plot(loss_train, loss_test)
# attention visualization
max_output = 10
input_sentence = 'jumpman jumpman jumpman jumpman'
visualize(input_sentence, lang, encoder, decoder, max_output)