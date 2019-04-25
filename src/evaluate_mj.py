import os
import torch
import numpy as np
from attention.visualization import visualize, loss_plot


# load weights
lang = torch.load('save/drake/vocab.pt')
encoder = torch.load('save/drake/encoder.pt')
decoder = torch.load('save/drake/decoder.pt')
# load losses
loss = np.load('save/drake/loss.npy')


# plot loss
# loss_plot(loss[0], loss[1])
# attention visualization
max_output = 10
input_sentence = 'me and my friends we got money to spend'
visualize(input_sentence, lang, encoder, decoder, max_output)