import os
import torch
import numpy as np
from attention.visualization import visualize, loss_plot


# load weights
lang = torch.load('save/vocab.pt')
encoder = torch.load('save/encoder.pt')
decoder = torch.load('save/decoder.pt')
# load losses
loss = np.load('save/loss.npy')


# plot loss
loss_plot(loss[0], loss[1])
# attention visualization
max_output = 10
input_sentence = 'jumpman jumpman jumpman jumpman'
visualize(input_sentence, lang, encoder, decoder, max_output)