import os
import torch
import numpy as np
from attention.visualization import visualize, loss_plot


# load weights
lang = torch.load('save/eminem/vocab.pt')
encoder = torch.load('save/eminem/encoder.pt')
decoder = torch.load('save/eminem/decoder.pt')
# load losses
loss = np.load('save/eminem/loss.npy')


# plot losses
# loss_plot(loss[0], loss[1])
# attention visualization
max_output = 20
color = 'Reds'

input_sentence = 'look i was gonna go easy'
save_path = '../images/eminem1.jpg'
visualize(input_sentence, lang, encoder, decoder, max_output, save_path, color)

input_sentence = 'i feel like a rap god'
save_path = '../images/eminem2.jpg'
visualize(input_sentence, lang, encoder, decoder, max_output, save_path, color)

input_sentence = 'where is slim shady'
save_path = '../images/eminem3.jpg'
visualize(input_sentence, lang, encoder, decoder, max_output, save_path, color)

input_sentence = 'dear slim'
save_path = '../images/eminem4.jpg'
visualize(input_sentence, lang, encoder, decoder, max_output, save_path, color)