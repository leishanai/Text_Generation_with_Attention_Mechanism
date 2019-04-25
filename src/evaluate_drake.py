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
max_output = 20
color = 'Greens'

input_sentence = 'jumpman jumpman jumpman'
save_path = '../images/drake1.jpg'
visualize(input_sentence, lang, encoder, decoder, max_output, save_path, color)

input_sentence = 'i guess you lose some and win some'
save_path = '../images/drake2.jpg'
visualize(input_sentence, lang, encoder, decoder, max_output, save_path, color)

input_sentence = 'people drain me energy'
save_path = '../images/drake3.jpg'
visualize(input_sentence, lang, encoder, decoder, max_output, save_path, color)

input_sentence = 'them boys up to something'
save_path = '../images/drake4.jpg'
visualize(input_sentence, lang, encoder, decoder, max_output, save_path, color)
