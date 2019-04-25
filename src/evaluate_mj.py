import os
import torch
import numpy as np
from attention.visualization import visualize, loss_plot


# load weights
lang = torch.load('save/mj/vocab.pt')
encoder = torch.load('save/mj/encoder.pt')
decoder = torch.load('save/mj/decoder.pt')
# load losses
loss = np.load('save/mj/loss.npy')


# plot loss
# loss_plot(loss[0], loss[1])
# attention visualization
max_output = 20
color = 'Purples'

input_sentence = 'jumpman jumpman jumpman'
save_path = '../images/mj1.jpg'
visualize(input_sentence, lang, encoder, decoder, max_output, save_path, color)

input_sentence = 'i guess you lose some and win some'
save_path = '../images/mj2.jpg'
visualize(input_sentence, lang, encoder, decoder, max_output, save_path, color)

input_sentence = 'people drain me energy'
save_path = '../images/mj3.jpg'
visualize(input_sentence, lang, encoder, decoder, max_output, save_path, color)

input_sentence = 'people drain me energy'
save_path = '../images/mj4.jpg'
visualize(input_sentence, lang, encoder, decoder, max_output, save_path, color)