import torch
from torch.autograd import Variable
from attention.preprocessing import indexes_from_sentence
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.style.use('ggplot') # plot style
mpl.rcParams["font.size"] = 18 # font size
# plt.switch_backend('agg')



def evaluate(input_sentence, lang, encoder, decoder, max_length, USE_CUDA=True):
    '''
    this function does greedy search
    '''
    SOS_token = 1
    EOS_token = 2
    
    # diable dropout and batchnorm
    encoder.eval()
    decoder.eval()
    # disable parameter update
    with torch.no_grad():
        # format input_sentence before feeding to the model
        input_sentence = [indexes_from_sentence(lang, input_sentence, EOS_token)]
        input_batches = Variable(torch.LongTensor(input_sentence)).transpose(0, 1)
        input_lengths = [len(input_batches)]
        
        if USE_CUDA:
            input_batches = input_batches.cuda()
        
        # Run through encoder
        encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

        # Create starting vectors for decoder
        decoder_input = Variable(torch.LongTensor([SOS_token])) # SOS
        decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
        
        if USE_CUDA:
            decoder_input = decoder_input.cuda()

        # Store output words and attention states
        decoded_words = []
        # row
        decoder_attentions = torch.zeros(max_length+1, max_length+1)
        
        # Run through decoder
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze().cpu()
            # Choose top word from output
            topv, topi = decoder_output.topk(1)
            ix = topi.item()
            if ix == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(lang.index2word[ix])
                
            # Next input is chosen word
            decoder_input = Variable(torch.LongTensor([ix]))
            if USE_CUDA: decoder_input = decoder_input.cuda()

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)
    
    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]

def plot_attention(input_sentence, output_words, attentions, save_path, color):
    plt.rcParams["axes.grid"] = False # disable grid of plots
    # Set up figure with colorbar
    fig, ax = plt.subplots(figsize=(10,8))
    att = ax.imshow(attentions.numpy(), cmap=color, aspect='auto')
    cbar = fig.colorbar(att)

    # add title
    ax.set_title('Attention Visualization')
    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split() + ['<EOS>'], rotation=30)
    ax.set_yticklabels([''] + output_words)
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig(save_path) # should try to plot attention for each epoch
    plt.show()

#####################################################################################
# visualize attention with give seed text

def visualize(input_sentence, lang, encoder, decoder, max_length, save_path, color, target_sentence=None, USE_CUDA=True):
    # function does both evaluate and plot
    output_words, attentions = evaluate(input_sentence, lang, encoder, decoder, max_length, USE_CUDA=True)
    output_sentence = ' '.join(output_words)
    print('<=== Seed_text: {} ===>'.format(input_sentence))
    if target_sentence is not None:
        print('<=== OG_lyrics: {} ===>'.format(target_sentence))
    print('<=== Generated_lyrics: {} ===>'.format(output_sentence))

    plot_attention(input_sentence, output_words, attentions, save_path, color)

#####################################################################################
# plot losses from train and test

def loss_plot(loss_train, loss_test):
    plt.rcParams["axes.grid"] = True
    fig, ax = plt.subplots(figsize=(10,6))
    n = len(loss_train)
    xtcks = np.arange(1, n+1)
    ax.set_title('Learning Curve')
    ax.set_xlabel('Training epochs')
    ax.set_ylabel('Loss')
    ax.plot(xtcks, loss_train, label='Train')
    ax.plot(xtcks, loss_test, label='Test')
    ax.legend()
    plt.savefig('../images/learning_curve.jpg')
    plt.show()








