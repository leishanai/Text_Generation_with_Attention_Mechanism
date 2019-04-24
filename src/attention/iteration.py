import os
import time
import random
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from attention.helper import timer
from attention.masked_cross_entropy import *
from attention.preprocessing import prep_data, random_batch



def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, 
          encoder_optimizer, decoder_optimizer, clip, USE_CUDA=True):
    
    SOS_token = 1
    EOS_token = 2
    batch_size = input_batches.size(1)
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
    
    # max_length of batch data
    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # reinforcement learning implementation
    teacher_forcing_ratio = 0.5
    # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    use_teacher_forcing = True
    if use_teacher_forcing:    
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            all_decoder_outputs[t] = decoder_output
            # teacher forcing
            decoder_input = target_batches[t] # Next input is current target

        # Loss for batch data
        loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
        target_batches.transpose(0, 1).contiguous(), # -> batch x seq
        target_lengths)

    else:
        # Without teacher forcing: use its own predictions as the next input
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            all_decoder_outputs[t] = decoder_output
            # obtain the prediction as new decoder_input
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            # break if EOS_token found
            if decoder_input.item() == EOS_token:
                break
        # Loss for batch data
        loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
        target_batches.transpose(0, 1).contiguous(), # -> batch x seq
        target_lengths)
    
    # accumulate gradients
    loss.backward()
    
    # Clip gradient norms to prevent exploding gradient issue
    # clip_grad_value_ clip grad between (-clip, +clip)
    # clip_grad_norm_ clip grad to a single value clip
    ec = torch.nn.utils.clip_grad_value_(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_value_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item()

def validation(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, USE_CUDA=True):
    
    SOS_token = 1
    loss = 0
    # take the whole validation set as a batch, as there is no need to update parameters
    batch_size = input_batches.size(1)

    # diable dropout and batchnorm
    encoder.eval()
    decoder.eval()
    # disable parameter update
    with torch.no_grad():

        # Run words through encoder
        encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder

        # max_length of batch data
        max_target_length = max(target_lengths)
        all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

        # Move new Variables to CUDA
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            all_decoder_outputs = all_decoder_outputs.cuda()

        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            all_decoder_outputs[t] = decoder_output
            decoder_input = target_batches[t] # Next input is current target

        # Loss for batch data
        loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),
        target_batches.transpose(0, 1).contiguous(),
        target_lengths)
    # Set back to training mode
    encoder.train(True)
    decoder.train(True)
    
    return loss.item()


def trainIter(n_epochs, batch_size, lang, pairs, encoder, decoder, clip, 
    encoder_learning_rate, decoder_learning_rate, test_size, random_state=666):
    # batch_size = 1 # stochastic is reproduce
    # setup start time
    start = time.time()
    # train/test split
    pairs_train, pairs_test = train_test_split(pairs, test_size = test_size, random_state = random_state)
    # initialize optimizer
    encoder_optimizer = optim.Adam(encoder.parameters(), lr = encoder_learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr = decoder_learning_rate)
    # load loss_array or create one if does not exist
    if os.path.exists('save/loss.npy'):
        loss_array = np.load('save/loss.npy')
    else:
        loss_array = np.array([[], []]) # initialize an empty 2d array
    # train loop starts from here
    for epoch in range(1, n_epochs+1):
        # find the number of loops to run in one epoch   
        # n_iters = 1 # this is for fast debugging purpose
        n_iters = len(pairs)//batch_size
        # iteration of batches starts      
        for itr in range(1, n_iters+1):
            print('<===| Training on epoch:{:d}/{:d} batch:{:d}/{:d}... | {:s} |===>'
                .format(epoch, n_epochs, itr, n_iters, timer(start, itr / n_iters)))
            # Get training data for this cycle
            input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size, lang, pairs_train)
            # Run the train function
            loss_train = train(input_batches, input_lengths, target_batches, target_lengths,
                encoder, decoder, encoder_optimizer, decoder_optimizer, clip)
            # save model every batch
            torch.save(encoder, 'save/encoder.pt')
            torch.save(decoder, 'save/decoder.pt')
        #####################################################################################################
        # validation
        valid_size = len(pairs_test)
        print('<===| Testing on validation set with {} sentence pairs, please be patient! |===>\n'.format(valid_size))
        input_batches, input_lengths, target_batches, target_lengths = random_batch(valid_size, lang, pairs_test)
        # get loss from validation
        loss_test = validation(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder)
        #####################################################################################################
        # print out loss for each epoch
        print("<===| {:.1%} assignment done | loss_train:{:.1f} | loss_test:{:.1f} | {:s} |===>\n"
        .format(epoch / n_epochs, loss_train, loss_test, timer(start, epoch / n_epochs)))
        # one epoch done!
        epoch += 1
        ######################################################################################################
        # write losses into file for every epoch
        loss_new = np.array([[loss_train, loss_test]])
        np.concatenate((loss_array, loss_new.T), axis=1)
        np.save('save/loss.npy', loss_array)