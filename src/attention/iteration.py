import time
import random
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
    learning_rate = 0.0001, decoder_learning_ratio = 5.0):
    
    # setup start time
    start = time.time()
    # lists to save losses
    loss_train_array = []
    loss_test_array = []
    # train/test split
    pairs_train, pairs_test = train_test_split(pairs, test_size=0.2, random_state=666)
    # initialize optimizer
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    # train loop starts from here
    epoch = 0
    while epoch < n_epochs:
        if epoch == 0: print('<==================== Let the fun begin ====================>')
        if epoch%10 ==0: print('<### {} epochs to run ###>'.format(n_epochs - epoch))
        epoch += 1
        # run len(pairs)//batch_size times      
        for itr in range(len(pairs)//batch_size):
            print('<=== Training on batch {}/{}... ===>'.format(itr+1, len(pairs)//batch_size))
            # Get training data for this cycle
            input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size, lang, pairs_train)
            # Run the train function
            loss_train = train(input_batches, input_lengths, target_batches, target_lengths,
                encoder, decoder, encoder_optimizer, decoder_optimizer, clip)
        
        #####################################################################################################
        # validation
        print('<=== Testing on validation... ===>')
        valid_size = len(pairs_test)
        input_batches, input_lengths, target_batches, target_lengths = random_batch(valid_size, lang, pairs_test)
        # get loss from validation
        loss_test = validation(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder)
        
        #####################################################################################################
        # append metric to array for each epoch
        loss_train_array.append(loss_train)
        loss_test_array.append(loss_test)
        # print out loss for each epoch
        print_summary = "<=== %d%% trained ===> loss_train: %.3f loss_test: %.3f <=== %s ===>" % (epoch / n_epochs * 100, loss_train, loss_test, timer(start, epoch / n_epochs))
        print(print_summary)

    return loss_train_array, loss_test_array
