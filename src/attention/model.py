import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from attention.preprocessing import embeds
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


# encoder batch data. added zero pad functionality, dropout, multilayer GRU
class EncoderRNN(nn.Module):
    def __init__(self, lang, hidden_size, n_layers=1, dropout=0.1, pretrained=True):
        super(EncoderRNN, self).__init__()
        
        self.lang = lang
        self.input_size = lang.n_words
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        if not pretrained:
            self.embedding = nn.Embedding(self.input_size, hidden_size)
        else:    
            self.embedding, _ = embeds(lang, hidden_size)

        # GRU layer 
        # first two parameters are emb_dim and hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        
    def forward(self, input_seqs, input_lengths, hidden=None):
        '''
        hidden=None initializes hidden unit with satisfying dimension
        input -> S x B, embedded -> S x B x N, hidden -> 2*2 x B x N
        output_raw -> S x B x 2N, hidden -> 4 x B x N (2layers x bidirection = 4)
        output -> S x B x N, which is the sum of bidirectional outputs
        output_raw concats two results
        '''
        embedded = self.embedding(input_seqs)
        packed = pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = pad_packed_sequence(outputs) # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        
        return outputs, hidden



# This is just the class calcuating the attention weights. NOT a decoder unit.
class Attn(nn.Module):
    def __init__(self, method, hidden_size, USE_CUDA):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        self.USE_CUDA = USE_CUDA
        
        if self.method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden is the output from decoder -> 1 x B x N
        # encoder_outputs -> S x B x N
        max_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        # attention weights -> B x S
        attn_energies = Variable(torch.zeros(batch_size, max_len))

        if self.USE_CUDA:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                # hidden -> 1 x B x N, encoder_outputs -> S x B x N
                # hidden[:, b].squeeze() -> N, encoder_outputs[i, b] -> N
                attn_energies[b, i] = self.score(hidden[:, b].squeeze(), encoder_outputs[i, b])

        # Normalize energies to weights in range 0 to 1
        # dim =0 along column, dim=1 along row. here it renormalizes all words within each batch
        return F.softmax(attn_energies, dim=1).unsqueeze(1) # B x S -> B x 1 x S
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':
            energy = hidden.view(-1).dot(encoder_output.view(-1))
            return energy
        
        # dot product of two vectors with length N
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy

# Luong attention
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, lang, hidden_size, n_layers=1, dropout=0.1, pretrained=True, USE_CUDA=True):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        # parameters for embedding layer
        self.lang = lang
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        # parameters for GRU layer
        self.n_layers = n_layers
        self.dropout = dropout
        self.USE_CUDA = USE_CUDA

        # Define layers
        if not pretrained:
            self.embedding = nn.Embedding(self.output_size, hidden_size)
        else:    
            self.embedding, _ = embeds(lang, hidden_size)
        # dropout layer
        self.embedding_dropout = nn.Dropout(dropout)
        # unidirectionoal GRU
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        # concat layer reduces length from 2N to N
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, self.output_size)
        
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size, USE_CUDA)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # input_seq starts with a vector of size batch_size x 1 (SOS_tokens)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq) # batch_size x 1 x N
        embedded = self.embedding_dropout(embedded)
        # as input_seq -> batch_size x 1, we need to reshape it
        embedded = embedded.view(1, batch_size, self.hidden_size) # 1 x B x N
        ########################################################################################    
        # Get current hidden state from input word and last hidden state
        # embedded -> 1 x B x N, last_hidden -> 2 x B x N
        # rnn_output -> 1 x B x N, hidden -> 2 x B x N (2layers x unidirection = 2)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        ########################################################################################
        # attention weights is the dot product between rnn_outputs and encoder_outputs
        # rnn_output -> 1 x B x N, encoder_outputs -> S x B x N
        # attn_weights -> B x 1 x S
        attn_weights = self.attn(rnn_output, encoder_outputs) # self.attn calls Attn
        # attn_weights -> B x 1 x S
        # encoder_outputs.transpose(0, 1) -> B x S x N
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x (S=1) x N
        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # rnn_output -> B x N
        context = context.squeeze(1)       # context -> B x N
        # concat rnn_output and context(weighted output from encoder)
        concat_input = torch.cat((rnn_output, context), 1) # 0:row, 1:col | concat_input -> B x 2N
        concat_output = torch.tanh(self.concat(concat_input)) # go through tanh
        ########################################################################################
        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output) # just a linear layer

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

# Bahdanau attention
class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        
        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, word_input, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        # TODO: FIX BATCHING
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        word_embedded = self.dropout(word_embedded)
        
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        context = context.transpose(0, 1) # 1 x B x N
        
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        
        # Final output layer
        output = output.squeeze(0) # B x N
        output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights







