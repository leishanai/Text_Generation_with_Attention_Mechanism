import os
import re
import string
import random
import numpy as np
import unicodedata
from io import open
import torch
import torch.nn as nn
from torch.autograd import Variable



class vocab:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'PAD', 1: 'SOS', 2: 'EOS'}
        self.n_words = 3  # Count PAD, SOS and EOS
        self.max_length = 1

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
    def addSentence(self, sentence):
        i = 0
        for word in sentence.split():
            self.addWord(word)
            i+=1
        if i > self.max_length:
            self.max_length = i    

# remove punctuations
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z]+", r" ", s)
    return s

def read_data(data_path):        
    '''
    Return cleaned a list of pairs, e.g. [[pair1, pair2], [pair3, pair4], ...]
    '''
    # Read the file and split into lines
    lines = open(data_path, encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = []
    for i in range(0,len(lines)-1): # be careful of step=2
        pairs.append([normalizeString(lines[i]), normalizeString(lines[i+1])])
    return pairs

# trim the pairs by max_length
def filterPair(p, MAX_LENGTH):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs, MAX_LENGTH):
    return [pair for pair in pairs if filterPair(pair, MAX_LENGTH)]

#######################################################################################
# data cleaning in one function

def prep_data(data_dir, MAX_LENGTH):
    '''
    Return vocab instance, and trimed and cleaned pairs
    '''
    # obtain all data_paths from data_dir    
    data_list = []
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            data_list.append(os.path.join(subdir, file))
    pairs_total = []
    # instantial vocab class
    lang = vocab()
    print('<====================| Data Cleaning |====================>')
    # iterate through all files
    for itr, data_path in enumerate(data_list):
        # get pairs
        pairs_raw = read_data(data_path)
        # filter by max_length
        pairs_trimed = filterPairs(pairs_raw, MAX_LENGTH)     
        # concatenate pairs
        pairs_total += pairs_trimed
        # initialize vocab instance
        for pair in pairs_trimed:
            lang.addSentence(pair[0])
            lang.addSentence(pair[1])
        print("<===| Read {:d} sentence pairs from Doc {:d} | Trimmed to {:d} sentence pairs | Counted {:d} unique words in total|===>\n"
            .format(len(pairs_raw), itr+1, len(pairs_trimed),lang.n_words))

    return lang, pairs_total



###############################################################################
# string to index, datatype -> tensor
# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence, EOS_token):
    return [lang.word2index[word] for word in sentence.split()] + [EOS_token]

# Pad a with the PAD symbol
def pad_seq(seq, max_length, PAD_token):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq



###############################################################################
# batch data
def random_batch(batch_size, lang, pairs, USE_CUDA=True):
	'''
	input -> batch_size
	return -> input/output of a batch, array of lengths of input/output
	batched data -> max_len x batch_size (note that this is required size feeding to gru layer)
	'''
	PAD_token = 0
	EOS_token = 2

	input_seqs = []
	target_seqs = []

	# integer encoding randomly picked samples of a batch
	for i in range(batch_size):
	    pair = random.choice(pairs)
	    input_seqs.append(indexes_from_sentence(lang, pair[0], EOS_token))
	    target_seqs.append(indexes_from_sentence(lang, pair[1], EOS_token))

	# rearrange seqs in a descending order
	seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
	input_seqs, target_seqs = zip(*seq_pairs)

	# For input and target sequences, get array of lengths and pad with 0s to max length
	input_lengths = [len(s) for s in input_seqs] # lengths of each seq in a batch
	input_padded = [pad_seq(s, max(input_lengths), PAD_token) for s in input_seqs]
	target_lengths = [len(s) for s in target_seqs]
	target_padded = [pad_seq(s, max(target_lengths), PAD_token) for s in target_seqs]

	# Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
	input_batch = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
	target_batch = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

	if USE_CUDA:
	    input_batch = input_batch.cuda()
	    target_batch = target_batch.cuda()
	    
	return input_batch, input_lengths, target_batch, target_lengths


###########################################################################################
# pretrained word embedding
def pretrained(glove_path):
    '''
    load pretrained glove and return it as a dictionary
    '''
    # dimension of import word2vec file
    glove = {}
    with open(glove_path,'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            glove[word] = coefs
    return glove

def embeds(lang, EMBEDDING_DIM, non_trainable=True):
    '''
    match pretrained embedding and dataset
    embedding -> unique_words x emb_size
    '''
    # load glove
    glove_path = '../data/glove.6B.50d.txt'
    glove = pretrained(glove_path)
    # initialize embedding matrix for our dataset
    embedding_matrix = np.zeros((lang.n_words, EMBEDDING_DIM))
    # count words that appear only in the dataset. word_index.items() yields dict of word:index pair
    for word, ix in lang.word2index.items():
        embedding_vector = glove.get(word)
        if embedding_vector is not None:
            # words not found in glove matrix will be all-zeros.
            embedding_matrix[ix] = embedding_vector
    # change the datatype from numpy_array to torch_tensor
    weight = torch.FloatTensor(embedding_matrix)
    # convert to the embedding of pytorch
    embedding_matrix = nn.Embedding.from_pretrained(weight)
    # make it non-trainable
    if non_trainable:
        embedding_matrix.weight.requires_grad = False
    
    return embedding_matrix, glove
