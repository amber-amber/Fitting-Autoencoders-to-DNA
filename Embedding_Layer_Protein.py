from __future__ import print_function

import os
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.utils import np_utils

BASE_DIR = ''
GLOVE_DIR = BASE_DIR + '/glove.6B/'

n_rows = 5000
dna_data = pd.read_csv('coreseed.train.tsv', names=["dna","protein"], usecols=[5,6], nrows= n_rows, delimiter ='\t', header =0)

#Tokenizing the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dna_data.protein)
sequences = tokenizer.texts_to_sequences(dna_data.protein)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences)

print('Shape of data tensor:', data.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
split_at = int(.9*n_rows)
protein_train, protein_val = data[:split_at], data[split_at:]

#Need to make an embedding matrix
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
what is this os.path.join
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))