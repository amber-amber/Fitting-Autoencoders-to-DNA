import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sys

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, Dense, Activation, Dropout
#from keras.optimizers import RMSprop
#from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, History

n_rows = 20000
MAXLEN = 60
#dna_data = pd.read_csv('coreseed.train.tsv', names=["dna","protein"], usecols=[5,6], delimiter ='\t', header =0)
dna_data = pd.read_csv('coreseed.train.tsv', names=["dna","protein"], usecols=[5,6], nrows= n_rows, delimiter ='\t', header =0)
num_amino_acids = len(dna_data.protein[0])
dna_data.protein=dna_data.protein.str[:MAXLEN]
print "DNA shape: ", dna_data.shape

#We need to come up with the list of training patterns
protein_in_len = 15
#protein_out_len = 1
protein_out_len = 3
step = 5
protein_in = []
protein_out = []
chars = ''
for i in range(n_rows):
    #for j in range(0,len(dna_data.protein[i])-protein_in_len, step):
    for j in range(0, len(dna_data.protein[i]) - protein_in_len - protein_out_len, step):
         current_protein_in = dna_data.protein[i][j:j+protein_in_len]
         prot_string = str(current_protein_in)
         chars= chars + prot_string
         if (j+protein_in_len) < (MAXLEN-protein_out_len):
            #current_protein_out = dna_data.protein[i][j + protein_in_len]
            current_protein_out = dna_data.protein[i][j+protein_in_len:j+protein_in_len+protein_out_len]
            protein_in.append(current_protein_in)
            protein_out.append(current_protein_out)
n_patterns = len(protein_in)
chars = sorted(list(set(chars)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
print "Number of protein characters: ", len(chars)
print "Number of patterns: ", n_patterns
print "Sample of bases in:", protein_in[8289]
print "Sample of base_out:", protein_out[8289]

#Vectorization
#Let's just use one hot encoding because that's all I know fml
# class CharacterTable(object):
#     def __init__(self, chars):
#         self.chars=sorted(set(chars))
#         self.char_indices=dict((c,i) for i,c in enumerate(self.chars))
#         self.indices_char=dict((i,c) for i,c in enumerate(self.chars))
#
#     def encode(self, C, num_rows):
#         x = np.zeros((num_rows,len(self.chars)))
#         for i,c in enumerate(C):
#             x[i, self.char_indices[c]]=1
#         return x
#
#     def decode(self, x, calc_argmax=True):
#         if calc_argmax:
#             x = x. argmax(axis=-1)
#         return ''.join(self.indices_char[x] for x in x)

#ctable= CharacterTable(chars)
#ONE HOT ENCODING
print 'VECTORIZATION'
hot_x=np.zeros((n_patterns,protein_in_len,len(chars)),dtype=np.bool)
#hot_y=np.zeros((n_patterns, len(chars)),dtype=np.bool)
hot_y=np.zeros((n_patterns, protein_out_len, len(chars)),dtype=np.bool)
print 'Shape of input vector: ', hot_x.shape
print 'Shape of output vector: ', hot_y.shape
for i, prot_str in enumerate(protein_in):
    for t, char in enumerate(prot_str):
        hot_x[i,t,char_indices[char]] = 1
    #hot_y[i, char_indices[protein_out[i]]] = 1
for i, next_prot in enumerate(protein_out):
    for t, char in enumerate(next_prot):
        hot_y[i,t, char_indices[char]]=1
print 'example of encoded protein: ', hot_x[888]
print 'example of encoded next protein: ', hot_y[888]

HIDDEN_SIZE =128
BATCH_SIZE=128
EMBEDDING_DIM = 100
LAYERS=1
dropout_rate = 0.4
learning_rate = 0.0001
epochs=50

print 'Build Model...'
model = Sequential()

model.add(LSTM(HIDDEN_SIZE, input_shape=(hot_x.shape[1], hot_x.shape[2])))
#here, the input shape is (protein_in_len, len(chars))
print "output shape after first LSTM layer", model.output_shape
#should be (None, HIDDEN_SIZE)
#model.add(LSTM(HIDDEN_SIZE, return_sequences=True, input_shape=(protein_in_len,len(chars))))

#model.add(Dense(len(chars)))
#print "output shape after dense", model.output_shape
#model.add(Activation('softmax'))
#print "output shape after activation", model.output_shape

#Adding LSTM layers
model.add(RepeatVector(protein_in_len))
model.add(Dropout(dropout_rate))
#model.add(LSTM(HIDDEN_SIZE,return_sequences=True))
model.add(LSTM(HIDDEN_SIZE))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

#for _ in range(LAYERS):
#   model.add(layers.LSTM(HIDDEN_SIZE, return_sequences=True))
#model.add(layers.TimeDistributed(layers.Dense(len(chars))))
#model.add(layers.Activation('softmax'))

# #model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#optimizer = RMSprop(lr=0.01)
#optimizer = SGD(lr=.01)
optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
#tb_callback = TensorBoard(log_dir='./2LayerLSTM_Graph', histogram_freq=0, write_graph=True, write_images=True)
history = model.fit(hot_x, hot_y, epochs=epochs, batch_size=BATCH_SIZE, validation_split=.25)
EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, mode='auto')
print(history.history.keys())
print(type(history.history))
print(history.history.val_acc)

# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# def sample(preds, temperature=1.0):
#     # helper function to sample an index from a probability array
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)
#
# start_index = random.randint(0, n_patterns)
# generated = ''
# this_prot = str(protein_in[start_index])
# generated += this_prot
# print 'Generating with protein: ', generated
#
#
# for diversity in [0.2, 0.5, 1.0, 1.2]:
#     print 'Diversity: ', diversity
#     for i in range(num_amino_acids):
#         x = np.zeros((1, protein_in_len, len(chars)))
#         for t, char in enumerate(this_prot):
#             x[0,t,char_indices[char]] = 1
#         preds = model.predict(x, verbose=0)[0]
#         next_index = sample(preds, diversity)
#         next_char = indices_char[next_index]
#         generated += next_char
#         this_prot = this_prot[1:]+next_char
#         sys.stdout.write(next_char)
#         sys.stdout.flush()
#     print()