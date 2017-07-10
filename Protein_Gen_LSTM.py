import numpy as np
import pandas as pd
import random

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

n_rows = 10000
MAXLEN = 60
dna_data = pd.read_csv('coreseed.train.tsv', names=["dna","protein"], usecols=[5,6], nrows= n_rows, delimiter ='\t', header =0)
dna_data.protein=dna_data.protein.str[:MAXLEN]
print "DNA shape: ", dna_data.shape

# teststring = 'alexandernicholasskolnick'
# print len(teststring)
# length=8
# for j in range(len(teststring)-length):
#     print current_letters_in
#     print teststring[j+length]


#We need to come up with the list of training patterns
protein_in_len = 20
protein_in = []
protein_out = []
chars = ''
for i in range(n_rows):
    for j in range(len(dna_data.protein[i])-protein_in_len):
         current_protein_in = dna_data.protein[i][j:j+protein_in_len]
         prot_string = str(current_protein_in)
         chars= chars + prot_string
         if (j+protein_in_len) < (MAXLEN-2):
            current_protein_out = dna_data.protein[i][j+protein_in_len]
            protein_in.append(current_protein_in)
            protein_out.append(current_protein_out)
n_patterns = len(protein_in)
chars = sorted(list(set(chars)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
print "Number of protein characters: ", len(chars)
print "Number of patterns: ", n_patterns
print "Sample of bases in:", protein_in[889]
print "Sample of base_out:", protein_out[889]

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
hot_y=np.zeros((n_patterns, len(chars)),dtype=np.bool)
print 'Shape of input vector: ', hot_x.shape
print 'Shape of output vector: ', hot_y.shape
for i, prot_str in enumerate(protein_in):
#     hot_x[i]=ctable.encode(prot_str, protein_in_len)
    for t, char in enumerate(prot_str):
            hot_x[i,t,char_indices[char]] = 1
    hot_y[i, char_indices[protein_out[i]]] = 1
# for i, next_prot in enumerate(protein_out):
#     hot_y[i]=ctable.encode(next_prot,1)
#print 'example of encoded protein: ', hot_x[8]
#print 'example of encoded next protein: ', hot_y[8]

#The Single Layer LSTM Model

#Can we try SOME OTHER ENCODING

HIDDEN_SIZE =128
BATCH_SIZE=128
# LAYERS=1

print 'Build Model...'
model = Sequential()
#What if we wanted to use an embedding?
#model.add(layers.Embedding(BATCH_SIZE, input_length = protein_in_len, embeddings_initializer='uniform')
model.add(layers.LSTM(HIDDEN_SIZE,input_shape=(hot_x.shape[1], hot_x.shape[2])))
model.add(layers.Dense(len(chars)))
model.add(layers.Activation('softmax'))
# # model.add(layers.RepeatVector(MAXLEN))
# # for _ in range(LAYERS):
# #     model.add(layers.LSTM(HIDDEN_SIZE, return_sequences=True))
# #
# # model.add(layers.TimeDistributed(layers.Dense(len(chars))))
# # model.add(layers.Activation('softmax'))
# #model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
model.fit(hot_x, hot_y, epochs=60, batch_size=BATCH_SIZE)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

start_index = random.randint(0, n_patterns)
generated = ''
this_prot = str(protein_in[start_index])
generated += this_prot
print 'Generating with protein: ', generated


for diversity in [0.2, 0.5, 1.0, 1.2]:
    print 'Diversity: ', diversity
    for i in range(400):
        x = np.zeros((1, protein_in_len, len(chars)))
        for t, char in enumerate(this_prot):
            x[0,t,char_indices[char]] = 1
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
        generated += next_char
        this_prot = this_prot[1:]+next_char
        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()