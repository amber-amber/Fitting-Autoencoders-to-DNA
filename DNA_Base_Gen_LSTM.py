import numpy as np
import pandas as pd

#from keras.models import Sequential
#from keras.layers import Dense, Dropout,LSTM

n_rows = 10000
MAXLEN = 60
dna_data = pd.read_csv('coreseed.train.tsv', names=["dna","protein"], usecols=[5,6], nrows= n_rows, delimiter ='\t', header =0)
#n,m = dna_data.shape
dna_data.dna=dna_data.dna.str[:MAXLEN]
print "DNA shape: ", dna_data.shape

# teststring = 'alexandernicholasskolnick'
# print len(teststring)
# length=8
# for j in range(len(teststring)-length):
#     print current_letters_in
#     print teststring[j+length]


#We need to come up with the list of training patterns
base_in_len = 20
bases_in = []
base_out = []
for i in range(n_rows):
    for j in range(len(dna_data.dna[i])-base_in_len):
         current_base_in = dna_data.dna[i][j:j+base_in_len]
         if (j+base_in_len) < (MAXLEN-2):
            current_base_out = dna_data.dna[i][j+base_in_len]
            bases_in.append(current_base_in)
            base_out.append(current_base_out)
n_patterns = len(bases_in)
print "Number of patterns: ", n_patterns
print "Sample of bases in:" bases_in[888]
print "Sample of base_out:" base_out[888]

#Vectorization
class CharacterTable(object):
    def __init__(self, chars):
        self.chars=sorted(set(chars))
        self.char_indices=dict((c,i) for i,c in enumerate(self.chars))
        self.indices_char=dict((i,c) for i,c in enumerate(self.chars))

    def encode(self, C, num_rows):
        x = np.zeros((num_rows,len(self.chars)))
        for i,c in enumerate(C):
            x[i, self.char_indices[c]]=1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x. argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)

chars='actg'
ctable= CharacterTable(chars)

print('VECTORIZATION')
hot_x=np.zeros((n_patterns,MAXLEN,len(chars)), dtype=np.bool)
#print('shape of vector: ',hot_x.shape)
for i, dna_str in enumerate(bases_in):
    hot_x[i]=ctable.encode(dna_str, MAXLEN)
print hot_x[1:3]