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
# print teststring[3:3+8] #this is xanderni
# print teststring[11] #this is c

#We need to come up with the list of training patterns
base_in_len = 20
bases_in = []
base_out = []
for i in range(n_rows):
    for j in range(0, MAXLEN-base_in_len,1):
        if j+base_in_len < MAXLEN-1:
            current_base_in = dna_data.dna[i][j:j+base_in_len]
            #current_base_out = dna_data.dna[i][j+base_in_len]
            bases_in.append(current_base_in)
            #base_out.append(current_base_out)
n_patterns = len(bases_in)
print "Number of patterns: ", n_patterns