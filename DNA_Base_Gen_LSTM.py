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
# counter=0
# for j in range(len(teststring)-length):
#     current_letters_in = teststring[j:j+length]
#     counter= counter+1
#     print current_letters_in
#     print teststring[j+length]
# print counter

#We need to come up with the list of training patterns
base_in_len = 20
bases_in = []
base_out = []
for i in range(n_rows):
    for j in range(MAXLEN-base_in_len):
         current_base_in = dna_data.dna[i][j:j+base_in_len]
         #if (j+2+base_in_len) < MAXLEN:
         current_base_out = dna_data.dna[i][j+base_in_len]
         bases_in.append(current_base_in)
         base_out.append(current_base_out)
n_patterns = len(bases_in)
print "Number of patterns: ", n_patterns