# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras import layers
import sys

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

class colors:
    ok='\033[92m'
    fail= '\033[91m'
    close='\033[0m'

chars='actg'
ctable= CharacterTable(chars)

#MAXLEN=25
n_rows = 50000
dna_data = pd.read_csv('coreseed.train.tsv', names=["dna","protein"], usecols=[5,6], nrows= n_rows, delimiter ='\t', header =0)
n,m=dna_data.shape
for MAXLEN in range(30,100,10):
    print('The number of bases in the DNA is ', MAXLEN, \n)
    dna_data.dna=dna_data.dna.str[:MAXLEN]

    print('VECTORIZATION')
    x=np.zeros((n,MAXLEN,len(chars)),dtype=np.bool)
    print('shape of vector: ' ,x.shape)
    for i, dna_str in enumerate(dna_data.dna):
        x[i]=ctable.encode(dna_str,MAXLEN)

    split_at = int(.9*n)
    dna_train, dna_test= x[:split_at], x[split_at:]

    print('Training Data:')
    print(dna_train.shape)

    print('Testing Data:')
    print(dna_test.shape)

    RNN = layers.LSTM
    #RNN = layers.SimpleRNN
    HIDDEN_SIZE=128
    BATCH_SIZE=128
    LAYERS=1

    print('Build Model...')
    model=Sequential()
    model.add(RNN(HIDDEN_SIZE,input_shape=(MAXLEN,len(chars))))
    model.add(layers.RepeatVector(MAXLEN))
    for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE,return_sequences=True))

    model.add(layers.TimeDistributed(layers.Dense(len(chars))))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    #model.compile(loss='MSE',optimizer='adam',metrics=['accuracy'])
    model.summary()
    #model.fit(dna_train, dna_train, batch_size=BATCH_SIZE, epochs=1, validation_data=(dna_test, dna_test))
    # for i in range(10):
    #        ind=np.random.randint(0, len(dna_test))
    #        rowx, rowy = dna_test[np.array([ind])], dna_test[np.array([ind])]
    #        preds=model.predict_classes(rowx, verbose=0)
    #        q = ctable.decode(rowx[0])
    #        correct = ctable.decode(rowx[0])
    #        guess = ctable.decode(preds[0],calc_argmax=False)
    #        print('Q', q[::-1] )
    #        print('T', correct)
    #        if correct == guess:
    #            print(colors.ok + '☑' + colors.close)
    #        else:
    #            print(colors.fail + '☒' + colors.close)
    #        print(guess)
    #        print('---')

    def main(epochs):
    print()
    print('-'*50)
    model.fit(dna_train, dna_train, batch_size=BATCH_SIZE, epochs=epochs, validation_data=(dna_test, dna_test))
    for i in range(10):
        ind=np.random.randint(0, len(dna_test))
        rowx, rowy = dna_test[np.array([ind])], dna_test[np.array([ind])]
        preds=model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowx[0])
        guess = ctable.decode(preds[0],calc_argmax=False)
        print('Q', q[::-1])
        print('T', correct)
        if correct == guess:
            print(colors.ok + '☑' + colors.close)
        else:
            print(colors.fail + '☒' + colors.close)
        print(guess)
        print('---')

    if __name__ == '__main__':
    args = sys.argv[1:]
    main(int(args[0]))