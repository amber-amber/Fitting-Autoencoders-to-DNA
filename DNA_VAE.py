import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics

#batch_size = 100
#original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0

#the vectorization

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

n_rows = 10000
MAXLEN = 30
dna_data = pd.read_csv('coreseed.train.tsv', names=["dna","protein"], usecols=[5,6], nrows= n_rows, delimiter ='\t', header =0)
n,m=dna_data.shape
dna_data.dna=dna_data.dna.str[:MAXLEN]

print('VECTORIZATION')
hot=np.zeros((n,MAXLEN,len(chars)), dtype=np.bool)
print('shape of vector: ',x.shape)
for i, dna_str in enumerate(dna_data.dna):
    hot[i]=ctable.encode(dna_str, MAXLEN)

#the VAE

x = Input(batch_shape=(hot.shape))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(n_rows, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(MAXLEN, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = MAXLEN * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)

#fit to DNA data
split_at = int(.9 * n)
dna_train, dna_test = hot[:split_at], hot[split_at:]

dna_train = dna_train.reshape((len(dna_train), np.prod(dna_train.shape[1:])))
dna_test = dna_test.reshape((len(dna_test), np.prod(dna_test.shape[1:])))


vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))


