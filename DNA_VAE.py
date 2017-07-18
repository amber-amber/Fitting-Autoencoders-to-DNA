import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics

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

n_rows = 10000
MAXLEN = 30
dna_data = pd.read_csv('coreseed.train.tsv', names=["dna","protein"], usecols=[5,6], nrows= n_rows, delimiter ='\t', header =0)
n,m=dna_data.shape
dna_data.dna=dna_data.dna.str[:MAXLEN]

print('VECTORIZATION and/or CREATING TRAIN/TEST SETS.......')
hot=np.zeros((n,MAXLEN,len(chars)), dtype=np.bool)
print 'Shape of encoded data: ',hot.shape
for i, dna_str in enumerate(dna_data.dna):
    hot[i]=ctable.encode(dna_str, MAXLEN)
#Do we need to one hot vectorize if we are using variational autoencoder?

# Split the DNA data
# split_at = int(.75 * n)
# dna_train, dna_test = hot[:split_at], hot[split_at:]
# print "Previous training set shape", dna_train.shape
# print "Previous test set shape", dna_test.shape
# #
# dna_train = dna_train.reshape((len(dna_train), np.prod(dna_train.shape[1:])))
# dna_test = dna_test.reshape((len(dna_test), np.prod(dna_test.shape[1:])))
# print "New training set shape", dna_train.shape
# print "New test set shape", dna_test.shape

hot = hot.reshape(len(hot), np.prod(hot.shape[1:]))
print "New Shape of encoded data: ", hot.shape
print type(hot)

#the VAE
batch_size = 100
#original_dim = dna_train.shape[1]
original_dim = hot.shape[1]
latent_dim = 2
intermediate_dim = 60
epochs = 30
epsilon_std = 1.0

#this is how we generate new test samples?
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


x = Input(batch_shape=(batch_size, original_dim))
print 'Input shape: ', x._keras_shape
h = Dense(intermediate_dim, activation='relu')(x)
print 'Dense shape: ', h._keras_shape
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
print "z_mean shape: ", z_mean.shape
print "z_log_var shape: ", z_log_var.shape
z = Lambda(sampling)([z_mean, z_log_var])
print "Shape after lambda layer: ", z._keras_shape

# # note that "output_shape" isn't necessary with the TensorFlow backend
#z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
#z = Lambda(sampling)([z_mean, z_log_var])
# #this is giving me some sort of error although both z_mean shape and z_log_var shape are (100,2)

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)
print "x_decoded_mean shape: ", x_decoded_mean._keras_shape

#Custom loss layer
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
#
y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None, metrics=['accuracy'])
print('THE VARIATIONAL AUTOENCODER MODEL...')
vae.summary()

#vae.fit(hot, shuffle=True, epochs=epochs, batch_size=batch_size, validation_split=.25)
#why is this giving me >>>>> TypeError: unhashable type: 'list' <<<<< omg hot is not a list. can't use this because hot is hashable?!
#vae.fit(dna_test, shuffle=True, epochs=epochs, batch_size=batch_size, test_data=(dna_test, dna_test))
#but this will give me the error ValueError: The model expects 0 input arrays, but only received one array.

#encoder = Model(x, z_mean)
#serWarning: Model inputs must come from a Keras Input layer, they cannot be the output of a previous non-Input layer. Here, a tensor specified as input to "model_2" was not an Input tensor, it was generated by layer custom_variational_layer_1.

#We want to somehow determine the generated DNA seqences
#Based off the Variational Autoencoders tutorial, we should be sampling from a normal distribution to get the test samples

print('GENERATING TEST SAMPLES...')

#this is the decoder that will generate the sample
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
#print "Test sample in the intermediate dim: ", _h_decoded._keras_shape
_x_decoded_mean = decoder_mean(_h_decoded)
#print "Test sample in the orginal dim", _x_decoded_mean._keras_shape
generator = Model(decoder_input, _x_decoded_mean)

#let's sample some random Gaussians which we will plug into the decoder
num_test_samples = 15
for i in range(num_test_samples):
    Gaussian_sample_x = np.random.normal(0,1)
    Gaussian_sample_y = np.random.normal(0,1)
    z_sample = np.array([[Gaussian_sample_x,Gaussian_sample_y]])
    sample_decoded = generator.predict(z_sample)
    sample_decoded = sample_decoded.reshape(MAXLEN, len(chars))
    print ctable.decode(sample_decoded)
    i+=1

