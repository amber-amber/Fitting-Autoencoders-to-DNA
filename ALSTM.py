#almost like the VAE EXCEPT the encoding layer has feature extraction

#LSTM Layer, output is (NONE, BATCHSIZE)

#Input batch_shape in VAE  (BATCHSIZE, MAXLEN)
#Then encode (dense layer) + (another dense layer)


import numpy as np
import pandas as pd

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

from keras.layers import Input, Dense, Lambda, Layer, LSTM, RepeatVector
from keras.models import Model
from keras import backend as K
from keras.metrics import categorical_accuracy
from keras.optimizers import SGD, Adam, RMSprop
from keras.losses import kullback_leibler_divergence

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

n_rows = 20000
MAXLEN = 80
dna_data = pd.read_csv('coreseed.train.tsv', names=["dna","protein"], usecols=[5,6], nrows= n_rows, delimiter ='\t', header =0)
#dna_data = pd.read_csv('coreseed.train.tsv', names=["dna","protein"], usecols=[5,6], delimiter ='\t', header =0)
n,m=dna_data.shape
dna_data.dna=dna_data.dna.str[:MAXLEN]

print('VECTORIZATION and/or CREATING TRAIN/TEST SETS.......')
hot=np.zeros((n,MAXLEN,len(chars)), dtype=np.bool)
print 'Shape of encoded data: ',hot.shape
for i, dna_str in enumerate(dna_data.dna):
    hot[i]=ctable.encode(dna_str, MAXLEN)

#some parameters
batch_size = 100
#original_dim = dna_train.shape[1]
original_dim = hot.shape[1]
#latent_dim = 2
#why is the latent dimension so small in comparison to the intermediate dim?
#Maybe our latent dim should be 8? Like an 8-mer...
latent_dim = 8
intermediate_dim = 60
epochs = 100
epsilon_std = 1.0

#this is how we generate new test samples
#we will sample an epsilon from ~N(0,1) then convert
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

#ENCODER MODEL
#LSTM Layer(s) then encoder
inputs = Input(batch_shape=(n_rows, MAXLEN,len(chars)))
#inputs = Input(batch_shape=(MAXLEN,len(chars)))
print('Input into the LSTM layer', inputs._keras_shape)
x = LSTM(batch_size, input_shape = (MAXLEN,len(chars)))
print('Output shape of the LSTM Layer', x._keras_shape)
#output shape IS NOT (None, batch_size) but rather (n_rows, batch_size)
#want to add a Dense layer
h = Dense(intermediate_dim)(x)
print('Outputshape of the dense layer', h._keras_shape)
#Compute Mu(x) and Sigma(x) from h via more dense layers
z_mean = Dense(latent_dim)(h)
print('z_mean shape: ', z_mean._keras_shape)
z_log_var = Dense(latent_dim)(h)
print('z_log_var shape: ', z_log_var._keras_shape)
#should be (batch_size, latent_dim) NOT (n_rows, latent_dim)

encoder = Model(x, z_mean)

optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
encoder.compile(optimizer= optimizer, loss=kullback_leibler_divergence, metrics=['categorical_accuracy'])

#z = Lambda(sampling)([z_mean, z_log_var])



# #for P(X|z) the decoder
# decoder_h = Dense(intermediate_dim, activation='relu')
# decoder_mean = Dense(original_dim, activation='sigmoid')
# h_decoded = decoder_h(z)
# x_decoded_mean = decoder_mean(h_decoded)
# print "x_decoded_mean shape: ", x_decoded_mean._keras_shape
#
# #Custom loss layer
# class CustomVariationalLayer(Layer):
#     def __init__(self, **kwargs):
#         self.is_placeholder = True
#         super(CustomVariationalLayer, self).__init__(**kwargs)
#
#     def vae_loss(self, x, x_decoded_mean):
#         xent_loss = MAXLEN * metrics.binary_crossentropy(x, x_decoded_mean)
#         kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#         #this is like the KL divergence. I'm guessing z_mean and z_log_var are like the 2 distributions
#         #return K.mean(xent_loss + kl_loss)
#         return kl_loss
#
#     def call(self, inputs):
#         x = inputs[0]
#         x_decoded_mean = inputs[1]
#         loss = self.vae_loss(x, x_decoded_mean)
#         self.add_loss(loss, inputs=inputs)
#         # We won't actually use the output.
#         return x
#
# #What happens whe we use the categorical accuracy metric?
# def categorical_accuracy(y_true, y_pred):
#     return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))
#
# #The actual VAE
#
# y = CustomVariationalLayer()([x, x_decoded_mean])
# vae = Model(x, y)
#
# learning_rate = 0.0001
# #optimizer = SGD(lr=learning_rate)
# #optimizer = RMSprop(lr=learning_rate)
# optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# vae.compile(optimizer= optimizer, loss=None, metrics=['categorical_accuracy'])
# print('THE VARIATIONAL AUTOENCODER MODEL...')
# vae.summary()
#
# vae.fit(hot, hot, shuffle=True, epochs=epochs, batch_size=batch_size, validation_split=.25)
#
# encoder = Model(x, z_mean)
# #serWarning: Model inputs must come from a Keras Input layer, they cannot be the output of a previous non-Input layer. Here, a tensor specified as input to "model_2" was not an Input tensor, it was generated by layer custom_variational_layer_1.
#
# #We want to somehow determine the generated DNA seqences
# #Based off the Variational Autoencoders tutorial, we should be sampling from a normal distribution to get the test samples
#
# print('GENERATING TEST SAMPLES...')
# #this is the decoder that will generate the sample
# decoder_input = Input(shape=(latent_dim,))
# _h_decoded = decoder_h(decoder_input)
# #print "Test sample in the intermediate dim: ", _h_decoded._keras_shape
# _x_decoded_mean = decoder_mean(_h_decoded)
# #print "Test sample in the orginal dim", _x_decoded_mean._keras_shape
# generator = Model(decoder_input, _x_decoded_mean)
#
# #let's sample some random Gaussians which we will plug into the decoder
# num_test_samples = 5
# for i in range(num_test_samples):
#     Gaussian_sample_x = np.random.normal(0,1)
#     Gaussian_sample_y = np.random.normal(0,1)
#     z_sample = np.array([[Gaussian_sample_x,Gaussian_sample_y]])
#     sample_decoded = generator.predict(z_sample)
#     sample_decoded = sample_decoded.reshape(MAXLEN, len(chars))
#     print(ctable.decode(sample_decoded))
#     i+=1
#
