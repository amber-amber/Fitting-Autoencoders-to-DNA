import numpy as np
import pandas as pd
import tensorflow as tf

import sys
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

from keras.layers import Input, Dense, Lambda, LSTM, Dropout, Reshape
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.optimizers import SGD, Adam, RMSprop
from keras.losses import kullback_leibler_divergence, categorical_crossentropy, binary_crossentropy
from keras.callbacks import TensorBoard, EarlyStopping

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

n_rows = 200000
MAXLEN = 20
dna_data = pd.read_csv('coreseed.train.tsv', names=["dna","protein"], usecols=[5,6], nrows= n_rows, delimiter ='\t', header =0)
#dna_data = pd.read_csv('coreseed.train.tsv', names=["dna","protein"], usecols=[5,6], delimiter ='\t', header =0)
n,m=dna_data.shape
dna_data.dna=dna_data.dna.str[:MAXLEN]

print('VECTORIZATION and/or CREATING TRAIN/TEST SETS.......')
hot=np.zeros((n,MAXLEN,len(chars)), dtype=np.bool)
print('Shape of encoded data: ',hot.shape)
for i, dna_str in enumerate(dna_data.dna):
    hot[i]=ctable.encode(dna_str, MAXLEN)
#Do we need to one hot vectorize if we are using variational autoencoder?

# #Split the DNA data
# split_at = int(.75 * n)
# dna_train, dna_test = hot[:split_at], hot[split_at:]
# print "Previous training set shape", dna_train.shape
# print "Previous test set shape", dna_test.shape
# #
# dna_train = dna_train.reshape((len(dna_train), np.prod(dna_train.shape[1:])))
# dna_test = dna_test.reshape((len(dna_test), np.prod(dna_test.shape[1:])))
# print "New training set shape", dna_train.shape
# print "New test set shape", dna_test.shape

# hot_reshaped = hot.reshape(len(hot), np.prod(hot.shape[1:]))
# print("New Shape of encoded data: ", hot_reshaped.shape)
#print(type(hot))

#the VAE

#some parameters
batch_size = 100
#original_dim = dna_train.shape[1]
print("ORIGINAL DIM: ", hot.shape[1] )
original_dim = hot.shape[1]
latent_dim = 24
#why is the latent dimension so small in comparison to the intermediate dim?
intermediate_dim = 100
epochs = 2
epsilon_std = 1.0
dropout_rate = 0.4
lstm_size = 100

#this is how we generate new test samples
#we will sample an epsilon from ~N(0,1) then convert
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

#for Q(z|X) the encoder
#this is a neural net with ONE hidden layer
# x = Input(batch_shape=(batch_size, original_dim))
x = Input(shape=(MAXLEN, len(chars)))
# x = Input(shape=(batch_size, MAXLEN, len(chars)))
print('Input shape: ', x._keras_shape)
#x =Dropout(dropout_rate, input_shape=(MAXLEN,len(chars)))(x) CANNOT USE DROPOUT ON THE INPUT LAYER ?!?!?!
h = Reshape((MAXLEN * len(chars),))(x)
h = Dense(intermediate_dim, activation='relu')(h)
# h = LSTM(intermediate_dim, input_shape =(MAXLEN,len(chars)))(x)
# h = LSTM(lstm_size, input_shape =(MAXLEN,len(chars)))(x)
h = Dropout(dropout_rate)(h)
print('Shape after LSTM Layer: ', h._keras_shape)
h = Dense(intermediate_dim)(h)
#print('Shape after Dense Layer: ', h.output_shape)

# x = Input(shape=(n_rows, MAXLEN,len(chars)))
# print('Input shape: ', x._keras_shape)
# x = LSTM(batch_size, input_shape=(MAXLEN, len(chars)))
# print('Shape after LSTM Layer: ', x._keras_shape)
# h = Dense(intermediate_dim, activation='relu')(x)


#print 'Dense shape: ', h._keras_shape
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
#WHY ARE THESE THE EXACT SAME?!?!!
print("z_mean shape: ", z_mean._keras_shape)
print("z_log_var shape: ", z_log_var._keras_shape)
z = Lambda(sampling)([z_mean, z_log_var])
print("Shape after lambda layer: ", z._keras_shape)

#for P(X|z) the decoder
decoder_h = Dense(intermediate_dim, activation='relu')
#print("Shape after first NN layer of the decoder: ", decoder_h._keras_shape)
decoder_mean = Dense(original_dim*len(chars), activation='sigmoid')
decoder_mean_reshaped = Reshape((original_dim, len(chars)))
#print("Shpae after second NN layer of the decoder: ", decoder_mean._keras_shape)
h_decoded = decoder_h(z)
#print('h_decoded shape: ', h_decoded._keras_shape)
x_decoded_mean = decoder_mean(h_decoded)
x_decoded_mean_reshaped = decoder_mean_reshaped(x_decoded_mean)
#x_decoded_mean = decoder_h(h_decoded).reshape(batch_size,MAXLEN,len(char))
print("x_decoded_mean shape: ", x_decoded_mean._keras_shape)

# #Custom loss layer

def vae_loss(y_true, y_pred):
    recon_loss = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
    #loss obviously depends on MAXLEN. but it seems like binary crossentropy stays approx the same regardless
    KL_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return recon_loss
    # return recon_loss + KL_loss/MAXLEN

# class CustomVariationalLayer(Layer):
#     def __init__(self, **kwargs):
#         self.is_placeholder = True
#         super(CustomVariationalLayer, self).__init__(**kwargs)
#
#     def vae_loss(self, x, x_decoded_mean):
#         xent_loss = MAXLEN * metrics.binary_crossentropy(x, x_decoded_mean)
#         kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#         #this is like the KL divergence. I'm guessing z_mean and z_log_var are like the 2 distributions
#         return K.mean(xent_loss + kl_loss)
#         #return kl_loss
#
#     def call(self, inputs):
#         x = inputs[0]
#         x_decoded_mean = inputs[1]
#         loss = self.vae_loss(x, x_decoded_mean)
#         self.add_loss(loss, inputs=inputs)
#         # We won't actually use the output.
#         return x

# class CustomVariationalLayerKL(Layer):
#     def __init__(self, **kwargs):
#         self.is_placeholder = True
#         super(CustomVariationalLayerKL, self).__init__(**kwargs)
#
#     def vae_loss(self):
#         #xent_loss = MAXLEN * metrics.binary_crossentropy(x, x_decoded_mean)
#         kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#         #this is like the KL divergence. I'm guessing z_mean and z_log_var are like the 2 distributions
#         return kl_loss
#         #return kl_loss
#
#     def call(self, inputs):
#         x = inputs[0]
#         #x_decoded_mean = inputs[1]
#         loss = self.vae_loss
#         self.add_loss(loss, inputs=inputs)
#         # We won't actually use the output.
#         return x

#What happens whe we use the categorical accuracy metric?
# def categorical_accuracy(y_true, y_pred):
#     return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))

def covariance(x, y):
    return K.mean(x * y) - K.mean(x) * K.mean(y)

def corr_matrix(x,y):
    #x = np.array(x, dtype=)
    matrix = np.corrcoef(x,y)
    return np.argmax(matrix)

# def hamming_distance(x,y):
#     x = tf.reshape(x, shape=(n_rows, MAXLEN*len(chars)))
#     x = x.eval(session=sess)
#     y = tf.reshape(y, shape=(n_rows, MAXLEN * len(chars)))
#     y = y.eval(session=sess)
#     row = np.random.randint(0,n_rows)
#     num_incorrect = 0
#     for i, value in enumerate(x[row]):
#         if value != y[row][i]:
#             num_incorrect+=1
#     return num_incorrect/(MAXLEN*len(chars))

def corr(y_true, y_pred):
    cov = covariance(y_true, y_pred)
    var1 = covariance(y_true, y_true)
    var2 = covariance(y_pred, y_pred)
    return cov / (K.sqrt(var1 * var2) + K.epsilon())


def xent(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)

#The actual VAE

#y = CustomVariationalLayer()([x, x_decoded_mean])
#y = CustomVariationalLayerKL()([x,x_decoded_mean])
#vae = Model(x, y)
# vae = Model(x, x_decoded_mean)
vae = Model(x, x_decoded_mean_reshaped)

learning_rate = 0.00001
#optimizer = SGD(lr=learning_rate)
#optimizer = RMSprop(lr=learning_rate)
optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

for_tb = TensorBoard(log_dir='DNA_VAE',histogram_freq=0, write_graph=True, write_images=True)
vae.compile(optimizer= optimizer, loss=vae_loss, metrics=[xent, corr, 'acc'])
print('THE VARIATIONAL AUTOENCODER MODEL...')
vae.summary()

# vae.fit(hot, hot_reshaped, shuffle=True, epochs=epochs, batch_size=batch_size, validation_split=.25, callbacks=[for_tb])
vae.fit(hot, hot, shuffle=True, epochs=epochs, batch_size=batch_size, validation_split=.25, callbacks=[for_tb])

encoder = Model(x, z_mean)

#We want to somehow determine the generated DNA seqences
#Based off the Variational Autoencoders tutorial, we should be sampling from a normal distribution to get the test samples
#this is the decoder that will generate the sample
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
x_decoded_mean_reshaped = decoder_mean_reshaped(x_decoded_mean)
generator = Model(decoder_input, x_decoded_mean_reshaped)


#let's sample some random Gaussians which we will plug into the decoder
num_test_samples = 5
for i in range(num_test_samples):
    Gaussian_sample_x = np.random.normal(0,1)
    Gaussian_sample_y = np.random.normal(0,1)
    z_sample = np.array([[Gaussian_sample_x,Gaussian_sample_y]])
    sample_decoded = generator.predict(z_sample)
    #sample_decoded = sample_decoded.reshape(MAXLEN, len(chars))
    print(ctable.decode(sample_decoded))
    i+=1
