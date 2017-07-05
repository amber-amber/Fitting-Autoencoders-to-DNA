#import numpy
import gzip, cPickle

mnist = gzip.open('mnist.pkl.gz mnist.pkl.gz','wb')
train_set, valid_set, test_set = cPickle.load(mnist)
mnist.close()

print('Training set shape: ', train_set.shape)
print('Validation set shape: ', valid_set.shape)
