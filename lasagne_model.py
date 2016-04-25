import h5py
import numpy as np
import theano
import theano.tensor as T
import lasagne

names = ['min_train_X', 'min_train_y', 'min_test_X', 'min_test_y',
         'max_train_X', 'max_train_y', 'max_test_X', 'max_test_y']

h5f = h5py.File('station_data.h5', 'w')
min_train_X = h5f['min_train_X'][:]
min_train_y = h5f['min_train_y'][:]
min_spread = len(min_train_X[0, 0, :])

# Sequence Length
SEQ_LENGTH = len(min_train_X[0, :, 0])

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 512

# Optimization learning rate
LEARNING_RATE = .01

# All gradients above this will be clipped
GRAD_CLIP = 100

# Number of epochs to train the net
NUM_EPOCHS = 50

# Batch Size
BATCH_SIZE = 128

l_in = lasagne.layers.InputLayer(shape=(None, None, min_spread))

l_forward_1 = lasagne.layers.LSTMLayer(
    l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
    nonlinearity=lasagne.nonlinearities.tanh)

l_forward_2 = lasagne.layers.LSTMLayer(
    l_forward_1, N_HIDDEN, grad_clipping=GRAD_CLIP,
    nonlinearity=lasagne.nonlinearities.tanh)

l_forward_slice = lasagne.layers.SliceLayer(l_forward_2, -1, 1)

l_out = lasagne.layers.DenseLayer(
    l_forward_slice, num_units=min_spread,
    W = lasagne.init.Normal(),
    nonlinearity=lasagne.nonlinearities.softmax)

target_values = T.ivector('target_output')
network_output = lasagne.layers.get_output(l_out)

cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()

all_params = lasagne.layers.get_all_params(l_out,trainable=True)
updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

print("Compiling functions ...")
train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
compute_cost = theano.function([l_in.input_var, target_values], cost, allow_input_downcast=True)

probs = theano.function([l_in.input_var],network_output,allow_input_downcast=True)

for epoch in range(0, NUM_EPOCHS):
    cost = train(min_train_X, min_train_y)
    print('Epoch: ' + str(epoch) + ' | Cost: ' + str(cost))

h5f.close()
