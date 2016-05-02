import h5py
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.regularization import regularize_layer_params, l2

names = ['min_train_X', 'min_train_y', 'min_test_X', 'min_test_y',
         'max_train_X', 'max_train_y', 'max_test_X', 'max_test_y']

h5f = h5py.File('station_data.h5', 'r')
print(h5f.name)
val_split = 13000
min_train_X = h5f['.']['min_train_X'].value[:val_split]
print(len(min_train_X))
print(np.shape(min_train_X))
min_train_y = h5f['.']['min_train_y'].value[:val_split]
print(len(min_train_y))
print(np.shape(min_train_y))
min_val_X = h5f['.']['min_train_X'].value[val_split:]
print(len(min_val_X))
min_val_y = h5f['.']['min_train_y'].value[val_split:]
print(len(min_val_y))
min_test_X = h5f['.']['min_test_X'].value
min_test_y = h5f['.']['min_test_y'].value
h5f.close()
min_spread = len(min_train_X[0, 0, :])
print('Min spread: ' + str(min_spread))

# Hyperparameters:

# Sequence Length
SEQ_LENGTH = len(min_train_X[0, :, 0])

# Dropout Value
DROPOUT_VAL = 0.7

# L2 Regularization Value
L2_REG = 1e-4

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

def iterate_minibatches(inputs, targets, batchsize=128, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

l_in = lasagne.layers.InputLayer(shape=(None, None, min_spread))

l_forward_1 = lasagne.layers.LSTMLayer(
    l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
    nonlinearity=lasagne.nonlinearities.tanh)

l_dropout_1 = lasagne.layers.DropoutLayer(l_forward_1, DROPOUT_VAL)

l_forward_2 = lasagne.layers.LSTMLayer(
    l_dropout_1, N_HIDDEN, grad_clipping=GRAD_CLIP,
    nonlinearity=lasagne.nonlinearities.tanh)

l_dropout_2 = lasagne.layers.DropoutLayer(l_forward_2, DROPOUT_VAL)

l_forward_slice = lasagne.layers.SliceLayer(l_dropout_2, -1, 1)

l_out = lasagne.layers.DenseLayer(
    l_forward_slice, num_units=min_spread,
    W=lasagne.init.Normal(),
    nonlinearity=lasagne.nonlinearities.softmax)

net = l_out

l2_penalty = regularize_layer_params([l_forward_1, l_forward_2, l_out], l2) * L2_REG

target_values = T.imatrix('target_output')
network_output = lasagne.layers.get_output(net)

loss = T.nnet.categorical_crossentropy(network_output, target_values).mean()
loss += l2_penalty

all_params = lasagne.layers.get_all_params(l_out, trainable=True)
updates = lasagne.updates.adagrad(loss, all_params, LEARNING_RATE)

test_output = lasagne.layers.get_output(net, deterministic=True)
test_loss = T.nnet.categorical_crossentropy(test_output, target_values).mean()
test_acc = T.mean(T.eq(T.argmax(test_output, axis=1), T.argmax(target_values, axis=1)),
                       dtype=theano.config.floatX)

print("Compiling functions ...")
train_fn = theano.function([l_in.input_var, target_values], loss, updates=updates, allow_input_downcast=True)
test_fn = theano.function([l_in.input_var, target_values], [test_loss, test_acc], allow_input_downcast=True)

# probs = theano.function([l_in.input_var],network_output,allow_input_downcast=True)

for epoch in range(0, NUM_EPOCHS):
    train_err = 0
    train_batches = 0
    for batch in iterate_minibatches(min_train_X, min_train_y):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1
    print('Epoch: ' + str(epoch) + ' | Loss: ' + str(train_err / train_batches))
    test_loss = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(min_val_X, min_val_y):
        inputs, targets = batch
        [loss, acc] = test_fn(inputs, targets)
        test_loss += loss
        test_acc += acc
        test_batches += 1
    print('Val Loss: ' + str(test_loss / test_batches) + ' | Val Acc: ' + str(test_acc / test_batches))

params = lasagne.layers.get_all_param_values([l_forward_1, l_forward_2, l_out])
np.savez('lasagne_weights.npz', *params)
