from __future__ import print_function
import sys
import os
import time

from json import dumps
import numpy as np
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
import random

import tensorflow as tf
tf.set_random_seed(42)

import keras
from keras.optimizers import RMSprop
import pandas as pd
from pandas import read_csv
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import backend as K
from keras import callbacks as cb
from keras.utils import plot_model

import nn as nets
from tools import load_dataset, preprocess_dataset, Handle, retrieve_past, plot_loss_history

from absl import flags

# import seaborn as sns
import time

##########################
# todo: normalize each columns (by .describe())
#       v remove NaN
#       v test_set on unseen data
#       v put weights in the sample in test_set
#       balance class in model.fit() by adding class_weight
#       timestep = 10 for predicting future actions. now it is 1 row for 1
#       plot pos/neg distribution on time
#       TODO: plot zones of DMP events zone (check R scripts?)
#       TODO: add PCs to
#
##########################
flags.DEFINE_string("infile", '', 'The protein dataset file to be trained on.')
flags.DEFINE_string("cohst_neg_file", '', 'The protein dataset file to use as a negative set on CoHST')

flags.DEFINE_string("key", 'fam', 'The key to use for codes.')
flags.DEFINE_enum("mode", 'predict', ['predict', 'generate', 'image_predict'], 'The mode to train CoMET.')
flags.DEFINE_integer("epochs", 50, 'The number of training epochs to perform.', lower_bound=1)
flags.DEFINE_integer("batch_size", 50, 'The size of the mini-batch.', lower_bound=1)
flags.DEFINE_float("validation_split", 0.2, "The fraction of data to use for cross-validation.", lower_bound=0.0,
                   upper_bound=1.0)

flags.DEFINE_string("model", '', 'Continue training the given model. Other architecture options are unused.')

flags.DEFINE_string("data_dir", '', 'The directory to store elegan output data.')

FLAGS = flags.FLAGS

try:
    FLAGS(sys.argv)
except flags.Error as e:
    print(e)
    print(FLAGS)
    sys.exit(1)

def precision(y_true, y_pred):

    # Count positive samples
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    predicted_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    print (predicted_positives, all_positives)
    if all_positives != 0:
        recall = predicted_positives / (all_positives + K.epsilon())
    else:
        recall = 0
    return recall

def prepare_data(data, dropna=False, size=10000):
    X = data.loc[:,['length', 'comSpeed1', 'bodyAxisSpeed1', 'pumpingRate']]
    Y = data.loc[:, 'DMPevent']
    print ('total value count', (Y==1).sum(), (Y==0).sum())

    nPos = (Y==1).sum()
    nNeg = (Y==0).sum()
    DMP_weight = Y.apply(lambda x: nNeg if x == 1 else nPos)
    # print (DMP_weight)
    train_set = (X.head(n=size), Y.head(n=size))
    # y_train = Y.head(n=100000)
    print ('train value count', (train_set[1]==1).sum(), (train_set[1]==0).sum)
    y_test = Y.sample(n=size/2, weights=DMP_weight)
    print ('test value count', (y_test==1).sum(), (y_test==0).sum())
    indexList = y_test.index.tolist()
    x_test = X.loc[indexList]
    test_set = (x_test, y_test)
    return train_set, test_set

# TODO: build pipeline to feed multiple experiement, yield dataframe while making separate comparisons
def predict(x_data, y_data, handle):
    if type(x_data) == np.ndarray:
        input_shape = x_data[0].shape
    elif type(x_data) == list:
        input_shape = (None, x_data[0].shape[1])
    else:
        raise TypeError('Something went wrong with the dataset type')

    if FLAGS.model:
        lstm_net = nets.build_lstm_model(saved_model=FLAGS.model)
        print('Loaded model')
    else:
        print('Building model')
        lstm_net = nets.build_lstm_model(input_shape)

    handle.model = lstm_net.name
    lstm_net.display_network_info()

    expId = random.getrandbits(16)
    param_size = lstm_net.n_params

    headers = {"size": str(param_size), "batch_size": str(batch_size), "id": expId, "epochs": FLAGS.epochs, "opt": FLAGS.optimizer}
    headers = dumps(headers)
    print (headers)
    remote = cb.RemoteMonitor(root='http://localhost:9000', headers={'hyper': headers})
    callbacks = [remote]
    # print('callbacks', callbacks, type(callbacks))

    print('Started training at {}'.format(time.asctime()))
    print ('x_data', x_data.shape)
    print ('bincount 0, 1 ', np.bincount(y_data))

    lstm_net.fit(x_data, y_data,
                 epochs=FLAGS.epochs,
                 batch_size=FLAGS.batch_size,
                 validation_split=FLAGS.validation_split,
                 callbacks=callbacks)

    lstm_net.save_train_history(handle, data_dir=FLAGS.data_dir)
    lstm_net.save(handle, data_dir=FLAGS.data_dir)

    # Extract the motifs from the convolutional layers
    # plot_loss_history(history_path)

########## Hyper ##########
path = '20170818_AM_N2_comp1_fullDataTable.tsv'
do = 0.3
size = 1000
batch_size = FLAGS.batch_size

def main():
    # dataset = read_csv(path, sep='\t', header=0)
    if FLAGS.model:
        handle = Handle.from_filename(FLAGS.model)
        assert handle.ftype == 'model'
        assert handle.model in ['prediction', 'generation', 'image_prediction'], 'The model file provided is for another program.'
    else:
        handle = Handle(**FLAGS.flag_values_dict())

    col_names = ['length', 'comSpeed1', 'bodyAxisSpeed1', 'unnamed:_2', 'unnamed:_3', 'unnamed:_4', 'unnamed:_5', 'unnamed:_6', 'unnamed:_7', 'unnamed:_8', 'unnamed:_9', 'unnamed:_10', 'unnamed:_11', 'unnamed:_12', 'unnamed:_13', 'unnamed:_14', 'unnamed:_15']
    x_raw, y_raw = load_dataset(path, columns=True, columns_name=col_names, codes=True, code_key='DMPevent')
    x_data, y_data = preprocess_dataset(x_raw, y_raw)
    # x_data still a dataframe

    # TODO: if dropna, check i didn't take neighbouring frames

    '''
    x_train = x_train.as_matrix()
    x_train = np.reshape(x_train, x_train.shape + (1,))
    print ('xtrain shape', x_train.shape)
    x_test = x_test.as_matrix()
    x_test = np.reshape(x_test, x_test.shape + (1,))
    print (x_test.shape)
    '''


    INPUT_SHAPE = x_data.shape
    print('input', INPUT_SHAPE)
    if FLAGS.mode == 'predict' or handle.model == 'predict':
        # refractor into load_dataset
        timestep = FLAGS.past
        print (timestep)
        if True:
            x_selected, y_selected = retrieve_past(x_data, y_data, timestep, sample=True)
            print (x_selected.shape, y_selected.shape)
            np.save('lstm_output/data/'+ path.split('.')[-2] + '_'+ str(timestep)       , x_selected)
            np.save('lstm_output/data/'+ path.split('.')[-2] + '_'+ str(timestep) + '_y', y_selected)
        else:
            start_time = time.time()
            x_selected = np.load('lstm_output/data/'+ path.split('.')[-2] + '_'+ str(timestep) + '.npy')
            y_selected = np.load('lstm_output/data/'+ path.split('.')[-2] + '_'+ str(timestep) + '_y.npy')
            print('load data. Total time: {0:.3f}s'.format(time.time() - start_time))

        assert (x_selected.shape[0] == y_selected.shape[0]), 'x y samples are not matched'

        predict(x_selected, np.asarray(y_selected), handle)

if __name__ == '__main__':
    main()
