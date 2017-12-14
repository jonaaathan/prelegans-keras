# -*- coding: utf-8 -*-
from __future__ import division, print_function

import json
import os
import time
# import warnings
from collections import OrderedDict, defaultdict

import keras
import keras.backend as K
import keras.optimizers as opt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.engine import topology
from sklearn.model_selection import train_test_split
from tabulate import tabulate

def load_model(filepath, custom_objects=None, compile=True):
    pass

class Model(keras.models.Model):
    def __init__(self, inputs, outputs, name=None,
                 classification=False,
                 verbose=False,
                 generator=None,
                 nb_inputs=1,
                 nb_outputs=1):
        self.verbose = verbose
        self.generator = generator
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs

        self.classification = classification

        self.history = None
        self.train_loss = []
        self.valid_loss = []
        self.train_acc = []
        self.valid_acc = []
        '''
        self.fold_train_losses = None
        self.fold_val_losses = None
        self.k_fold_history = defaultdict(list)
        '''
        self.history_list = []
        super(Model, self).__init__(inputs=inputs, outputs=outputs, name=name)

    def compile(self, optimizer, loss, metrics=None, loss_weights=None,
                sample_weight_mode=None, **options):

        if isinstance(optimizer, str):
            opts = {'sgd': opt.SGD(lr=options.get('lr', .01),
                                   decay=options.get('decay', 1e-6),
                                   momentum=options.get('momentum', 0.9), nesterov=True,
                                   clipnorm=options.get('clipnorm', 0)),
                    'rmsprop': opt.RMSprop(lr=options.get('lr', .001)),
                    'adadelta': opt.Adadelta(lr=options.get('lr', 1.)),
                    'adagrad': opt.Adagrad(lr=options.get('lr', .01)),
                    'adam': opt.Adam(lr=options.get('lr', .001)),
                    'nadam': opt.Nadam(lr=options.get('lr', .002),
                                       clipnorm=options.get('clipnorm', 0)),
                    'adamax': opt.Adamax(lr=options.get('lr', .002))
                    }
            optimizer = opts[optimizer]
        super(Model, self).compile(loss=loss,
                                   loss_weights=loss_weights,
                                   optimizer=optimizer,
                                   sample_weight_mode=sample_weight_mode,
                                   metrics=metrics)

    def fit(self, x=None, y=None,
            batch_size=32,
            epochs=1,
            initial_epoch=0,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            return_best_model=True,
            reduce_factor=.5,
            **kwargs):
        # Check arguments
        assert (validation_split >= 0)
        assert epochs > 0

        if validation_data:
            x_train, y_train = x, y
            x_valid, y_valid = validation_data

        elif validation_split > 0:

            if self.classification:
                stratify = y
            else:
                stratify = None

            x_train, x_valid = train_test_split(x, test_size=validation_split,
                                                stratify=stratify,
                                                shuffle=shuffle,
                                                random_state=5)
            y_train, y_valid = train_test_split(y, test_size=validation_split, stratify=stratify, random_state=5)
        else:
            x_train, y_train = x, y
            x_valid, y_valid = [], []

        if return_best_model:
            rn = np.random.random()
            checkpoint = ModelCheckpoint('/tmp/best_{0}.h5'.format(rn),
                                         monitor='val_loss',
                                         verbose=1,
                                         mode='min',
                                         save_best_only=True,
                                         save_weights_only=True)
            # TODO: this is multiply defined, define only once per fit
            callbacks.append(checkpoint)

        start_time = time.time()
        try:
            super(Model, self).fit(x=x_train, y=y_train,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   initial_epoch=initial_epoch,
                                   verbose=verbose,
                                   callbacks=callbacks,
                                   validation_data=(x_valid, y_valid),
                                   shuffle=True,
                                   class_weight=None,
                                   sample_weight=None)
        except KeyboardInterrupt:
            pass

        if return_best_model:
            try:
                self.load_all_param_values('/tmp/best_{0}.h5'.format(rn))
            except:
                print('Unable to load best parameters, saving current model.')

        self.history_list.append(self.history)

        print('Model trained for {0} epochs. Total time: {1:.3f}s'.format(len(self.history.epoch),
                                                                          time.time() - start_time))

        return x_valid, y_valid

    def display_network_info(self):

        print("Neural Network has {0} trainable parameters".format(self.n_params))

        layers = self.get_all_layers()

        ids = list(range(len(layers)))

        names = [layer.name for layer in layers]

        shapes = ['x'.join(map(str, layer.output_shape[1:])) for layer in layers]
        # TODO: maybe show weights shape also

        params = [layer.count_params() for layer in layers]

        tabula = OrderedDict([('#', ids), ('Name', names), ('Shape', shapes), ('Parameters', params)])

        print(tabulate(tabula, 'keys'))

    def get_all_layers(self):
        return self.layers
    def get_all_params(self, trainable=True):
        if trainable:
            return self.weights
        else:
            return self.non_trainable_weights

    @property
    def n_params(self):
        return self.count_params()

    def save(self, handle, data_dir=None, **save_args):
        handle.ftype = 'model'
        handle.epochs = len(self.history.epoch)
        filename = 'models/' + handle
        if data_dir:
            filename = os.path.join(data_dir, filename)

        if not os.path.exists('/'.join(filename.split('/')[:-1])):
            os.makedirs('/'.join(filename.split('/')[:-1]))

        super(Model, self).save(filename, **save_args)

        print('Model saved to: ' + filename)

    def save_train_history(self, handle, data_dir=None):
        handle.ftype = 'history'
        handle.epochs = len(self.history.epoch)
        filename = 'models/' + handle
        if data_dir:
            filename = os.path.join(data_dir, filename)

        if not os.path.exists('/'.join(filename.split('/')[:-1])):
            os.makedirs('/'.join(filename.split('/')[:-1]))

        np.savez_compressed(filename, **self.history.history)

        print('History saved to: ' + filename)
