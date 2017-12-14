import keras.backend as K
from absl import flags

from keras.layers import  Dense, Flatten, Input, LSTM
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.metrics import binary_accuracy, categorical_accuracy
from keras import regularizers

#  Model and load_model
from . import network_parameters
from engines import load_model, Model

flags.adopt_module_key_flags(network_parameters)

FLAGS = flags.FLAGS

def build_lstm_model(input_shape=None, saved_model=None):
    if saved_model:
        model = load_model(saved_model, compile=False)
        model.classification = True
    else:
        # Check input parameters
        print (input_shape)
        assert len(input_shape) == 2, 'Unrecognizable input dimensions'
        # Model Architecture
        timeframe, feature_length = input_shape
        # Input LayerRO
        inp = Input(shape=input_shape, name='Features') # timestep, data_dim
        # LSTM layers
        '''
        long_dep = LSTM(7, activation='relu',
                         kernel_initializer='glorot_uniform',
                         name='LSTM1')(inp)
         '''
        flat = Flatten()(inp)

        # Fully-Connected encoding layers
        fc_enc = [Dense(FLAGS.filters[-1],
                        kernel_initializer='glorot_uniform',
                        activation='relu',
                        name='FCEnc1')(flat)]

        for d in range(1, FLAGS.n_fc_layers):
            fc_enc.append(Dense(FLAGS.filters[-1],
                                kernel_initializer='glorot_uniform',
                                activation='relu',
                                name='FCEnc{}'.format(d + 1))(fc_enc[-1]))

        encoded = fc_enc[-1]

        classifier = Dense(1, activation='sigmoid',
                           name='Classifier')(encoded)
        model = Model(inputs=inp, outputs=classifier, name='lstm', classification=True)

    # Loss Functions
    losses = [binary_crossentropy]

    # Metrics
    metrics = [binary_accuracy, 'accuracy']

    # Compilation

    model.compile(optimizer=FLAGS.optimizer,
                  loss=losses,
                  metrics=metrics,
                  lr=FLAGS.learning_rate)
    return model


def build_bilstm_model(input_shape=None, saved_model=None):
    if saved_model:
        model = load_model(saved_model, compile=False)
        model.classification = True
    else:
        pass
    raise NotImplementedError('Bi-directional LSTM To be implemented soon')

def build_imagelstm_model(input_shape=None, saved_model=None):
    if saved_model:
        model = load_model(saved_model, compile=False)
        model.classification = True
    else:
        pass
    raise NotImplementedError('Image To be implemented soon')
