# coding=utf-8
import numpy as np

############################
# -------- Classes ------- #
############################

class Handle(object):
    """ Handles names for loading and saving different models.
    """

    def __init__(self,
                 epochs=None,
                 filters=None,
                 filter_length=None,
                 model=None,
                 ftype=None,
                 infile=None,
                 n_conv_layers=None,
                 n_fc_layers=None,
                 **kwargs):
        self.epochs = epochs
        self.filters = filters
        self.filter_size = filter_length

        self.model = model
        self.ftype = ftype
        self.data_id = infile.split('/')[-1].split('.')[0] if infile else None

        self.n_convs = n_conv_layers
        self.n_fc = n_fc_layers

        self.filename = str(self).split('/')[-1]

    def __str__(self):
        return '{0}/{1}_{2}_{3}_{4}_{5}_{6}.{7}'.format(self.data_id,
                                                        self.filters,
                                                        self.filter_size,
                                                        self.epochs,
                                                        self.n_convs,
                                                        self.model,
                                                        self.n_fc,
                                                        self.ftype)

    def __repr__(self):
        return '{0}/{1}_{2}_{3}_{4}_{5}_{6}.{7}'.format(self.data_id,
                                                        self.filters,
                                                        self.filter_size,
                                                        self.epochs,
                                                        self.n_convs,
                                                        self.model,
                                                        self.n_fc,
                                                        self.ftype)


    def __add__(self, other):
        return str(self) + other
    
    def __radd__(self, other):
        return other + str(self)
    '''
    @classmethod
    def from_filename(cls, filename):
        try:
            basename, ftype, __ = filename.split('.')
        except ValueError:
            basename, ftype = filename.split('.')
        dataset = basename.split('/')[-2]

        info = basename.split('/')[-1]

        filters, filter_size, epochs, conv, fc = map(eval, info.split('_')[:5])

        model = info.split('_')[-1]

        obj = cls(epochs=epochs, filters=filters, filter_length=filter_size, conv=conv, fc=fc,
                  data_id=dataset, model=model, ftype=ftype)

        return obj
    '''
