# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_history(path):
    '''
    if isinstance(model, str):
        print (model)
        with np.load(model) as f:
            train_err_mem = f['train_loss']
            valid_err_mem = f['val_err_mem']
    else:
        train_err_mem = model.train_loss
        valid_err_mem = model.val_err_mem
    '''
    f = np.load('/Users/jonathankhin/Documents/Academic/MIT_II/9.50/prelegans/lstm_output/models/None/[64]_[30]_20_1_lstm_2.history.npz')
    train_loss = f['loss']
    val_loss = f['val_loss']
    plt.figure(3)
    plt.plot(train_loss, label='Train loss')
    plt.plot(val_loss, label='Valid loss')
    plt.legend(loc="best")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training history')
    plt.show()

def plot_acc_history(path):
    '''
    if isinstance(model, str):
        print (model)
        with np.load(model) as f:
            train_err_mem = f['train_loss']
            valid_err_mem = f['val_err_mem']
    else:
        train_err_mem = model.train_loss
        valid_err_mem = model.val_err_mem
    '''
    f = np.load('/Users/jonathankhin/Documents/Academic/MIT_II/9.50/prelegans/lstm_output/models/None/[64]_[30]_20_1_lstm_2.history.npz')
    print (f.keys())
    train_acc = f['binary_accuracy']
    val_acc = f['val_binary_accuracy']
    plt.figure(3)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Valid Accuracy')
    plt.legend(loc="best")

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training history')
    plt.show()

def plot_auroc_history(path):
    raise NotImplementedError()

def plot_lr_schedule(path):
    f = np.load('/Users/jonathankhin/Documents/Academic/MIT_II/9.50/prelegans/lstm_output/models/N2_50/opt/sgd_[64]_[30]_20_1_lstm_2_50.history.npz')
    sgd_lr = f['lr']
    plt.figure(3)
    plt.plot(sgd_lr, label='SGD')
    # plt.plot(val_acc, label='Valid Accuracy')
    plt.legend(loc="best")

    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Adjustment')
    plt.show()

def plot_loss_opt(path):
    f = np.load('/Users/jonathankhin/Documents/Academic/MIT_II/9.50/prelegans/lstm_output/models/N2_50/opt/sgd_[64]_[30]_20_1_lstm_2_50.history.npz')
    g = np.load('/Users/jonathankhin/Documents/Academic/MIT_II/9.50/prelegans/lstm_output/models/N2_50/opt/nadam_[64]_[30]_20_1_lstm_2_50.history.npz')
    h = np.load('/Users/jonathankhin/Documents/Academic/MIT_II/9.50/prelegans/lstm_output/models/N2_50/opt/adagrad_[64]_[30]_20_1_lstm_2_50.history.npz')
    i = np.load('/Users/jonathankhin/Documents/Academic/MIT_II/9.50/prelegans/lstm_output/models/N2_50/opt/adadelta_[64]_[30]_20_1_lstm_2_50.history.npz')
    j = np.load('/Users/jonathankhin/Documents/Academic/MIT_II/9.50/prelegans/lstm_output/models/N2_50/opt/rms_[64]_[30]_20_1_lstm_2_50.history.npz')

    sgd_lr = f['loss']
    nadam_lr = g['loss']
    adagrad_lr = h['loss']
    adadelta_lr = i['loss']
    rms_lr = j['loss']
    plt.figure(3)
    plt.plot(sgd_lr, label='sgd')
    plt.plot(nadam_lr, label='nadam')
    plt.plot(adagrad_lr, label='adagrad')
    plt.plot(rms_lr, label='rms')
    plt.plot(adadelta_lr, label='adadelta')
    # plt.plot(val_acc, label='Valid Accuracy')
    plt.legend(loc="best")

    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Loss Change on Optimizer')
    plt.show()


if __name__== '__main__':
    # plot_loss_history('a')
    # plot_acc_history('a')
    # plot_lr_schedule('a')
    plot_loss_opt('a')
