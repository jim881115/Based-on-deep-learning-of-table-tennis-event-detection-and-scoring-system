import time
import tqdm
import random
import os

from keras.models import *
from tensorflow.keras import Model
from keras.layers import *
from keras import metrics
import tensorflow as tf
import numpy as np

from LoadBatches import generator
from model import EventNet, loss_func
from data_processor import shuffle_csv

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

all_data_csv_path = "./dataset_path/all_data_file_path.csv"
training_data_csv_path = "./dataset_path/train_data_file_path.csv"
testing_data_csv_path = "./dataset_path/test_data_file_path.csv"

pred_training_data_csv_path = "./dataset_path/prepare_dataset.csv"
pred_testing_data_csv_path = "./dataset_path/pred_test_data_file.csv"


def plot_loss_acc(model_name, dict):
    import matplotlib.pyplot as plt

    # save figure loss
    epochs = np.arange(1, len(dict['loss'])+1, 1)
    loss_fig = plt.figure()
    # plt.style.use('ggplot')
    plt.plot(epochs, dict['loss'], label='train_loss')
    plt.plot(epochs, dict['val_loss'], label='val_loss')
    plt.ylabel('loss', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.legend()
    plt.title('loss', fontsize=20)
    loss_fig.savefig('./figure/' + model_name + '_loss_fig.png')

    # plt.close()
    # save figure acc
    acc_fig = plt.figure()
    # plt.style.use('ggplot')
    plt.plot(epochs, dict['accuracy'], label='train_acc')
    plt.plot(epochs, dict['val_accuracy'], label='val_acc')
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('accuracy', fontsize=20)
    plt.legend()
    plt.title('accuracy', fontsize=20)
    plt.grid(color='w', linestyle='dotted', linewidth=1)
    acc_fig.savefig('./figure/' + model_name + '_acc_fig.png')


def train(optimizer, epoch, batch_size, step_per_epochs, model_name):
    model = EventNet()
    
    model.compile(loss=loss_func, optimizer=optimizer, 
                  # metrics=['accuracy'])
                  metrics=metrics.BinaryAccuracy(threshold=0.9))

    h = {'loss':[], 'accuracy':[], 'val_loss':[], 'val_accuracy':[]}
    print('--model variable infor--\n' +
          '| model_name: {}\n'.format(model_name) + 
          '| epoch: {}\n'.format(epoch, '|') +
          '| batch_size: {}\n'.format(batch_size, '|')  +
          '| step_per_epochs: {}\n'.format(step_per_epochs, '|') +
          '| lr: {}\n'.format(lr, '|') +
          '-------------------------')

    for ep in range(epoch):
        train_generator = generator(training_data_csv_path, batch_size)
        test_generator = generator(testing_data_csv_path, batch_size)

        print(">>> Epoch :", ep+1, '/', epoch)
        his = model.fit_generator(train_generator,
                                validation_data=test_generator,
                                validation_steps=192, 
                                steps_per_epoch=step_per_epochs
                                )
        h['loss'].append(his.history['loss'][0])
        h['accuracy'].append(his.history['binary_accuracy'][0])
        h['val_loss'].append(his.history['val_loss'][0])
        h['val_accuracy'].append(his.history['val_binary_accuracy'][0])
        if ep > 20:
            model.save_weights('./weights/' + model_name + '.h5')

    plot_loss_acc(model_name, h)
    # return h


if __name__ == '__main__':
    # tf.get_logger().setLevel('WARNING')

    # default var
    epoch = 150
    batch_size = 2
    step_per_epochs = 1701
    lr = 1e-6 # adam: 1e-6 adadelta: 1e-2
    model_name = 'org_TTNet_weight_1'
    with tf.device('/gpu:0'):
        optimizer = tf.keras.optimizers.Adam(lr=lr)
        train(optimizer, epoch, batch_size, step_per_epochs, model_name)
        # plot_loss_acc(model_name, his)
    