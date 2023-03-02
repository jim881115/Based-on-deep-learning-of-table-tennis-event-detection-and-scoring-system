"""
this cell is for testing the CNN work or not.
not for train-test model.
"""

import time
from tqdm import tqdm, trange

from model_utils import create_model
from train_utils import create_optimizer
from data_loader import get_event_infor
import tensorflow as tf

from tensorflow import sigmoid
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, ReLU, Dropout
from tensorflow.keras import Model

from tensorflow.keras.activations import relu, sigmoid

from tensorflow.keras.optimizers import Adam
from keras import optimizers

from losses import Events_Spotting_Loss
from eventNet import eventNet

opt = tf.keras.optimizers.Adam(learning_rate=0.001)

location = ['Clip11', 'Clip13', 'Clip16', 'Clip20', 'Clip25', 'Clip28', 'Clip31', 'Clip34',
            'Clip41', 'Clip42', 'Clip45', 'Clip48', 'Clip50', 'Clip55', 'Clip59', 'Clip62',
            'Clip66', 'Clip67', 'Clip68', 'Clip70', 'Clip74', 'Clip75', 'Clip77', 'Clip78',
            'Clip79', 'Clip81', 'Clip82', 'Clip84', 'Clip87', 'Clip90', 'Clip91', 'Clip92',
            'Clip93', 'Clip94', 'Clip95', 'Clip96', 'Clip97', 'Clip98', 'Clip101']

# cnn model
class ConvBlock_without_Pooling(Model):
    def __init__(self, in_channels):
        super(ConvBlock_without_Pooling, self).__init__()
        # self.conv = Conv2D(in_channels, out_channels, kernel_size=3, strides=1, padding=1)
        # self.conv = Conv2D(out_channels, kernel_size=3, strides=1, padding=1)
        # self.conv = Conv2D(out_channels, (3, 3), kernel_initializer='random_uniform', strides=1, padding='same',
        # data_format='channels_first')

        # input_shape=(1024, 45, 80)
        self.conv = Conv2D(in_channels, 3, kernel_initializer='random_uniform',
                           padding='same', data_format='channels_first')

        # out_channels?
        # self.batchnorm = BatchNormalization(out_channels)
        self.batchnorm = BatchNormalization(epsilon=1e-5)

    def call(self, x):
        # x = self.relu(self.batchnorm(self.conv(x)))
        x = relu(self.batchnorm(self.conv(x)))
        return x


class eventNet(Model):
    def __init__(self):
        super(eventNet, self).__init__()
        # out_channel = 64
        # input_shape = (1, 1024, 45, 80)

        # input_shape=input_shape[1:]
        self.conv1 = Conv2D(64, 1, kernel_initializer='random_uniform',
                           padding='same', data_format='channels_first')
        # out_channels?
        self.batchnorm = BatchNormalization(epsilon=1e-5)
        # self.relu = ReLU()
        self.dropout2d = Dropout(0.5)
        # out_channel = 64
        self.convblock1 = ConvBlock_without_Pooling(in_channels=64)
        self.convblock2 = ConvBlock_without_Pooling(in_channels=64)
        self.fc1 = Dense(units=512)
        self.fc2 = Dense(units=2)
        # self.sigmoid = sigmoid()

        # self._set_inputs(tf.TensorSpec([80, 45], tf.float32, name='inputs'))

        """
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)
        """

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 512, 45, 80], dtype=tf.float32),
        tf.TensorSpec(shape=[1, 512, 45, 80], dtype=tf.float32)])
    def call(self, global_features, local_features):
        input_eventspotting = tf.concat((global_features, local_features), axis=1)

        x = relu(self.batchnorm(self.conv1(global_features)))
        x = self.dropout2d(x)
        x = self.convblock1(x)
        x = self.dropout2d(x)
        x = self.convblock2(x)
        # print(x.shape)
        x = self.dropout2d(x)
        x = tf.reshape(x, (tf.shape(x)[0], -1))
        # print(x.shape)
        x = relu(self.fc1(x))
        out = sigmoid(self.fc2(x))
        return out

    def get_loss(self, global_input, local_input, target_events, custom_loss):
        out = self.call(global_input, local_input)
        loss = custom_loss(out, target_events)
        return loss

    def network_learn(self, global_input, local_input, target_events, custom_loss):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(global_input, local_input, target_events, custom_loss)
            g = tape.gradient(L, self.trainable_variables)
            # print('*********** gradient ************')
            # print(g)
            # print('********** loss ***********')
            # print(L)
            # input()
        # opt.minimize(L, var_list=self.trainable_variables)
        # tf.keras.optimizers.Adam(learning_rate=0.001).minimize()
        opt.apply_gradients(zip(g, self.trainable_variables))
        # tape



    def demo(self, global_input, local_input):
        out = self.call(global_input, local_input)
        return out

# main
if __name__ == '__main__':
    model = eventNet()
    start_time = time.time()
    for epoch in range(10):
        time_slip = time.time() - start_time
        print('>>> Epochs: [{}/10], time: [{}]'.format(epoch + 1, time_slip))
        for pos in location:
            print('\t-->  Dataset: [{}]'.format(pos))
            # load data
            events_infor, events_label = get_event_infor(pos)
            # loss function
            loss = Events_Spotting_Loss()

            for global_features, local_features, a, target_event, *_ in tqdm(events_infor):
                '''
                print('************** global_features *************')
                print(global_features)
                print('************** local_features *************')
                print(local_features)
                input()
                
                print('************** event *************')
                print(target_event)
                '''
                model.network_learn(global_features, local_features, target_event, loss)
    model.save_weights('eventmodel')
