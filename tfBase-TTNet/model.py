from keras.models import *
from tensorflow.keras import Model
from keras.layers import *
from keras import metrics
import tensorflow as tf
import numpy as np
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# test_weight 3
def EventNet():
    input_features = Input(shape=(1024, 45, 80), dtype=tf.float32)

    x = Conv2D(512, (3, 3), padding='same', data_format='channels_first', activation='relu')(input_features)
    x = BatchNormalization(epsilon=1e-5)(x)

    # layer2
    x = Conv2D(512, (3, 3), padding='same', data_format='channels_first', activation='relu')(x)
    x = BatchNormalization(epsilon=1e-5)(x)

    # layer3
    x = Conv2D(512, (3, 3), padding='same', data_format='channels_first', activation='relu')(x)
    x = BatchNormalization(epsilon=1e-5)(x)

    # layer1
    x = Conv2D(64, (3, 3), padding='same', data_format='channels_first', activation='relu')(x)
    x = BatchNormalization(epsilon=1e-5)(x)

    # layer2
    x = Conv2D(64, (3, 3), padding='same', data_format='channels_first', activation='relu')(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    
    # layer3
    x = Conv2D(64, (3, 3), padding='same', data_format='channels_first', activation='relu')(x)
    x = BatchNormalization(epsilon=1e-5)(x)

    # Flatten and Linner
    x = Flatten(data_format='channels_first')(x)
    x = Dense(units=2048, activation='relu')(x)
    x = Dense(units=512, activation='relu')(x)
    out = Dense(units=2, activation='sigmoid')(x)

    model = Model(input_features, out)
    model.summary()
    return model

def loss_func(pred_events, target_events, preweight=np.array([3, 1], dtype=np.float32), epsilon=np.float32(1e-9)):
    preweight = preweight / np.sum(preweight)
    loss = - tf.math.reduce_mean(preweight * (target_events * tf.math.log(pred_events + epsilon) + (1. - target_events) * tf.math.log(1 - pred_events + epsilon)))
    return loss