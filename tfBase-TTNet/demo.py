import numpy as np
import tensorflow as tf
import gc
import os
from tensorflow import keras

from model import EventNet

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class TTNet:
    def __init__(self, model_path='./train_model_weight3.h5'):
        self.model = EventNet()
        self.model.load_weights(model_path)
    
    def run(self, global_feature, local_feature):
        input_feature = np.concatenate((global_feature, local_feature), axis=0)
        input_feature = np.array([input_feature])

        out = self.model.predict(input_feature)[0]

        del input_feature
        gc.collect()

        if out[0] > 0.9:
            return [1, 0]
        elif out[1] > 0.9:
            return [0, 1]
        else:
            return [0, 0]


def create_model(model_path):
    model = EventNet()
    model.load_weights(model_path)
    return model

def run_model(model, global_feature, local_feature):
    input_feature = np.concatenate((global_feature, local_feature), axis=0)
    input_feature = np.array([input_feature])

    out = model.predict(input_feature)[0]
    del input_feature
    '''
    if out[0] > 0.9:
        return [1, 0]
    elif out[1] > 0.9:
        return [0, 1]
    else:
        return [0, 0]
    '''
    return out

if __name__ == '__main__':
    TTNet_model = create_model('./weights/train_model_weight9.h5')
    global_feature = np.load('../TrackNet/event_data/Clip136/global/0064.npz')
    local_feature = np.load('../TrackNet/event_data/Clip136/local/0064.npz')

    local_feature = local_feature.f.arr_0.reshape(512, 45, 80)
    global_feature = global_feature.f.arr_0.reshape(512, 45, 80)

    out = run_model(TTNet_model, global_feature, local_feature)
    print(out)
