import argparse
import Models , LoadBatches
from keras import optimizers
from keras.utils import plot_model
import matplotlib.pyplot as plt

import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from tensorflow.compat.v1.keras.backend import set_session
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.compat.v1.Session(config=config))

#parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--training_images_name", type = str  )
parser.add_argument("--n_classes", type=int )
parser.add_argument("--input_height", type=int , default = 360  )
parser.add_argument("--input_width", type=int , default = 640 )
parser.add_argument("--epochs", type = int, default = 1000 )
parser.add_argument("--batch_size", type = int, default = 2 )
parser.add_argument("--step_per_epochs", type = int, default = 200 )
val_images_name = "val_model2.csv"

args = parser.parse_args()
training_images_name = args.training_images_name
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
save_weights_path = args.save_weights_path
epochs = args.epochs
#load_weights = args.load_weights
step_per_epochs = args.step_per_epochs
optimizer_name = optimizers.Adadelta(lr=1.0)

#load TrackNet model
modelTN = Models.TrackNet.TrackNet
m = modelTN(n_classes , input_height=input_height, input_width=input_width)
m.compile(loss='categorical_crossentropy', optimizer= optimizer_name, metrics=['accuracy'])

#show TrackNet details, save it as TrackNet.png
#plot_model(m , show_shapes=True , to_file='TrackNet.png')

#get TrackNet output height and width
model_output_height = m.outputHeight
model_output_width = m.outputWidth

#creat input data and output data
Generator = LoadBatches.InputOutputGenerator(training_images_name, train_batch_size,  n_classes , input_height , input_width , model_output_height , model_output_width)
val_Generator = LoadBatches.InputOutputGenerator(val_images_name, 1, n_classes, input_height, input_width, model_output_height, model_output_width)


def show_train_history(train_h, acc, loss):
    plt.plot(train_h[acc])
    plt.plot(train_h[loss])
    plt.title('Train History')
    #plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['acc', 'loss'], loc='upper left')
    plt.savefig('train')
    plt.show()
    plt.close()

#start to train the model, and save weights per 50 epochs  
h = {'loss':[], 'accuracy':[], 'val_loss':[], 'val_accuracy':[]}
for ep in range(1, epochs+1):
	print("Epoch :", str(ep) + "/" + str(epochs))
	#CallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0)
	his = m.fit_generator(Generator, step_per_epochs, validation_data=val_Generator, validation_steps=10)
	print()
	h['loss'].append(his.history['loss'][0])
	h['accuracy'].append(his.history['accuracy'][0])
	h['val_loss'].append(his.history['val_loss'][0])
	h['val_accuracy'].append(his.history['val_accuracy'][0])

	#if ep % 5 == 0:
		#m.save_weights(save_weights_path + "_12p_test.h5")
		#show_train_history(h, 'accuracy', 'val_accuracy')

#python3 train.py --save_weights_path=weights/model --training_images_name="training_model2.csv" --epochs=2 --n_classes=256 --input_height=360 --input_width=640 --step_per_epochs=200 --batch_size=2
print(h)


