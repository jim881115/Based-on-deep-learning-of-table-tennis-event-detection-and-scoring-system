import tensorflow as tf
import keras
from keras.models import *
from keras.layers import *
import numpy as np
import os
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
import Models, LoadBatches
import glob
import csv
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
width = 640
height = 360

modelTN = Models.TrackNet.TrackNet 
m = modelTN(256, input_height=height, input_width=width) 
m.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
m.load_weights("weights/model12p.h5")
tracknet = Model(inputs=m.input, outputs=m.layers[33].output)
"""
images_path = '/home/tt/TableTennis/TrackNet/Dataset'
dirs = glob.glob(images_path + '/Clip*')
dirs.sort()

for index in dirs:
    images = glob.glob(index + '/*.jpg')
    images.sort() 
    for i in range(4, len(images)):

        output_pics_path = os.path.split(images_path)[0]+'/event_data/' + os.path.split(index)[-1] + '/global/'
        if not os.path.exists(output_pics_path):
            os.makedirs(output_pics_path)

        output_pics_path = output_pics_path + os.path.splitext(os.path.split(images[i])[-1])[0]
        if os.path.exists(output_pics_path + '.npz'):
                continue

        print('Global:', images[i])
        X = LoadBatches.getInputArr(images[i], images[i-1], images[i-2], images[i-3], images[i-4], width, height)
        pr = tracknet.predict(np.array([X]))

        np.savez_compressed(output_pics_path, pr[0])
    #break
"""
path = '/home/tt/TableTennis/TrackNet/newdata'
dirs = glob.glob(path + '/Clip*')
dirs.sort()

for index in dirs:
    i = 4
    images = glob.glob(index + '/*.jpg')
    images.sort()

    label_path = index + '/Label.csv'
    with open(label_path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

        next(spamreader, None) #header
        next(spamreader, None) #0000
        next(spamreader, None) #0001
        next(spamreader, None) #0002
        next(spamreader, None) #0003

        for row in spamreader:
            FileName = row[0]

            output_pics_path = os.path.split(path)[0]+'/event_data/' + os.path.split(index)[-1] + '/local/'
            if not os.path.exists(output_pics_path):
                os.makedirs(output_pics_path)
        
            output_pics_path = output_pics_path + FileName.split('.')[-2]
            if os.path.exists(output_pics_path + '.npz'):
                i = i + 1
                continue
                
            output_pics_path_ = os.path.split(path)[0]+'/event_data/' + os.path.split(index)[-1] + '/global/'
            if not os.path.exists(output_pics_path_):
                os.makedirs(output_pics_path_)
        
            output_pics_path_ = output_pics_path_ + FileName.split('.')[-2]
            if os.path.exists(output_pics_path_ + '.npz'):
                i = i + 1
                continue


            vis = int(float(row[1]))
            if vis == 0:
                i = i + 1
                continue
                x = 960
                y = 540
            else:
                x = int(float(row[2]))
                y = int(float(row[3]))
            print('Global:', images[i])
            X_ = LoadBatches.getInputArr(images[i], images[i-1], images[i-2], images[i-3], images[i-4], width, height)    
            print('Local:', images[i])
            X = LoadBatches.getInputArr2(images[i], images[i-1], images[i-2], images[i-3], images[i-4], width, height, x, y)

            i = i + 1
            pr_ = tracknet.predict(np.array([X_]))[0]
            pr = tracknet.predict(np.array([X]))[0]
            
            np.savez_compressed(output_pics_path_, pr_)
            np.savez_compressed(output_pics_path, pr)

            #break
    #break






