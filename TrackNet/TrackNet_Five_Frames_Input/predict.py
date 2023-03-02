import argparse
import Models , LoadBatches
import cv2
import numpy as np
import glob
import os
import sys
import csv

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow.compat.v1.keras.backend import set_session
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.compat.v1.Session(config=config))

pic_number = 0
n_classes = 256
width =  640
height = 360
output_width =  1920
output_height = 1080
GroundTruth = {}

#load TrackNet model
modelTN = Models.TrackNet.TrackNet
m = modelTN(n_classes , input_height=height, input_width=width)
m.compile(loss='categorical_crossentropy', optimizer= 'adadelta', metrics=['accuracy'])
m.load_weights("weights/model12p.h5")


#get TrackNet output height and width
model_output_height = m.outputHeight
model_output_width = m.outputWidth

images_path = '/home/tt/TableTennis/TrackNet/Dataset'
dirs = glob.glob(images_path + '/Clip*')
dirs.sort()


for number in dirs:
    img_path = number+'/'   # /home/tt/TableTennis/TrackNet/Dataset/Clip*
    label_path = img_path + "Label.csv"     # /home/tt/TableTennis/TrackNet/Dataset/Clip*/Label.csv

    #read ground truth from Label.csv
    with open(label_path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(spamreader, None)  # skip the headers
        for row in spamreader:
            if row[2] != '':
                FileName = row[0]
                visibility = int(float(row[1]))
                x = int(row[2])
                y = int(row[3])
                GroundTruth[img_path+FileName] = [visibility,x,y]
            else:
                FileName = row[0]
                GroundTruth[img_path+FileName] = [0,-1,-1]

testing_file_path = "testing_model2.csv"

#predict each images
with open(testing_file_path, 'r') as csvfile:    
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(spamreader, None)  # skip the headers
    for row in spamreader:
        pic_name = row[0]
        pic_number = pic_number + 1
        #load input data
        X = LoadBatches.getInputArr(row[0], row[1], row[2], row[3], row[4], width, height)
        pr = m.predict(np.array([X]))[0]
        pr = pr.reshape((height,  width , 256)).argmax(axis=2)
        pr = pr.astype(np.uint8) 
        heatmap = cv2.resize(pr, (1920, 1080))
        ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT,dp=1,minDist=1,param1=50,param2=2,minRadius=4,maxRadius=8)
        print(len(circles))
        print(pic_name.split("/")[-1])
        print(pic_number, "x:", GroundTruth[pic_name][1], "y:", GroundTruth[pic_name][2])
        if circles is not None:
            #if only one tennis be detected
            if len(circles) == 1:
                x = int(circles[0][0][0])
                y = int(circles[0][0][1])
                print(pic_number, "x:", x, "y:", y)
  
            else:
                print(pic_number, "x: None ", "y: None")
        else:
            print(pic_number, "x: None ", "y: None")
        
        
        #if pic_number > 100:
        #    break
        #input()
"""python3 predict.py --save_weights_path=weights/model1w14p.h5 --test_images_path="test" --output_path="testoutput" --input_height=360 --input_width=640 --output_height=1080 --output_width=1920 --n_classes=256"""
