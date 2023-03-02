import argparse
import Models, LoadBatches
import queue
import cv2
import numpy as np
import time
import func

from PIL import Image, ImageDraw
from keras.models import *
from keras.layers import *


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow.compat.v1.keras.backend import set_session
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.compat.v1.Session(config=config))

import sys
sys.path.append("/home/tt/TableTennis/tfBase-TTNet")
import demo

def plot_prob(dict):
    import matplotlib.pyplot as plt

    # save figure loss
    frame = np.arange(1, currentFrame+1-4, 1)
    demo_fig = plt.figure()
    plt.plot(frame, dict['Bounce'], label='Bounce prob.')
    plt.plot(frame, dict['Net'], label='Net prob.')
    plt.ylabel('prob.')
    plt.xlabel('frame num')
    plt.legend()
    plt.title("demo")
    demo_fig.savefig('./demo_fig.png')
prob_his = {'Bounce':[], 'Net':[]}

#parameters
##########
currentFrame = 0
max_y, need_L_B, need_R_B, bounce_count, serve ,bounce_in_round, bounce_q = 1080, False, False, 0, False, 0, 0
attack = [False, False]
q = queue.deque()
for i in range(0,17):
    q.appendleft(None)

img_q = queue.deque(maxlen=5)
imq_q_no = queue.deque(maxlen=5)
point = np.zeros((2, 2)).astype(int)
event_q = queue.deque(maxlen=1)
event_q.appendleft(None)
#########

input_video_path =  "/home/tt/TableTennis/TrackNet/TrackNet_Five_Frames_Input/video/demo4.mp4"
output_video_path = input_video_path.split('.')[0] + "_out.mp4"
save_weights_path = "weights/model12p.h5"
n_classes =  256
width , height = 640, 360

#get video fps&video size
video = cv2.VideoCapture(input_video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))



#load TrackNet model
with tf.device('/gpu:1'):
    modelFN = Models.TrackNet.TrackNet
    m = modelFN(n_classes, input_height=height, input_width=width)
    m.compile(loss='categorical_crossentropy', optimizer= 'adadelta', metrics=['accuracy'])
    m.load_weights(save_weights_path)
    feature = Model(inputs=m.input, outputs=m.layers[33].output)
#with tf.device('/gpu:1'):
    TTNET_model = demo.create_model('/home/tt/TableTennis/tfBase-TTNet/weights/train_model_test_weight3.h5')





#頭四張pic跳過
for i in range(4): 
    video.set(1, currentFrame);
    ret, img = video.read()
    
    if not ret: 
        print("Not read video")
        break
    imq_q_no.appendleft(img)
    output_video.write(img)
    currentFrame += 1
    img = cv2.resize(img, (width, height))
    img = img.astype(np.float32)
    img_q.appendleft(img)

while(True):
    #capture frame-by-frame
    video.set(1,currentFrame); 
    ret, img = video.read()
    
    if not ret: #if there dont have any frame in video, break
        break
    imq_q_no.appendleft(img)
    output_img = img
    img = cv2.resize(img, (width, height))
    img = img.astype(np.float32)
    img_q.appendleft(img)
    
    global_input = LoadBatches.get_global_input(img_q, width, height)
 
    pr = m.predict(np.array([global_input]))[0]
    circles = func.predict_ball(pr, width, height, output_width, output_height, n_classes)

    x, y = func.get_ball(circles, q)
    
    global_feature = feature.predict(np.array([global_input]))[0]
    local_input = LoadBatches.local_input(imq_q_no[0], imq_q_no[1], imq_q_no[2], imq_q_no[3], imq_q_no[4], x, y)
    local_feature = feature.predict(np.array([local_input]))[0]
    
    out = [0, 0]
    if x != None:
        global_feature = feature.predict(np.array([global_input]))[0]
        local_input = LoadBatches.local_input(imq_q_no[0], imq_q_no[1], imq_q_no[2], imq_q_no[3], imq_q_no[4], x, y)
        local_feature = feature.predict(np.array([local_input]))[0]
        #pass
        out = demo.run_model(TTNET_model, global_feature, local_feature)
    
    if not serve and x != None:
        max_y = min(max_y, y)
    
    max_y, need_L_B, need_R_B, bounce_count, serve, point, bounce_in_round, bounce_q = func.event_caculation(x, y, max_y, out, need_L_B, need_R_B, bounce_count, serve, point, bounce_q, bounce_in_round, event_q, attack)
    output_img = func.draw_out(q, need_L_B, need_R_B, point, output_img, out, bounce_in_round, event_q, attack)

    if out[0] == -1:
        prob_his['Bounce'].append(0)
        prob_his['Net'].append(0)
    else:
        prob_his['Bounce'].append(out[1])
        prob_his['Net'].append(out[0])

    print(currentFrame, x, y)
    print(out)
    
    output_video.write(output_img)
    
    
    currentFrame += 1

# everything is done, release the video
print("video release")
video.release()
output_video.release()
print("finish")
plot_prob(prob_his)
