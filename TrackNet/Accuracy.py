#Check Accuracy
import glob
import csv
import cv2
import numpy
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
from os.path import expanduser
from TrackNet_Five_Frames_Input import Models, LoadBatches
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
width = 640
height = 360

modelTN = Models.TrackNet.TrackNet 
m = modelTN(256, input_height=height, input_width=width) 
m.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
m.load_weights("TrackNet_Five_Frames_Input/weights/model12p.h5")


x_array = []
c_array = []
PE_larger_than_5  = []
True_Positive = [0,0,0,0]
False_Positive = [0,0,0,0]
Negative = [0,0,0,0]
pic_number = 0
statistics  = []
GroundTruth = {}

images_path = '/home/tt/TableTennis/TrackNet/testdata'
dirs = glob.glob(images_path + '/Clip*')
dirs.sort()
for number in dirs:
    img_path = number+'/'   # /home/tt/TableTennis/TrackNet/testdata/Clip*
    label_path = img_path + "Label.csv"     # /home/tt/TableTennis/TrackNet/testdata/Clip*/Label.csv

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



testing_file_path = "./TrackNet_Five_Frames_Input/testing_model2.csv"
occluded = []

#predict all of the testing image, and check True_Positive, False_Positive, Negative
with open(testing_file_path, 'r') as csvfile:    
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(spamreader, None)  # skip the headers
    for row in spamreader:
        pic_name = row[0]
        pic_number = pic_number + 1

        X = LoadBatches.getInputArr(row[0], row[1], row[2], row[3], row[4], width, height)
        pr = m.predict(np.array([X]))[0]
        pr = pr.reshape((height,  width , 256)).argmax(axis=2)
        pr = pr.astype(np.uint8) 
        heatmap = cv2.resize(pr, (1920, 1080))
        ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT,dp=1,minDist=1,param1=50,param2=2,minRadius=4,maxRadius=8)

        x, y = None, None
        #if there is no ball in ground truth, any prediction should be False Positive, else be Negative
        if GroundTruth[pic_name][1] == -1 and GroundTruth[pic_name][2] == -1:
            if circles is not None:
                    False_Positive[GroundTruth[pic_name][0]] += 1

            else:
                    Negative[GroundTruth[pic_name][0]] += 1

        #the ground truth has x, y be labeled
        else:
            if circles is not None:
                    #if there has only one circle be predicted
                    if len(circles) == 1:
                        x = int(circles[0][0][0])
                        y = int(circles[0][0][1])
                        x2 = pow(GroundTruth[pic_name][1]-x,2)
                        y2 = pow(GroundTruth[pic_name][2]-y,2)
                        
                        #In order to draw the plot, save all of distance in statistics 
                        s = math.ceil(math.ceil(pow(x2+y2,0.5) - 1) / 1.5)
                        statistics.append(s)
                        print(s)
                        if GroundTruth[pic_name][0] == 3:
                            occluded.append(pic_name)
                        #check if distance > 6
                        if pow(x2+y2,0.5) > 9: 
                            False_Positive[GroundTruth[pic_name][0]] += 1

                        else:
                            True_Positive[GroundTruth[pic_name][0]] += 1

                    #if there has more than one circle be predicted, the prediction will be seen as Nagative
                    else:
                        Negative[GroundTruth[pic_name][0]] += 1

            else:
                    Negative[GroundTruth[pic_name][0]] += 1
        print("Pictures = ", pic_number)
        print("G x:", GroundTruth[pic_name][1], "y:", GroundTruth[pic_name][2])
        print("P x:", x, "y:", y)
        #if pic_number > 300:
            #break
        #input()


print("Pictures = ", pic_number)
print(True_Positive[0], True_Positive[1], True_Positive[2], True_Positive[3])
print(False_Positive[0], False_Positive[1], False_Positive[2], False_Positive[3])
print(Negative[0], Negative[1], Negative[2], Negative[3])
TP = True_Positive[1] + True_Positive[2] + True_Positive[3]

print("Precision:", TP / (0.0 + TP + False_Positive[0] + False_Positive[1] + False_Positive[2] + False_Positive[3]))
print("Recall:", TP / (0.0 + TP + Negative[1] + Negative[2] + Negative[3] + False_Positive[1] + False_Positive[2] + False_Positive[3]))

x = []
y = []
c = []
count = 0
others = 0
for i in range(0,7):
    x.append(i)
    y.append(0)
    
for s in statistics:
    
    if s - float(int(s)) == 0 and s<=6:
        y[int(s)] +=1
        count +=1
    elif s<=6:
        y[int(s)] +=1
        count +=1
    else:
        count +=1
        others += 1
print("Mean:", np.mean(statistics), "Variance", np.var(statistics))


for i in range(0,7):
    c.append( round(y[i]/(count+0.0), 4) )
    print(str(i),y[i], y[i]/(count+0.0))
print("Others:", others, others/(count+0.0))
PE_larger_than_5 = (round(others/(count+0.0)*100, 2))
x_array.append(x)
c_array.append(c)

fig, ax = plt.subplots()
axes = plt.gca()
axes.set_ylim([0,0.4])
plt.plot(x,c,color='green', label='Model, ' + str(PE_larger_than_5) + '% PE > 6')
plt.scatter(x,c,color='blue')
for i, txt in enumerate(c):
    ax.annotate(str(round(txt*100, 2))+"%", (x[i],c[i]),weight="bold", horizontalalignment = "center")
plt.ylabel('Percentage',weight="bold")
plt.xlabel('Positioning Error (PE) in Pixel',weight="bold")
plt.legend()
plt.show()
fig.savefig('Model' + '.png', dpi=2000)
print(occluded)
"""
fig, ax = plt.subplots()
axes = plt.gca()
axes.set_ylim([0,0.4])
    
plt.scatter(x_array[1],c_array[1],color='chocolate')
plt.plot(x_array[1],c_array[1],color='red', label='Model II, ' + str(PE_larger_than_5[1]) + '% PE > 5', linestyle="--")
for i, txt in enumerate(c_array[1]):
    ax.annotate(str(txt*100)+"%", (x_array[1][i],c_array[1][i]),weight="bold", horizontalalignment = "center")
    
plt.ylabel('Percentage',weight="bold")
plt.xlabel('Positioning Error (PE) in Pixel',weight="bold")
plt.legend()
plt.show()
fig.savefig('Model1&2.png', dpi=2000)
"""
