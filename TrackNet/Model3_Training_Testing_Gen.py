#3.Output training data name to cvs file for model 2
import numpy as np
import cv2
import glob
import itertools
import random
import csv
import os

training_file_name = "./TrackNet_Five_Frames_Input/training_model2.csv" #建立csv檔
testing_file_name = "./TrackNet_Five_Frames_Input/testing_model2.csv"
val_file_name = "./TrackNet_Five_Frames_Input/val_model2.csv"
visibility_for_testing = []

images_path = '/home/tt/TableTennis/TrackNet/testdata/'
dirs = glob.glob(images_path+'Clip*')
dirs.sort()
with open(training_file_name,'w') as file:
    for index in dirs:
        #################change the path####################################################
        annos_path = '/home/tt/TableTennis/TrackNet/heatmap_12p/' + os.path.split(index)[-1]+'/'
        images_path = index+'/'
        ####################################################################################
        images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") +  glob.glob(images_path + "*.jpeg")
        images.sort()

        annotations  = glob.glob(annos_path + "*.jpg") + glob.glob(annos_path + "*.png") +  glob.glob(annos_path + "*.jpeg")
        annotations.sort()

        print(index + ":", len(images), len(annotations))
        #check if annotation counts equals to image counts
        assert len(images) == len(annotations)
        for im, seg in zip(images, annotations):
            #print(im, seg)
            #print(im.split('\\')[-1].split(".")[0], seg.split('\\')[-1].split(".")[0])
            assert(im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])

        visibility = {}
        with open(images_path + "Label.csv", 'r') as csvfile:   #row[0] = xxxx.jpg, row[1] = 球可見度(0:不在場, 1:輕易, 2:不易, 3:被遮擋)
                                                                #row[2]row[3] = (x, y), row[4] = 球狀態(0:飛行, 1:命中, 2:彈跳)
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            #skip the headers
            next(spamreader, None)  
            
            for row in spamreader:
                #print(row[0])
                visibility[row[0]] = row[1]
                    
                    
        #output all of images path, 0000.jpg & 0001.jpg cant be used as input, so we have to start from 0004.jpg
        for i in range(4,len(images)): 
                #remove image path, get image name   
                #ex: D/Dateset/Clip1/0056.jpg => 0056.jpg 
                file_name = os.path.split(images[i])[-1]
                #print(images[i], file_name)

                #visibility 3 will not be used for training
                if visibility[file_name] == '3': 
                    visibility_for_testing.append(images[i])

                #check if file image name same as annotation name
                assert(images[i].split('/')[-1].split(".")[0] == annotations[i].split('/')[-1].split(".")[0])

                #write all of images path
                file.write(images[i] + "," + images[i-1] + "," + images[i-2] + ","  + images[i-3] + "," + images[i-4] + "," + annotations[i] + "\n")
                #print(images[i], images[i-1], images[i-2], images[i-3], images[i-4], annotations[i])
                    

file.close()

#read all of images path
lines = open(training_file_name).read().splitlines()

#2w for training,other for testing 
training_images_number = len(lines)
val_images_number = 2000
testing_images_number = len(lines) - training_images_number - val_images_number

print("Total images:", len(lines), "Training images:", training_images_number, "Val images:", val_images_number, "Testing images:", testing_images_number)

#shuffle the images
random.shuffle(lines)
#training images
with open(training_file_name,'w') as training_file:
    training_file.write("img, img1, img2, img3, img4, ann\n")
    #testing images
    with open(testing_file_name,'w') as testing_file:
        testing_file.write("img, img1, img2, img3, img4, ann\n")
        with open(val_file_name,'w') as val_file:
            val_file.write("img, img1, img2, img3, img4, ann\n")
            #write img, img1, img2, ann to csv file
            for i in range(0,len(lines)):
                if lines[i] != "":
                    if training_images_number > 0 :#and lines[i].split(",")[0] not in visibility_for_testing  
                        testing_file.write(lines[i] + "\n")
                        #training_file.write(lines[i] + "\n")
                        training_images_number -= 1
                    """elif val_images_number > 0:
                        val_file.write(lines[i] + "\n")
                        val_images_number -= 1
                    else:
                        testing_file.write(lines[i] + "\n")"""

                    
training_file.close()
testing_file.close()
