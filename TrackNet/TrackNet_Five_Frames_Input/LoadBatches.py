import numpy as np
import cv2
import itertools
import csv
from collections import defaultdict
np.set_printoptions(threshold=np.inf)

#get input array
def getInputArr(path ,path1 ,path2, path3, path4, width , height):
    try:
        img = cv2.imread(path, 1)
        img = cv2.resize(img, ( width , height ))
        img = img.astype(np.float32)

        img1 = cv2.imread(path1, 1)
        img1 = cv2.resize(img1, ( width , height ))
        img1 = img1.astype(np.float32)

        img2 = cv2.imread(path2, 1)
        img2 = cv2.resize(img2, ( width , height ))
        img2 = img2.astype(np.float32)
        
        img3 = cv2.imread(path3, 1)
        img3 = cv2.resize(img3, ( width , height ))
        img3 = img3.astype(np.float32)

        img4 = cv2.imread(path4, 1)
        img4 = cv2.resize(img4, ( width , height ))
        img4 = img4.astype(np.float32)

        #combine three imgs to  (width , height, rgb*3)
        imgs =  np.concatenate((img, img1, img2, img3, img4),axis=2)
        #since the odering of TrackNet  is 'channels_first', so we need to change the axis
        imgs = np.rollaxis(imgs, 2, 0)

        return imgs

    except Exception as e:

        print(path , e)

#get input array
def getInputArr2(path ,path1 ,path2, path3, path4, width , height, x, y):
    try:
        if x < 320 or x > 1600:
            if x < 320:
                x = 320
            else:
                x = 1600
        if y < 180 or y > 900:
            if y < 180:
                y = 180
            else:
                y = 900

        img = cv2.imread(path, 1)
        img = img[y-180:y+180, x-320:x+320]
        img = img.astype(np.float32)

        img1 = cv2.imread(path1, 1)
        img1 = img1[y-180:y+180, x-320:x+320] 
        img1 = img1.astype(np.float32)

        img2 = cv2.imread(path2, 1)
        img2 = img2[y-180:y+180, x-320:x+320]
        img2 = img2.astype(np.float32)
        
        img3 = cv2.imread(path3, 1)
        img3 = img3[y-180:y+180, x-320:x+320] 
        img3 = img3.astype(np.float32)

        img4 = cv2.imread(path4, 1)
        img4 = img4[y-180:y+180, x-320:x+320] 
        img4 = img4.astype(np.float32)

        #combine three imgs to  (width , height, rgb*3)
        imgs =  np.concatenate((img, img1, img2, img3, img4),axis=2)
        #since the odering of TrackNet  is 'channels_first', so we need to change the axis
        imgs = np.rollaxis(imgs, 2, 0)

        return imgs

    except Exception as e:

        print(path , e)


def get_global_input(img_q, width, height):
    try:
        imgs =  np.concatenate((img_q[0], img_q[1], img_q[2], img_q[3], img_q[4]),axis=2)
        imgs = np.rollaxis(imgs, 2, 0)
        return imgs

    except Exception as e:
        print("Get global_input error")
        
def local_input(img ,img1 ,img2, img3, img4, x, y):
    try:
        if x == None:
            x, y = 960, 540
        if x < 320 or x > 1600:
            if x < 320:
                x = 320
            else:
                x = 1600
        if y < 180 or y > 900:
            if y < 180:
                y = 180
            else:
                y = 900 

        img = img[y-180:y+180, x-320:x+320]
        img = img.astype(np.float32)

        img1 = img1[y-180:y+180, x-320:x+320] 
        img1 = img1.astype(np.float32)

        img2 = img2[y-180:y+180, x-320:x+320]
        img2 = img2.astype(np.float32)
        
        img3 = img3[y-180:y+180, x-320:x+320] 
        img3 = img3.astype(np.float32)

        img4 = img4[y-180:y+180, x-320:x+320] 
        img4 = img4.astype(np.float32)
        
        imgs =  np.concatenate((img, img1, img2, img3, img4),axis=2)
        imgs = np.rollaxis(imgs, 2, 0)
        return imgs

    except Exception as e:
        print("Get local_input error")
        
        
#get output array
def getOutputArr(path , nClasses ,  width , height):

    seg_labels = np.zeros((  height , width  , nClasses ))
    try:
        img = cv2.imread(path, 1)
        img = cv2.resize(img, ( width , height ))
        img = img[:, : , 0]

        for c in range(nClasses):
            seg_labels[: , : , c ] = (img == c ).astype(int)

    except Exception as e:
        print(e)
        
    seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
    return seg_labels



#read input data and output data
def InputOutputGenerator(images_path,  batch_size,  n_classes , input_height , input_width , output_height , output_width):
    #read csv file to 'zipped'
    columns = defaultdict(list)
    with open(images_path) as f:
        reader = csv.reader(f)
        next(reader) #跳過標題
        for row in reader:
            for (i,v) in enumerate(row): #i = 0~3 v = row[0]~row[3]
                #print(i, v)
                columns[i].append(v) #colums[0] = last_pic
    zipped = itertools.cycle(zip(columns[0], columns[1], columns[2], columns[3], columns[4], columns[5])) #zip:((一列), (一列), ...)

    while True:
        Input = []
        Output = []
        #read input&output for each batch
        for _ in range(batch_size) :
            path, path1, path2, path3, path4, anno = next(zipped)
            #print()
            #print(path, path1, path2, path3, path4, anno, sep="\n")
            Input.append(getInputArr(path, path1, path2, path3, path4, input_width, input_height))
            Output.append(getOutputArr(anno, n_classes, output_width, output_height))
            #print(np.shape(Input), np.shape(Output))
        #return input&output
        #input()
        yield np.array(Input), np.array(Output)
        del Input, Output

