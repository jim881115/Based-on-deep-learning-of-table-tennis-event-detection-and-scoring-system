import glob
import csv
import numpy
import matplotlib.pyplot as plt
from PIL import Image
import os
from os.path import expanduser

size = 20
#create gussian heatmap
def gaussian_kernel(variance):
    x, y = numpy.mgrid[-size:size+1, -size:size+1]  # x = [-20, -20, ...], y = [-20, -19, ...]
    g = numpy.exp(-(x**2+y**2)/float(2*variance))
    return g


#make the Gaussian by calling the function
variance = 12
gaussian_kernel_array = gaussian_kernel(variance)
#rescale the value to 0-255
gaussian_kernel_array =  gaussian_kernel_array * 255/gaussian_kernel_array[int(len(gaussian_kernel_array)/2)][int(len(gaussian_kernel_array)/2)]
#change type as integer
gaussian_kernel_array = gaussian_kernel_array.astype(int)
#print(gaussian_kernel_array[20][12])

#show heatmap
"""plt.imshow(gaussian_kernel_array, cmap=plt.get_cmap('gray'), interpolation='nearest')
plt.colorbar()
plt.show()"""


#create the heatmap as ground truth
images_path = '/home/tt/TableTennis/TrackNet/testdata'
dirs = glob.glob(images_path+'/Clip*')  #抓game1下所有檔案路徑
print(dirs)
for index in dirs:
        #################change the path####################################################
        pics = glob.glob(index + "/*.jpg")
        output_pics_path = os.path.split(images_path)[0]+'/heatmap12p/' + os.path.split(index)[-1]
        print(output_pics_path)
        label_path = index + "/Label.csv"
        ####################################################################################

        #check if the path need to be create
        if not os.path.exists(output_pics_path ):
            os.makedirs(output_pics_path)


        #read csv file
        with open(label_path, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            #skip the headers
            next(spamreader, None) # 略過第一行名稱

            for row in spamreader: #row[0] = xxxx.jpg, row[1] = 球可見度(0:不在場, 1:輕易, 2:不易, 3:被遮擋)
                                   #row[2]row[3] = (x, y), row[4] = 球狀態(0:飛行, 1:命中, 2:彈跳)
                    visibility = int(float(row[1]))
                    FileName = row[0]
                    #if visibility == 0, the heatmap is a black image 
                    if visibility == 0:
                        heatmap = Image.new("RGB", (1920, 1080))
                        pix = heatmap.load()
                        for i in range(1920):
                            for j in range(1080):
                                    pix[i,j] = (0,0,0)
                    else:
                        x = int(float(row[2]))
                        y = int(float(row[3]))

                        #create a black image
                        heatmap = Image.new("RGB", (1920, 1080))
                        pix = heatmap.load()
                        for i in range(1920):
                            for j in range(1080):
                                    pix[i,j] = (0,0,0)

                        #copy the heatmap on it
                        for i in range(-size,size+1):
                            for j in range(-size,size+1):
                                    if x+i<1920 and x+i>=0 and y+j<1080 and y+j>=0 :
                                        temp = gaussian_kernel_array[i+size][j+size]
                                        if temp > 0:
                                            pix[x+i,y+j] = (temp,temp,temp)
                    #save image
                    heatmap.save(output_pics_path + "/" + FileName.split('.')[-2] + ".png", "PNG")
