import glob
import os
import shutil
import cv2
import matplotlib.pyplot as plt
import csv

images_path = '/home/tt/TableTennis/TrackNet/Dataset/'
dirs = glob.glob(images_path+'Clip*')
dirs.sort()
sum = 0
print('Clip: ', len(dirs))
for index in dirs:

	images_path = index+'/'
	images = glob.glob(images_path + "*.jpg")
	images.sort()

	images_Gpath = '/home/tt/TableTennis/TrackNet/event_data/' + os.path.split(index)[-1]+'/global'
	imagesG = glob.glob(images_Gpath + "/*.npz")
	imagesG.sort()


	images_Lpath = '/home/tt/TableTennis/TrackNet/event_data/' + os.path.split(index)[-1]+'/local'
	imagesL = glob.glob(images_Lpath + "/*.npz")
	imagesL.sort()

	print(index + ":", len(images))
	#check if annotation counts equals to image counts
	print("Data", "Global", "Local")
	print(len(images) - 4, len(imagesG), len(imagesL),sep="    ")
	assert((len(images) - 4) == len(imagesG) ==  len(imagesL))
	sum += len(images)

print('sum: ', sum)


