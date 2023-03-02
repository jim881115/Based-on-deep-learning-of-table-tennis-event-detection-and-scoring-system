import csv
import tensorflow as tf
import itertools
import numpy as np
from collections import defaultdict

from data_processor import shuffle_csv

training_data_csv_path = "./train_data_file_path.csv" #建立csv檔
testing_data_csv_path = "./test_data_file_path.csv"
# val_file_path = "./TrackNet_Five_Frames_Input/val_model2.csv"

location = ['Clip11', 'Clip13', 'Clip16', 'Clip20', 'Clip25', 'Clip28', 'Clip31', 'Clip34',
            'Clip41', 'Clip42', 'Clip45', 'Clip48', 'Clip50', 'Clip55', 'Clip59', 'Clip62',
            'Clip66', 'Clip67', 'Clip68', 'Clip70', 'Clip74', 'Clip75', 'Clip77', 'Clip78',
            'Clip79', 'Clip81', 'Clip82', 'Clip84', 'Clip87', 'Clip90', 'Clip91', 'Clip92',
            'Clip93', 'Clip94', 'Clip95', 'Clip96', 'Clip97', 'Clip98', 'Clip101']

images_path = '/home/tt/TableTennis/TrackNet/Dataset/'


def load_feature(global_path, local_path):
	local_feature = np.load(local_path)
	global_feature = np.load(global_path)

	local_feature = local_feature.f.arr_0.reshape(512, 45, 80)
	global_feature = global_feature.f.arr_0.reshape(512, 45, 80)

	input_feature = np.concatenate((global_feature, local_feature), axis=0)

	return input_feature


def generator(img_path, batch_size):
	input_feature = []
	output_label = []
	columns = defaultdict(list)
	
	shuffle_csv(img_path)

	with open(img_path) as f:
		reader = csv.reader(f)
		for row in reader:
			for (i, v) in enumerate(row):
				columns[i].append(v)
	zipped = itertools.cycle(zip(columns[0], columns[1], columns[2]))
	
	while True:
		Input = []
		Output = []
		for _ in range(batch_size):
			path, label0, label1 = next(zipped)

			head_path, target_npz = path.split('|')
			
			# last frame
			target_npz = str(int(target_npz.split('.')[0]) - 2).zfill(4) + target_npz[-4:]
			
			global_path = head_path + '/global/' + target_npz
			local_path = head_path + '/local/' + target_npz
			input_feature = load_feature(global_path, local_path)

			Input.append(input_feature)
			
			
			out1 = np.array(float(label0), dtype=np.float32)
			out2 = np.array(float(label1), dtype=np.float32)
			
			out = np.array([out1, out2])
			Output.append(out)
		
		yield np.array(Input), np.array(Output)


if __name__ == '__main__':
	# create_csv(location, training_data_csv_path)

	while True:
		# print(next(generator(training_data_csv_path, 2)))
		# print(next(generator(training_data_csv_path, 2))[0].shape)
		test = next(generator(training_data_csv_path, 1))
		# print(type(test), test.shape)
		print(type(test[0][0]), test[0][0].shape)
		print(type(test[1][0]), test[1][0].shape)

		input()
