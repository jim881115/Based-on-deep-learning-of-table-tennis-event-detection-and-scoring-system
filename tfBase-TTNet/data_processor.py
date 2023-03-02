import csv
import itertools
import random
import os
from collections import defaultdict

import numpy as np

training_data_csv_path = "./dataset_path/train_data_file_path.csv" #建立csv檔
testing_data_csv_path = "./dataset_path/test_data_file_path.csv"
prepare_data_csv_path = "./dataset_path/prepare_dataset.csv"
# val_file_path = "./TrackNet_Five_Frames_Input/val_model2.csv"

images_path = '/home/tt/TableTennis/TrackNet/Dataset/'

def test_dataset():
    pos = './data_csv_path_for_test.csv'
    pos_ = './data_csv_path_for_test2.csv'
    with open(pos_, 'w') as file:
        write_list = []
        with open(pos) as f:
            reader = csv.reader(f)
            for row in reader:
                print(row)
                poss, clip = row[0].split('|')
                if row[1] == '0':
                    for i in range(5):
                        clip_num = int(clip.split('.')[0])
                        clip_num = str(clip_num + i).zfill(4)
                        prob = round(int(row[2]) - 0.2 * i, 1)
                        
                        write_list.append(poss + '|' + clip_num + '.npz' 
                        + ',' + str(float(row[1])) + ',' + str(prob))
                
                else:
                    for i in range(5):
                        clip_num = int(clip.split('.')[0])
                        clip_num = str(clip_num + i).zfill(4)
                        prob = round(int(row[1]) - 0.2 * i, 1)
                        
                        write_list.append(poss + '|' + clip_num + '.npz' 
                        + ',' + str(prob) + ',' + str(float(row[2])))

                    # write_list.append( + ',' + row[2] + ',' + row[1])
        for i in write_list:
            file.write(i + '\n')




def error_declation():
    with open('./new_data_file_path.csv', 'w') as file:
        write_list = []
        with open('./train_data_file_path.csv') as f:
            reader = csv.reader(f)
            for row in reader:
                print(row)
                write_list.append(row[0] + ',' + row[2] + ',' + row[1])
                print(row[0] + ',' + row[2] + ',' + row[1])
        for list_ in write_list:
            file.write(list_ + '\n')


def csv_rename():
    cont = 0
    for i in range(138, 165):
        cont += 1
        path = '../TrackNet/event_data/Clip{}'.format(i)
        os.rename(path + '/Label({}).csv'.format(cont), path + '/Label.csv')

def create_csv(target_path):
	with open(target_path, 'w') as file:
		event_label = dict()
		ball_pos = dict()
		for i in range(1, 165):
			try:
				with open('../TrackNet/event_data/' + 'Clip' + str(i) + '/Label.csv') as f:
					reader = csv.reader(f)
					next(reader)
					for row in reader:
						# ball_pos[row[0]] = [row[2], row[3]]
                        # bounce
						if row[4] == '2':
							event_label['../TrackNet/event_data/' + 'Clip' + str(i) + '|' + str(int(row[0][:4])+2).zfill(4) + '.npz'] = [0, 1]
                        # net
						if row[4] == '0':
							event_label['../TrackNet/event_data/' + 'Clip' + str(i) + '|' + str(int(row[0][:4])+2).zfill(4) + '.npz'] = [1, 0]
			except IOError as e:
				continue
			'''
			with open('../TrackNet/event_data/' + pos + '/Label.csv') as f:
				reader = csv.reader(f)
				next(reader)
				for row in reader:
					ball_pos[row[0]] = [row[2], row[3]]
			'''
        # file_path, net, bounce
		for item in event_label.items():
			file.write(item[0] + ',' + str(item[1][0]) + ',' + str(item[1][1]) + '\n')


def shuffle_csv(target_csv_path):
    lines = open(target_csv_path).read().splitlines()
    random.shuffle(lines)
    with open(target_csv_path, 'w') as file:
        for line in lines:
            file.write(line + '\n')


def add_empty_event(target_csv_path):
    lines = open(target_csv_path).read().splitlines()
    with open(target_csv_path, 'w') as file:
        cont = 0
        for line in lines:
            file.write(line + '\n')
            if cont < 8:
                clip = line.split('|')[0]
                target_event = line.split('|')[1]
                empty_event_id = int(target_event.split('.')[0])
                empty_event = str(empty_event_id + 10).zfill(4) + '.' + target_event.split('.')[1]
                empty_event = empty_event.split(',')[0] + ',0,0'

                file.write(clip + '|' + empty_event + '\n')
                cont += 1

if __name__ == '__main__':
    # error_declation()
    # csv_rename()
    # create_csv('./data_csv_path_for_test.csv')
    # create_csv(images_path, training_data_csv_path)
    # test_dataset()
    # shuffle_csv(training_data_csv_path)
    # shuffle_csv(testing_data_csv_path)
    shuffle_csv(prepare_data_csv_path)
    # add_empty_event(testing_data_csv_path)
    # csv_rename()