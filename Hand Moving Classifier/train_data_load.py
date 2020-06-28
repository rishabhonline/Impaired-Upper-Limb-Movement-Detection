import os
from os.path import join
import numpy as np
import json 
from tqdm import tqdm


TRAIN_DIR = 'C:/Users/hp/Documents/qub/TRAIN'

def get_hand_coordinates(DIR, frame_count, debug=False, tolerance=0.3):
	hand = np.zeros((frame_count, 84)) # 0-41:Right, 42-83: Left

	KEYPOINTS = ['hand_right_keypoints_2d', 'hand_left_keypoints_2d']

	for file_index, file in enumerate(os.listdir(DIR)):
		if file_index == frame_count:
			break
	
		dir = join(DIR, file)
		data = json.load(open(dir, 'r'))
		try:
			people = data['people'][0]
		except:
			if debug:
				print('No person detected in {} frame of {} file'.format(file_index, file))
			continue

		pointer = [0,42]
		for index, kp in enumerate(KEYPOINTS):
			keypoint =  people[kp]
			# print(len(keypoint))

			for i in range(len(keypoint)-1):
				ele = keypoint[i]

				if(i%3==0):
					hand[file_index, pointer[index]] = ele
					pointer[index] +=1
					hand[file_index, pointer[index]] = keypoint[i+1]
					pointer[index] +=1

	return hand

def batch_generator(direc, num_example, frames=100, debug=False):

	X = np.zeros((num_example, frames, 84))

	index = 0
	for e in tqdm(direc):
		for folder in direc.get(e):
			if(debug):
				print('Reading from {} folder in {}'.format(folder, e))

			X[index, :, :] = get_hand_coordinates(e + '/' + folder, frames, debug=debug)
			index +=1


	return X

def preprocess(debug=False):
	LABEL_DIR = next(os.walk(TRAIN_DIR))[1]

	LABEL_COUNT = np.zeros(len(LABEL_DIR))

	direc = {}

	for i, label in enumerate(LABEL_DIR):
		LABEL_COUNT[i] = len(next(os.walk(join(TRAIN_DIR, label)))[1])
		direc[join(TRAIN_DIR, label)] = next(os.walk(join(TRAIN_DIR, label)))[1]

	Y = []
	for i, c in enumerate(LABEL_COUNT):

		Y.append(int(LABEL_DIR[i]) * np.ones(int(c)))

	Y = np.asarray(Y)
	Y = Y.reshape((1, Y.size))[0]
	

	num_example = len(Y)
	
	if(debug):
		print('There are {} training examples'.format(num_example))

	X = batch_generator(direc, num_example, frames=100, debug=debug)
	return [X,Y]

def main():
	[X, Y] = preprocess(debug=True)

	print(X.shape)
	print(Y.shape)

	print(X[3,0,:])

if __name__ == '__main__':
	main()
