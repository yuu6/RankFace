from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt


def shape_of_array(arr):
	array = np.array(arr)
	return array.shape


def get_label(num):
	with open('./label.csv') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			if row['#Image'] == str(num):
				return float(row['Attractiveness label'])


def load_image_data(filedir):
	picdir = filedir + "Data_Collection/"
	# label = []
	image_data_list = []
	train_image_list = os.listdir(picdir)
	# train_image_list.remove('.DS_Store')
	label = pd.read_pickle(filedir + "/labels.pickle")["rate"]
	# print(label)

	for img in train_image_list:
		url = os.path.join(picdir + img)
		image = cv2.imread(url)
		# plt.imshow(image)  # 显示图片
		# plt.show()
		image = cv2.resize(image, (128, 128))
		image_data_list.append(image)

	img_data = np.array(image_data_list)
	img_data = img_data.astype('float32')
	img_data /= 255
	return img_data, label


def make_network():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same', input_shape=(128, 128, 3)))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(128))  # Dense 是一个全连接层
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))

	return model


def main():
	# train_x, train_y = load_image_data('drive/study/TensorFlow/CNN/RankFace/data/')
	train_x, train_y = load_image_data('./data/')
	print("x的形状是{}，y的形状是{}".format(train_x.shape, train_y.shape))
	print(train_x[0])
	model = make_network()
	model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
	hist = model.fit(train_x, train_y, batch_size=100, epochs=15, verbose=1)
	# plot_model(model, show_shapes=True, to_file=('model.png'))

	# model.evaluate(train_x, train_y)
	model.save('faceRank.h5')


if __name__ == '__main__':
	main()
