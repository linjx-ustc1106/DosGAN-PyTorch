#-*- coding: utf-8 -*-
import sys
import os
import shutil

def split2train_val(root_dir, train_dir, val_dir):
	try:
		os.mkdir(train_dir)
		os.mkdir(val_dir)
	except:
		print ("train_dir and val_dir have existed!")

	person_names = os.listdir(root_dir)
	index = 0

	size_list = list()

	for person_name in person_names:

		os.mkdir(os.path.join(train_dir, person_name))
		os.mkdir(os.path.join(val_dir, person_name))

		index += 1
		img_names = os.listdir(os.path.join(root_dir, person_name))
		n_face = len(img_names)
		n_test = int(n_face/10)
		n_train = n_face - n_test

		print (len(person_names), str(index), person_name, str(n_train), str(n_test))

		for i in range(len(img_names)):
			img_name = img_names[i]
			source_path = os.path.join(root_dir, person_name, img_name)
			if i < n_train:
				target_path = os.path.join(train_dir, person_name, img_name)
			else:
				target_path = os.path.join(val_dir, person_name, img_name)

			shutil.copy(source_path, target_path)


if __name__ == "__main__":
	root_dir = sys.argv[1]
	train_dir = sys.argv[2]
	val_dir = sys.argv[3]
	split2train_val(root_dir, train_dir, val_dir)
