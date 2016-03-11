from PIL import Image
import numpy as np
import os
import gzip,cPickle
import random
import csv
from sklearn.preprocessing import OneHotEncoder

no_of_folder = 1000                                        """ Number of folders changed to 1000 """

# Function to get labels from business ids
def get_labels(bus_ids):
	labels_ids=[]
	for i in bus_ids:
	    with open('/Users/deepakmuralidharan/Downloads/train.csv', 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		for row in spamreader:
		    if i == row[0].split(',')[0]:
				t=[]
				t.append(int(row[0].split(',')[1]))
				for r in range(1,len(row)):
					t.append(int(row[r]))
				labels_ids.append(t)

	return labels_ids
# Function to get business ids from image name as int(without .jpg)
def get_busi_ids(rand_sel_images):
	bus_ids=[]
	count=1

	for k in range(len(rand_sel_images)):
		i = rand_sel_images[k]
		print (i)
		with open('/Users/deepakmuralidharan/Downloads/train_photo_to_biz_ids.csv', 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			for row in spamreader:
				temp = int(row[0].split(',')[0])
				if i == temp:
					#print i
					#print row[0].split(',')[0]
					bus_ids.append(row[0].split(',')[1])
					break

	return bus_ids






dirs='/Users/deepakmuralidharan/sample_train/'
for k in range(no_of_folder):
	name='sample'+str(k+1)
	path=dirs+name
	sample_images=[]
	if(os.path.isdir(path)):					""" This check is added for existence of folders(can be less than 1000) """
		f_names=os.listdir(path)
		for i in range(len(f_names)):
	#choice=random.choice(f_names)
			choice_num=f_names[i].split('.')
			sample_images.append(int(choice_num[0]))
#print len(rand_sel_images)

		b_IDS=get_busi_ids(sample_images)
#print(b_IDS)

		final_labels=get_labels(b_IDS)
#print (final_labels)

		big_li = []
		for element in final_labels:
			li = [0]*9
			for small_element in element:
				li[small_element] = 1
			big_li.append(li)

		big_li = np.asarray(big_li)
	#print len(final_labels)
#print (big_li[0])
		x_data = []
		for name in f_names:
			img_name = path+'/'+name
			try:
				img = Image.open(img_name)
				img.load()
				data = np.asarray(img, dtype="int32" )
				x_data.append(data)
			except IOError:
				print 'iteration '+str(k)
	#return x_data
		x_data=np.asarray(x_data)
		print x_data.shape
		print big_li.shape
		numpy_file_name='/Users/deepakmuralidharan/pickle_sample_files/sample_data'+str(k+1)
		np.savez(numpy_file_name,X_train = x_data,y_train = big_li)
