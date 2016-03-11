from PIL import Image
import numpy as np
import os
import gzip,cPickle
import random
import csv
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from sklearn import cross_validation
from extract import create_graph, iterate_mini_batches, batch_pool3_features
from datetime import datetime
import matplotlib.pyplot as plt
from tsne import tsne
import seaborn as sns
import pandas as pd

""" script to generate CNN Codes directly from the training images"""
sample_size=250                             # size for randomly selecting images from all the training images
no_of_folder = 1000                         # max number of numpy files(with CNN codes)


def serialize_cifar_pool3(X,filename):                                          # to generate the CNN codes from numpy array of images
    print 'About to generate file: %s' % filename
    sess = tf.InteractiveSession()
    X_pool3 = batch_pool3_features(sess,X)
    np.save(filename,X_pool3)


def serialize_data(X_train,y_train,k):
        #X_test = X_train
        #y_test = y_train
        serialize_cifar_pool3(X_train, 'X_train'+str(k))
        #serialize_cifar_pool3(X_test, 'X_test'+str(k))
        np.save('y_train'+str(k),y_train)
        #np.save('y_test'+str(k),y_test)



def get_labels(bus_ids):                                        # Function to get labels from business ids
	labels_ids=[]
	for i in bus_ids:
	    with open('train.csv', 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		for row in spamreader:
		    if i == row[0].split(',')[0]:
				t=[]
				t.append(int(row[0].split(',')[1]))
				for r in range(1,len(row)):
					t.append(int(row[r]))
				labels_ids.append(t)

	return labels_ids

def get_busi_ids(rand_sel_images):                        # Function to get business ids from image_id
	bus_ids=[]
	count=1

	for k in range(len(rand_sel_images)):
		i = rand_sel_images[k]
		print (i)
		with open('train_photo_to_biz_ids.csv', 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			for row in spamreader:
				temp = int(row[0].split(',')[0])
				if i == temp:
					#print i
					#print row[0].split(',')[0]
					bus_ids.append(row[0].split(',')[1])
					break

	return bus_ids


#########################################################################################################################################



dirs='/Users/ndolocal/Documents/CS249-Big-Data-Analytics-master/train_photos/'     # directory where training images are stored
f_names = []
for file_sample in os.listdir(dirs):                                               # select files which ends with .jpg or .jpeg
       if(file_sample.endswith(".jpg") or file_sample.endswith(".jpeg")):
            f_names.append(file_sample)




for j in range(no_of_folder):
    print(j)
    if len(f_names)>0:                                                    # condition check for number of folders to be created less than 1000
        if(len(f_names)>=sample_size):
            choice=random.sample(xrange(len(f_names)), sample_size)
        else:
            choice=random.sample(xrange(len(f_names)), len(f_names))      # randomly selecting "sample_size"(250) images

        sample_images=[]
        for i in choice:
			choice_num=f_names[i].split('.')
			sample_images.append(int(choice_num[0]))

        b_IDS=get_busi_ids(sample_images)                                  # obtain business ids for selected images


        final_labels=get_labels(b_IDS)                                      # obtain labels for selected images from business ids computed above
        big_li_list = []
        for element in final_labels:                                        # making one-hot encoded vector for the labels eg. if labels for an image are 2,3,5 the vector will be [0,0,1,1,0,1,0,0,0]]
            li = [0]*9
            for small_element in element:
                li[small_element] = 1
            big_li_list.append(li)

        big_li = np.asarray(big_li_list)

        x_data = []
        for i in choice:
			img_name = dirs+f_names[i]
			try:
				img = Image.open(img_name)                                  # reading a 224x224x3 image and storing as numpy array
				img.load()
				data = np.asarray(img, dtype="int32" )
				x_data.append(data)
			except IOError:
				print 'iteration '+str(j)
	#return x_data
        x_data=np.asarray(x_data)
        print x_data.shape
        print big_li.shape
        #numpy_file_name='/Users/ndolocal/Documents/CS249-Big-Data-Analytics-master/pickle_sample_files/sample_data'+str(j+1)
        #np.savez(numpy_file_name,X_train = x_data,y_train = big_li)
        serialize_data(x_data,big_li,j+1)                                     # function to generate and save CNN codes

        for c in sorted(choice,reverse=True):                                   # deleting the previously used images to avoid duplicate images
            del f_names[c]
