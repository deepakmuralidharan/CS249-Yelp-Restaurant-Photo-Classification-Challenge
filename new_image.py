from PIL import Image
import numpy as np
import os
import gzip,cPickle
import random
import csv
from sklearn.preprocessing import OneHotEncoder

sample_size=250
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


#########################################################################################################################################

#dirs='/Users/deepakmuralidharan/test_photos/'                       #directory where training images are stored '''
sample_dirs='/Users/deepakmuralidharan/sample_test/'
dirs='/Users/deepakmuralidharan/sample_train/'

for file_sample in os.listdir(dirs):                                  #  """" This will select only the files which ends with .jpg """
       if(file_sample.endswith(".jpg") or file_sample.endswith(".jpeg")):
            f_names.append(file_sample)




for j in range(no_of_folder):
    print(j)
    if len(f_names)>0:                                                    # """  condition check for number of folders to be created less than 1000 """"  
        if(len(f_names)>=sample_size):
            choice=random.sample(xrange(len(f_names)), sample_size)
        else:
            choice=random.sample(xrange(len(f_names)), len(f_names))      #""" for selecting number of images when size is less than the sample_size """
         
        sample_images=[]                  
        for i in choice:
			choice_num=f_names[i].split('.')
			sample_images.append(int(choice_num[0]))


		b_IDS=get_busi_ids(sample_images)


		final_labels=get_labels(b_IDS)


		big_li = []
		for element in final_labels:
			li = [0]*9
			for small_element in element:
				li[small_element] = 1
			big_li.append(li)

		big_li = np.asarray(big_li)
	
		x_data = []
		for i in choice:
			img_name = dirs+f_names[i]
			try:
				img = Image.open(img_name)
				img.load()
				data = np.asarray(img, dtype="int32" )
				x_data.append(data)
			except IOError:
				print 'iteration '+str(j)
	#return x_data
		x_data=np.asarray(x_data)
		print x_data.shape
		print big_li.shape
		numpy_file_name='/Users/deepakmuralidharan/pickle_sample_files/sample_data'+str(j+1)
		np.savez(numpy_file_name,X_train = x_data,y_train = big_li)


        for c in sorted(choice,reverse=True):
            del f_names[c]











