from PIL import Image
import numpy as np
import os
import gzip,cPickle
import random
import csv
from sklearn.preprocessing import OneHotEncoder

dirs='/Users/deepakmuralidharan/train_photos/'
rand_sel_images=[]
f_names=os.listdir(dirs)
for i in range(5):
	choice=random.choice(f_names)
	choice_num=choice.split('.')
	rand_sel_images.append(int(choice_num[0]))
#print len(rand_sel_images)

def get_busi_ids(rand_sel_images):
	bus_ids=[]
	count=1

	for k in range(len(rand_sel_images)):
		i = rand_sel_images[k]
		print(i)
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

b_IDS=get_busi_ids(rand_sel_images)
print(b_IDS)


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

final_labels=get_labels(b_IDS)
print (final_labels)

big_li = []
for element in final_labels:
	li = [0]*9
	for small_element in element:
		li[small_element] = 1
	big_li.append(li)

#big_li = np.asarray(big_li)

#print (big_li[0])

with open('image_to_attr.csv','wb') as csvfile:
	spamwriter=csv.writer(csvfile,delimiter=' ',quotechar='|')
	for ele in range(len(rand_sel_images)):
		spamwriter.writerow([rand_sel_images[ele],big_li[ele]])



'''

def get_vec(dirs):
	f_names = os.listdir(dirs)
	x_data = []
	for name in f_names:
		img_name = dirs+'/'+name
		try:
			img = Image.open(img_name)
			img.load()
			data = np.asarray(img, dtype="int32" )
			x_data.append(data)
		except IOError:
			print img_name
	return x_data

parent_dir = '/home/pulkit/Downloads/flower_photos_copy'
os.chdir(parent_dir)

res = []

all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
for dirs in all_subdirs:
    # print dirs
    d = os.path.join(parent_dir, dirs)
    a = np.asarray(get_vec(d))
    print a.shape
    if res == []:
    	res = a
    else:
    	res = np.vstack((res,a))




y = [i/8 for i in range(40)]

y = np.asarray(y)
#y=np.reshape(y,(40,1))
#y=y.T


with gzip.open('image_data.pkl.gz','wb') as f:
    cPickle.dump((res,y),f)
f.close()
'''
