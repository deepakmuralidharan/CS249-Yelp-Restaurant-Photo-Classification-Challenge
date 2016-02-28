from PIL import Image
import numpy as np
import os
import gzip,cPickle
import random
import csv

dirs=''
rand_sel_images=[]
f_names=os.listdir(dirs)
for i in range(25000):
	choice=random.choice(f_names)
	choice_num=choice.split('.')
	rand_sel_images.append(choice_num)


def get_busi_ids(rand_sel_images):
	bus_ids=[]
	with open('filename.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for i in rand_sel_images:
		for row in spamreader:
			if i==row[0].split(',')[0]:
				bus_ids.append(row[0].split(',')[1])
				break

b_IDS=get_busi_ids(rand_sel_images)

def get_labels(bus_ids):
	labels_ids=[]
	with open('filename.csv', 'rb') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for i in bus_ids:
		for row in spamreader:
			if i=row[0].split(',')[0]:
				labels_ids.append(row[0].split(',')[1])
				break

final_labels=get_labels(b_IDS)

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
