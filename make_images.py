import numpy as np
import os,sys
import gzip,cPickle
import random
import csv
import shutil

sample_size=10         # change this to 250
no_of_folder=5          # change  this to 100'''
#dirs='/Users/deepakmuralidharan/train_photos/'
dirs='/home/pulkit/codes/'                       #directory where training images are stored '''
sample_dirs='/home/pulkit/samples/'              # directory for creating 100 folders    '''
for j in range(no_of_folder):
    name='sample'+str(j+1)
    path=sample_dirs+name
    if(os.path.isdir(path)):
        shutil.rmtree(path)

rand_sel_images=[]
f_names=os.listdir(dirs)
print "Total number of files in the folder are "+str(len(f_names))

for j in range(no_of_folder):
    choice=random.sample(xrange(len(f_names)), sample_size)
    name='sample'+str(j+1)
    path=sample_dirs+name
    if(os.path.isdir(path)):
        shutil.rmtree(path)
    os.mkdir(path)                  # make folders with the name sample1 , sample2 ,......sample100 '''
    for ch in choice:
        shutil.copy2(dirs+f_names[ch], path)
    for c in sorted(choice,reverse=True):
        del f_names[c]
