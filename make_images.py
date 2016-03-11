import numpy as np
import os,sys
import gzip,cPickle
import random
import csv
import shutil

sample_size= 250         # change this to 250
no_of_folder= 1000       # change  this to 100'''
#dirs='/Users/deepakmuralidharan/train_photos/'
dirs='/Users/deepakmuralidharan/test_photos/'                       #directory where training images are stored '''
sample_dirs='/Users/deepakmuralidharan/sample_test/'              # directory for creating 100 folders    '''


#test directory for pulkit 
#dirs='/home/pulkit/sample_images/'
#sample_dirs='/home/pulkit/samples/'

for j in range(no_of_folder):
    name='sample'+str(j+1)
    path=sample_dirs+name
    if(os.path.isdir(path)):
        shutil.rmtree(path)

rand_sel_images=[]
f_names=[]
for file_sample in os.listdir(dirs):                                  #  """" This will select only the files which ends with .jpg """
       if(file_sample.endswith(".jpg") or file_sample.endswith(".jpeg")):
            f_names.append(file_sample)
#f_names=os.listdir(dirs)
print "Total number of files in the folder are "+str(len(f_names))

                                                         
for j in range(no_of_folder):
    print(j)
    if len(f_names)>0:                                                    # """  condition check for number of folders to be created less than 1000 """"  
        if(len(f_names)>=sample_size):
            choice=random.sample(xrange(len(f_names)), sample_size)
        else:
            choice=random.sample(xrange(len(f_names)), len(f_names))      #""" for selecting number of images when size is less than the sample_size """
        name='sample'+str(j+1)
        path=sample_dirs+name
        if(os.path.isdir(path)):
            shutil.rmtree(path)
        os.mkdir(path)                  # make folders with the name sample1 , sample2 ,......sample100 '''
        for ch in choice:
            shutil.copy2(dirs+f_names[ch], path)
        for c in sorted(choice,reverse=True):
            del f_names[c]
