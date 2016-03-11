import tensorflow as tf
import numpy as np
from sklearn import cross_validation
from extract import create_graph, iterate_mini_batches, batch_pool3_features
from datetime import datetime
import matplotlib.pyplot as plt
from tsne import tsne
import seaborn as sns
import pandas as pd
import gzip,cPickle
import os

""" script to concatenate the CNN codes into a single numpy array"""

Xtrain_file = '/Users/ndolocal/Documents/CS249-Big-Data-Analytics-master/X_train1.npy'      """ File Path for CNN codes """
y_file = '/Users/ndolocal/Documents/CS249-Big-Data-Analytics-master/y_train1.npy'			""" File path for the training labels""" 
X_train=np.load(Xtrain_file)
y_train=np.load(y_file)

for k in range(1,925):
	print(k)
	Xtrain_file='/Users/ndolocal/Documents/CS249-Big-Data-Analytics-master/X_train'+str(k+1)+'.npy'
	y_file='/Users/ndolocal/Documents/CS249-Big-Data-Analytics-master/y_train'+str(k+1)+'.npy'
	#if(os.path.isfile(Xtrain_file) and os.path.isfile(y_file)):
	X_train=np.vstack((X_train,np.load(Xtrain_file)))
	y_train=np.vstack((y_train,np.load(y_file)))


np.save('conc_CNN_code',X_train)
np.save('conc_CNN_label',y_train)
