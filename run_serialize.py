"""
Yelp Restaurant Photo Classification Challenge
"""
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

number_of_folder = 1000

def serialize_cifar_pool3(X,filename):
    print 'About to generate file: %s' % filename
    sess = tf.InteractiveSession()
    X_pool3 = batch_pool3_features(sess,X)
    np.save(filename,X_pool3)

def serialize_data():

    '''name='/Users/deepakmuralidharan/pickle_sample_files/sample_data1.pkl.gz'
    with gzip.open(name,'rb') as k:
        query=cPickle.load(k)
    k.close()
    X_train = query[0]
    y_train = query[1]

    for k in range(1,number_of_folder):
        name='/Users/deepakmuralidharan/pickle_sample_files/sample_data'+str(k+1)+'.pkl.gz'
        with gzip.open(name,'rb') as k:
            query=cPickle.load(k)
        k.close()
        X_train=np.vstack((X_train,query[0]))
        y_train=np.vstack((y_train,query[1]))'''

    for k in range(number_of_folder):
        name='/Users/deepakmuralidharan/pickle_sample_files/sample_data'+str(k+1)+'.npz'
        if(os.path.isfile(name)):                                                                """Check if file exists """
            query=np.load(name)
            X_train=query['X_train']
            y_train=query['y_train']           

            X_test = X_train
            y_test = y_train
            serialize_cifar_pool3(X_train, 'X_train'+str(k+1))                                  """ Next four lines are changed to store CNN codes with different names """
            serialize_cifar_pool3(X_test, 'X_test'+str(k+1))
            np.save('y_train'+str(k+1),y_train)
            np.save('y_test'+str(k+1),y_test)
                                                                                   
serialize_data()
