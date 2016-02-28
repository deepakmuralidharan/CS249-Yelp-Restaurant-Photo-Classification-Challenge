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

def serialize_cifar_pool3(X,filename):
    print 'About to generate file: %s' % filename
    sess = tf.InteractiveSession()
    X_pool3 = batch_pool3_features(sess,X)
    np.save(filename,X_pool3)

def serialize_data():

    with gzip.open('/home/pulkit/Downloads/image_data.pkl.gz','rb') as k:
        query=cPickle.load(k)
    k.close()

    X_train = query[0]
    y_train = query[1]
    X_test = query[0]
    y_test = query[1]

    serialize_cifar_pool3(X_train, 'X_train')
    serialize_cifar_pool3(X_test, 'X_test')
    np.save('y_train',y_train)
    np.save('y_test',y_test)

serialize_data()
