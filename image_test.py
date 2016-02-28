from PIL import Image
import numpy as np
import os
import gzip,cPickle
import random
import csv
from sklearn.preprocessing import OneHotEncoder
import ast

with open('/Users/deepakmuralidharan/Documents/CS249-Big-Data-Analytics/image_to_attr.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    rr = []
    for row in spamreader:
        x=ast.literal_eval(row[1])
        rr.append(x)

rr = np.asarray(rr)
print(rr[:,1])
