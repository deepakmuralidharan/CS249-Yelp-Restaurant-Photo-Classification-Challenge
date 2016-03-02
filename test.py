import numpy as np
import gzip,cPickle

with gzip.open('/Users/deepakmuralidharan/pickle_sample_files/sample_data1.pkl.gz','rb') as k:
    query=cPickle.load(k)
k.close()

print query[0][0]
print query[1][0]
