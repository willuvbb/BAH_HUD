from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input

# import the function that will get the data
from sklearn.datasets import load_breast_cancer

# load the data
data = load_breast_cancer()

# check the type of 'data'
type(data)

# note: it is a bunch object
# this is like a dictionary where you can treat the keys like attriburtes//?
data.keys()

# 'data' (the attribute) means the input data
data.data.shape