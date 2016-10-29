import numpy as np
import NeuralNet as nn
import csv
import random
from random import shuffle
import math
from datetime import datetime

data_file = "gridwatch.csv"
training_proportion = 0.75

data_list = []
with open(data_file, 'rb') as csv_file:
    reader = csv.reader(csv_file)
    next(reader)
    for row in reader:
        data_list.append(row)

shuffle(data_list)
full_len = len(data_list)
training_len = int(math.floor(full_len * training_proportion))
print "Sample cases loaded: {}".format(full_len)

training_data = data_list[:training_len]
test_data = data_list[training_len:]

print "Training cases: {}".format(len(training_data))
print "Test cases: {}".format(len(test_data))