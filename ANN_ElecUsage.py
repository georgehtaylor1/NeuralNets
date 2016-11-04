import numpy as np
import NeuralNet as nn
import csv
import random
from random import shuffle
import math
from datetime import datetime
import matplotlib.pyplot as pl
import NeuralNet as neunet

data_file = "gridwatch.csv"
training_proportion = 0.75
power_divisor = 100000

def load_data():
    data_list = []
    with open(data_file, 'rb') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            data_list.append(row)

    # shuffle(data_list)
    full_len = len(data_list)
    training_len = int(math.floor(full_len * training_proportion))
    print "Sample cases loaded: {}".format(full_len)

    data_list = map((lambda v: (int(v[0]), datetime.strptime(v[1], " %Y-%m-%d %X"), float(v[2].strip())/power_divisor)), data_list)



    training_data = data_list[:training_len]
    test_data = data_list[training_len:]

    print "Training cases: {}".format(len(training_data))
    print "Test cases: {}".format(len(test_data))
    return data_list, training_data, test_data


def plot(data):
    unz_data = [list(t) for t in zip(*data)]
    pl.plot(unz_data[1], unz_data[2])
    pl.show()


def genData(data, length):
    train_data = []
    test_data = []
    start = 4
    for j in range(length):
        i = random.randrange(start, len(data))
        train_data.append([data[i][1].timetuple().tm_yday,
                           (data[i][1].timetuple().tm_hour * 60) + data[i][1].timetuple().tm_min,
                           data[i - 1][2], data[i - 2][2]])
        test_data.append([data[i][2]])
    return np.array(train_data), np.array(test_data)


net = nn.NeuralNet(4, 6, 1, 2, 0.1)

d, train, test_data = load_data()

def run():
    input_length = 200
    iterations = 100000
    errors = []
    for i in xrange(iterations):
        sub_train, sub_out = genData(train, input_length)
        errors.append(net.train(sub_train, input_length, sub_out))
    return errors


def test():
    input_length = 200
    sub_test, sub_out = genData(test_data, input_length)
    error = net.test(sub_test, input_length, sub_out)
    return error