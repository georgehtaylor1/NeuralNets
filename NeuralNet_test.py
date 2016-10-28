import numpy as np
import NeuralNet as nn
import csv
import random

data_file = "gridwatch.csv"
iterations = 5000
sample_cases = 20000
test_cases = 500
input_size =
hidden_size = 10
hidden_layers = 4
output_size = 1
alpha = 1

inputG = np.array([[],
                   [0, 1, 0],
                   [1, 0, 0],
                   [1, 1, 0],
                   [0, 0, 1],
                   [0, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

outputG = np.array([[0, 0],
                    [1, 0],
                    [1, 0],
                    [0, 1],
                    [1, 0],
                    [0, 1],
                    [0, 1],
                    [1, 1]])

data_list = []
with open(data_file, 'rb') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        data_list.append(float(row[6]))
for i in range(0, len(data_list)-1):
    if data_list[i] < data_list[i+1]:
        data_list[i] = 1
    else:
        data_list[i] = 0

print len(data_list)

inputH = np.array([])
outputH = np.array([])
training_list = []
training_output_list = []
test_list = []
test_output_list = []
for i in range(sample_cases):
    pos = random.randint(0, len(data_list) - input_size - 1)
    training_list.append(data_list[pos:pos + input_size])
    training_output_list.append([data_list[pos + input_size]])

for i in range(test_cases):
    pos = random.randint(0, len(data_list) - input_size - 1)
    test_list.append(data_list[pos:pos + input_size])
    test_output_list.append([data_list[pos + input_size]])

inputH = np.asarray(training_list)
outputH = np.asarray(training_output_list)

input_test = np.asarray(test_list)
output_test = np.asarray(test_output_list)

print "----------------------------"

#print inputH
#print outputH

net = nn.NeuralNet(input_size, hidden_size, output_size, hidden_layers, alpha)
net.test(input_test, test_cases, output_test)
#net.train(inputH, sample_cases, outputH, iterations)

