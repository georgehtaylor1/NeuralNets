import numpy as np


def sigmoid(x):
    r = 0
    try:
        r = 1 / (1 + np.exp(-x))
    except err:
        print x
        1/0
    return r

def sigmoid_output_to_deriv(x):
    return x * (1 - x)


class NeuralNet(object):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers, alpha):
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.alpha = alpha

        self.synapse_input = 2 * np.random.random((input_size, hidden_size)) - 1
        self.synapse_output = 2 * np.random.random((hidden_size, output_size)) - 1
        self.synapse_hidden = 2 * np.random.random((hidden_layers, hidden_size, hidden_size)) - 1

    def feed_forward(self, input_data, input_length):
        layer_input = input_data
        layer_hidden = np.zeros((self.hidden_layers, input_length, self.hidden_size))
        layer_hidden[0] = sigmoid(np.dot(layer_input, self.synapse_input))
        for j in range(1, self.hidden_layers):
            layer_hidden[j] = sigmoid(np.dot(layer_hidden[j - 1], self.synapse_hidden[j - 1]))
        layer_output = sigmoid(np.dot(layer_hidden[self.hidden_layers - 1], self.synapse_output))
        return layer_input, layer_hidden, layer_output

    def train_full(self, sample_data, sample_cases, output_data, iterations):
        for i in xrange(iterations):
            train(sample_data, sample_cases, output_data)

    def train(self, sample_data, sample_cases, output_data):
        layer_input, layer_hidden, layer_output = self.feed_forward(sample_data, sample_cases)
        # Calculate the error in the output
        output_error = layer_output - output_data
        output_delta = output_error * sigmoid_output_to_deriv(layer_output)

        # Back propagate the errors
        hidden_errors = np.zeros((self.hidden_layers, sample_cases, self.hidden_size))
        hidden_deltas = np.zeros((self.hidden_layers, sample_cases, self.hidden_size))

        hidden_errors[self.hidden_layers - 1] = output_delta.dot(self.synapse_output.T)
        hidden_deltas[self.hidden_layers - 1] = hidden_errors[self.hidden_layers - 1] * sigmoid_output_to_deriv(
            layer_hidden[self.hidden_layers - 1])

        for j in range(self.hidden_layers - 2, -1, -1):
            hidden_errors[j] = hidden_deltas[j + 1].dot(self.synapse_hidden[j].T)
            hidden_deltas[j] = hidden_errors[j + 1] * sigmoid_output_to_deriv(layer_hidden[j])

            # Adjust the synapse weights
            self.synapse_output -= self.alpha * (layer_hidden[self.hidden_layers - 1].T.dot(output_delta))
        for j in range(self.hidden_layers - 2, -1, -1):
            self.synapse_hidden[j] -= self.alpha * (layer_hidden[j].T.dot(hidden_deltas[j + 1]))

            self.synapse_input -= self.alpha * (layer_input.T.dot(hidden_deltas[0]))

        _, _, layer_output = self.feed_forward(sample_data, sample_cases)
        #print layer_output
        training_error = np.mean(np.abs(output_error))
        print "Error after iteration:" + str(training_error)
        return training_error

    def test(self, input_data, input_length, output_data):
        layer_input, layer_hidden, layer_output = self.feed_forward(input_data, input_length)
        count = 0
        for i in range(input_length):
            print(layer_output[i][0])
            print(output_data)
            print("------------------------")
            if layer_output[i] == output_data[i]:
                count += 1
        print(count)
        return count / float(input_length)
