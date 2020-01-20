#!/anaconda3/bin/python
import pdb
import numpy as np
import pandas
import time

class Network(object):
    '''
    Target perceptron network
    [784 inputs perceptron] -> [10 perceptrons] -> [1 perceptron]
    '''
    def __init__(self):
        self.n_input = 784
        self.n_hidden = 10
        self.n_output = 1
        self.weight_range = 0.02 # sigma for normal distribution

        self.biases = [np.ones((self.n_hidden, self.n_output))] # all bias are 1
        self.weights = [self.weight_range * np.random.randn(self.n_hidden, self.n_input), np.random.randn(self.n_output, self.n_hidden)]

    def Train(self):
        pass

    def activation(self, training_data):
        pdb.set_trace()
        np.dot(training_data, self.weights[0]) + self.biases


def main():
    init_start = time.time()

    train = pandas.read_csv('mnist_train.csv', header=None)
    train_load = time.time()
    print("Loaded mnist training data in %.4ss" % (train_load - init_start))

    test = pandas.read_csv('mnist_test.csv', header=None)
    test_load = time.time()
    print("Loaded mnist test data in %.4ss" % (test_load - train_load))

    training_data = [(train.iloc[i][1:].values/255.0, train.iloc[i][0]) for i in range(len(train))]
    train_proc = time.time()
    print("Processed training data in %.4ss" % (train_proc - test_load))

    test_proc = time.time()
    test_data = [(test.iloc[i][1:].values/255.0, test.iloc[i][0]) for i in range(len(test))]
    print("Processed test data in %.4ss" % (test_proc - train_proc))

    nn_start = time.time()
    perceptron_network = Network()
    perceptron_network.activation(training_data)


if __name__ == '__main__':
    main()