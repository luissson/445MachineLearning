#!/anaconda3/bin/python
import pdb
import numpy as np
import pandas
import time
from collections import defaultdict
import pickle
from datetime import datetime

class Network(object):
    '''
    Perceptron network for identifying handwritten digits

    Target network:
    [784 inputs perceptron + 1 bias perceptron] -> [10 hidden perceptrons] -> [1 output perceptron]
    '''
    def __init__(self, msize=0.1, eta=0.001, print_logs=False):
        ## Network Setup
        self.n_input = 784
        self.n_hidden = 10
        self.n_output = 1
        self.weight_range = 0.02 # sigma for normal distribution

        ## Training Setup
        self.n_epochs = 50
        self.subset_fraction = msize
        self.learning_rate = eta

        ## Misc
        self.prints = print_logs

        ## Network Initialization
        self.weights = [self.weight_range * np.random.randn(self.n_hidden, self.n_input + 1), np.random.randn(self.n_output, self.n_hidden)] # + 1 is for bias input

        if self.prints:
            print("----------")
            print("Perceptron network parameters:")
            print(f"Input layer size: {self.n_input + 1} perceptrons")
            print(f"Hidden layer size: {self.n_hidden} perceptrons")
            print(f"Output layer size: {self.n_output} perceptrons")
            print("Training parameters: ")
            print(f"Training for {self.n_epochs} epochs")
            print(f"Subset fraction for stochastic training: {self.subset_fraction} of training data")
            print(f"Learning rate: {self.learning_rate}")
            print("----------")


    def Test(self, test_data, build_matrix=False):
        n_test = len(test_data)
        results = np.zeros(n_test)
        idx = 0
        if build_matrix:
            conf_matrix = defaultdict(int)

        ## Compute hidden layer outputs
        for features, target in test_data:
            features = np.append(features, 1) # append bias here
            _, perceptron_sums = self.hidden_outputs(features)
            prediction = np.argmax(perceptron_sums)

            if build_matrix:
                conf_matrix[(prediction, target)] += 1

            results[idx] = 1 if prediction == target else 0
            idx += 1

        accuracy = np.sum(results) / n_test
        
        if build_matrix:
            return accuracy, conf_matrix
        else:
            return accuracy


    def Train(self, training_data, test_data, fast=False):
        '''
        Runner method for perceptron training algorithm.

        Algorithm Overview:
        1. Select a data subset from the training data (size M)
        2. For each set of (features (x_i), target (t)) in the subset do:
            I. Compute y = a(x.w + b)
            II. If y != t, update weights w_i. For each w_i:
                1. w_i <- w_i + eta (t - y) x_i
        '''
        ## Build list of random-sub samples
        M = len(training_data)
        subset_size = int(M * self.subset_fraction)

        if not fast:
            results = []
            test_result = self.Test(test_data)
            train_result = self.Test(training_data)

            results.append((train_result, test_result))

            if self.prints:
                print(f"Training set accuracy with no training: {100*train_result:.2f}%")
                print(f"Test set accuracy with no training: {100*test_result:.2f}%")

        ## Select a subset
        np.random.shuffle(training_data)
        training_sets = [training_data[k:k+subset_size] for k in range(0, M, subset_size)]

        for _ in range(self.n_epochs):
            for training_set in training_sets:
                ## Compute hidden layer outputs
                for features, target in training_set:
                    features = np.append(features, 1) # append bias here
                    hidden_targets = np.zeros(self.n_hidden)
                    hidden_targets[target] = 1 # Assumes target is an int < n_hidden
                    hidden_outputs, _ = self.hidden_outputs(features)
                
                    ## Update weights for incorrect predictions
                    incorrect_hidden_perceptrons = np.where(hidden_targets != hidden_outputs)[0]
                    for j in incorrect_hidden_perceptrons:
                        self.weights[0][j] = self.weights[0][j] + self.learning_rate * (hidden_targets[j] - hidden_outputs[j]) * features

            if not fast:
                train_result = self.Test(training_data)
                test_result = self.Test(test_data)

                if self.prints:
                    print(f"Training set accuracy with no training: {100*train_result:.2f}%")
                    print(f"Test set accuracy with no training: {100*test_result:.2f}%")

                results.append((train_result, test_result))

        if not fast:
            badchars = [' ', ':']
            with open(f"results/training_results_{str(datetime.now()).translate({ord(x): '_' for x in badchars})}_{self.learning_rate}_{subset_size}.data", "wb") as f:
                pickle.dump(results, f)
            print("Training results written to file.")
            return results
        else:
            return None


    def hidden_outputs(self, features):
        hidden_outputs = []
        hidden_sums = []
        features = np.reshape(features, (features.shape[0], 1)) # reshape features array from eg. (785,) to (785, 1)
        for i in range(self.n_hidden):
            perceptron_sum = np.dot(self.weights[0][i], features)
            hidden_outputs.append(np.where(perceptron_sum > 0, 1, 0)[0])
            hidden_sums.append(perceptron_sum)

        return hidden_outputs, hidden_sums


def main():

    ## Load and process the training and test data from csv
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

    ## Initialize perceptron network
    ## For this assignment we will use 784 input, 10 hidden, and 1 output perceptrons
    ## Weights are randomly to be [-0.05, 0.05]
    nn_start = time.time()
    perceptron_network = Network()

    perceptron_network.Train(training_data, test_data)
    print(f"Network training completed in {(time.time() - nn_start)/60.0:.2f} minutes")

    accuracy = perceptron_network.Test(test_data)
    print(f"Test set accuracy: {100*accuracy:.2f}%")


if __name__ == '__main__':
    main()