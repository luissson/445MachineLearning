#!/anaconda3/bin/python
import numpy as np
import pandas
import time
from collections import defaultdict
import pickle
from datetime import datetime
import pdb

class Network(object):
    '''
    Perceptron network for identifying handwritten digits

    Target network:
    [784 input + 1 bias input units] -> [n_hidden hidden units] -> [10 output units]
    '''
    def __init__(self, msize=0.1, eta=0.001, momentum=1, print_logs=False):
        ## Network Setup
        self.n_input = 784
        self.n_hidden = 20 # 20, 50, 100
        self.n_output = 10 
        self.weight_range = 0.02 # sigma for normal distribution

        ## Training Setup
        self.n_epochs = 50
        self.subset_fraction = msize
        self.learning_rate = eta
        self.momentum = momentum 

        ## Misc
        self.prints = print_logs

        ## Network Initialization
        self.weights = [self.weight_range * np.random.randn(self.n_hidden, self.n_input + 1), np.random.randn(self.n_output, self.n_hidden + 1)] # + 1 is for bias input

        if self.prints:
            print("----------")
            print("Perceptron network parameters:")
            print(f"Input layer size: {self.n_input + 1} units")
            print(f"Hidden layer size: {self.n_hidden} units")
            print(f"Output layer size: {self.n_output} units")
            print("Training parameters: ")
            print(f"Training for {self.n_epochs} epochs")
            print(f"Subset fraction for stochastic training: {self.subset_fraction} of training data")
            print(f"Learning rate: {self.learning_rate}")
            print("----------")


    def sigma(self, z):
        return 1 / (1 + np.exp(-z))


    def Test(self, test_data, build_matrix=False):
        '''
        '''

        n_test = len(test_data)
        results = np.zeros(n_test)
        idx = 0
        if build_matrix:
            conf_matrix = defaultdict(int)

        ## Compute hidden layer outputs
        for features, target in test_data:
            features = np.append(features, 1) # append bias here
            perceptron_sums = self.hidden_activations(features)
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


    def Train(self, training_data, slow_training=False):
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
        results = []

        ## Select a subset
        np.random.shuffle(training_data) #randomize training set
        training_sets = [training_data[k:k+subset_size] for k in range(0, M, subset_size)]

        start_training = time.time()
        ho_delta_hist = [0]
        ih_delta_hist = [0]
        for epoch_num in range(self.n_epochs):
            for training_set in training_sets:
                for features, target in training_set:
                    ## Compute hidden layer activations
                    features = np.append(features, 1) # append input bias here

                    hidden_activations = self.hidden_activations(features)
                    hidden_activations = np.append(hidden_activations, 1) # append hidden bias here

                    ## Compute output layer activations
                    output_activations = self.output_activations(hidden_activations)

                    ## Compute output layer error
                    output_targets = np.full(self.n_hidden, 0.1)
                    output_targets[target] = 0.9
                    output_errors = self.output_error(output_activations, output_targets)

                    ## Compute hidden layer error
                    hidden_errors = self.hidden_error(hidden_activations, output_errors)

                    ## Update hidden to output layer weights
                    self.update_ho_weights(hidden_activations, output_errors, ho_delta_hist)

                    ## Update input to hidden layer weights
                    self.update_ih_weights(features, hidden_errors, ih_delta_hist)
                

            print(f"Epoch #{epoch_num} elapsed time: {(time.time() - start_training)/60.0:.2f} minutes")
            if epoch_num == 5:
                pdb.set_trace()

            if slow_training:
                # Test model with test and training data after each epoch
                # train_result = self.Test(training_data)

                #if self.prints:
                    #print(f"Training set accuracy with no training: {100*train_result:.2f}%")

                #results.append(train_result)

        if slow_training:
            # Save training results to file here
            badchars = [' ', ':']
            with open(f"results/training_results_{str(datetime.now()).translate({ord(x): '_' for x in badchars})}_{self.learning_rate}_{subset_size}.data", "wb") as f:
                pickle.dump(results, f)
            print("Training results written to file.")

            return results
        else:
            return None


    def hidden_activations(self, features):
        '''
        Computes the output from the hidden layer (number of perceptrons given by self.n_hidden)
        '''
        hidden_activations = []
        features = np.reshape(features, (features.shape[0], 1)) # reshape features array from eg. (785,) to (785, 1)
        for i in range(self.n_hidden):
            z = np.dot(self.weights[0][i], features) #weights*features dot product
            hidden_activations.append(self.sigma(z)[0]) #threshold activation function

        return hidden_activations


    def output_activations(self, features):
        '''
        Computes the output from the hidden layer (number of perceptrons given by self.n_hidden)
        '''
        output_activations = []
        # features = np.reshape(features, (features.shape[0], 1)) # reshape features array from eg. (785,) to (785, 1)
        for i in range(self.n_output):
            z = np.dot(self.weights[1][i], features) #weights*features dot product
            output_activations.append(self.sigma(z)) #threshold activation function

        return output_activations 


    def output_error(self, outputs, targets):
        output_errors = []
        for i in range(self.n_output):
            output_errors.append(outputs[i] * (1.0 - outputs[i]) * (targets[i] - outputs[i]))

        return output_errors


    def hidden_error(self, hidden_activations, output_errors):
        hidden_errors = []
        for j in range(self.n_hidden):
            weighted_out_errors = 0
            for k in range(self.n_output):
                weighted_out_errors += self.weights[1][k][j]
            hidden_errors.append(hidden_activations[j] * (1 - hidden_activations[j]) * weighted_out_errors)

        return hidden_errors

    def update_ho_weights(self, hidden_activations, output_errors, ho_delta_hist):
        ## w_kj <- w_kj + eta * d_k * h_j
        # self.weights[1] output units
        # self.weights[1][0] weights from n_hidden that attach to the 0th output unit
        # in class note notation: self.weights[1][k][j]
        for k in range(self.n_output):
            for j in range(self.n_hidden):
                ho_delta = self.learning_rate * output_errors[k] * hidden_activations[j]
                ho_delta_hist.append(ho_delta)
                self.weights[1][k][j] += ho_delta + self.momentum * ho_delta_hist.pop()


    def update_ih_weights(self, features, hidden_errors, ih_delta_hist):
        ## w_ji <- w_ji + eta * d_j * x_i
        # self.weights[0] hidden units
        # self.weights[0][0] weights from input that attach to the 0th hidden unit
        # in class note notation: self.weights[0][j][i]
        for j in range(self.n_hidden):
            for i in range(self.n_input):
                ih_delta = self.learning_rate * hidden_errors[j] * features[i]
                ih_delta_hist.append(ih_delta)
                self.weights[0][j][i] += ih_delta + self.momentum * ih_delta_hist.pop()


def main():

    ## Initialize neural network
    ## For this assignment we will use 784 input, 10 hidden, and 1 output perceptrons
    ## Weights are randomly to be [-0.05, 0.05]
    nn_start = time.time()
    nn_net = Network()

    ## Load training data
    fname = "half_train.data"
    with open("data/" + fname, "rb") as f:
        half_train = pickle.load(f)

    fname = "qrter_train.data"
    with open("data/" + fname, "rb") as f:
        qrter_train = pickle.load(f)

    nn_net.Train(qrter_train, None)
    print(f"Network training completed in {(time.time() - nn_start)/60.0:.2f} minutes")


if __name__ == '__main__':
    main()