#!/anaconda3/bin/python
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
    [784 input + 1 bias input units] -> [n_hidden hidden units] -> [10 output units]
    '''
    def __init__(self, subset_size=10, n_hidden=20, eta=0.1, momentum=0.0, print_logs=False):
        ## Network Setup
        self.n_input = 784
        self.n_hidden = n_hidden # 20, 50, 100
        self.n_output = 10 
        self.weight_range = 0.02 # sigma for normal distribution

        ## Training Setup
        self.n_epochs = 50
        self.learning_rate = eta
        self.momentum = momentum 
        self.subset_size = subset_size

        ## Misc
        self.print_logs = print_logs

        ## Network Initialization
        self.weights = [self.weight_range * np.random.randn(self.n_hidden, self.n_input + 1), self.weight_range * np.random.randn(self.n_output, self.n_hidden + 1)] # + 1 is for bias input

        if self.print_logs:
            print("----------")
            print("Perceptron network parameters:")
            print(f"Input layer size: {self.n_input + 1} units")
            print(f"Hidden layer size: {self.n_hidden} units")
            print(f"Output layer size: {self.n_output} units")
            print("Training parameters: ")
            print(f"Training for {self.n_epochs} epochs")
            print(f"Subset size for mini-batch training: {self.subset_size} of training data")
            print(f"Learning rate: {self.learning_rate}")
            print("----------")


    def sigma(self, z):
        return 1 / (1 + np.exp(-z))


    def Test(self, test_data, build_matrix=True):
        '''
        '''
        n_test = len(test_data)
        results = np.zeros(n_test)
        idx = 0
        if build_matrix:
            conf_matrix = defaultdict(int)

        ## Compute hidden layer outputs
        for features, target in test_data:
            ## Compute hidden layer activations
            hidden_activations = self.hidden_activations(features)
            hidden_activations = np.append(hidden_activations, 1) # append hidden bias here

            ## Compute output layer activations
            output_activations = self.output_activations(hidden_activations)

            prediction = np.argmax(output_activations)

            if build_matrix:
                conf_matrix[(prediction, target)] += 1

            results[idx] = 1 if prediction == target else 0
            idx += 1

        accuracy = np.sum(results) / n_test
        
        if build_matrix:
            return accuracy, conf_matrix
        else:
            return accuracy


    def Train(self, training_data, test_data, slow_training=False):
        '''
        Runner method for neural net training algorithm with backpropagation and momentum
        '''
        M = len(training_data)
        results = []

        ## Build list of random examples of size self.subset_size
        training_sets = [training_data[k:k+self.subset_size] for k in range(0, M, self.subset_size)]

        ## Get accuracy on training data with no training
        no_train_result = (self.Test(training_data), self.Test(test_data))

        if self.print_logs:
            print(f"Training set accuracy with no training: {100*no_train_result:.2f}%")

        results.append(no_train_result)

        ## Stack used for momentum terms (stack implemented with python list append + pop)
        ho_delta_hist = [np.zeros((self.n_hidden + 1, 1))]
        ih_delta_hist = [np.zeros((self.n_input + 1, 1))]
        for epoch_num in range(self.n_epochs):
            epoch_start = time.time()
            for training_set in training_sets:
                ## Algorithm utilizes mini-batches, the below containers hold corresponding computations for each mini-batch
                hidden_activations = []
                output_activations = []
                output_errors = []
                hidden_errors = []
                for features, target in training_set:
                    ## Compute hidden layer activations
                    hidden_activations.append(self.hidden_activations(features))
                    hidden_activations[-1].append(1) # append hidden bias here

                    ## Compute output layer activations
                    output_activations.append(self.output_activations(hidden_activations[-1]))

                    ## Compute output layer error
                    output_targets = np.full(self.n_output, 0.1) # expensive operation here, should re-write elements instead of allocating new arrays
                    output_targets[target] = 0.9
                    output_errors.append(self.output_error(output_activations[-1], output_targets))

                    ## Compute hidden layer error
                    hidden_errors.append(self.hidden_error(hidden_activations[-1], output_errors[-1]))

                ## Update hidden to output layer weights
                self.update_ho_weights(hidden_activations, output_errors, ho_delta_hist)

                ## Update input to hidden layer weights
                self.update_ih_weights(training_set, hidden_errors, ih_delta_hist)

            print(f"Epoch #{epoch_num + 1} took: {(time.time() - epoch_start):.2f} seconds")

            if slow_training:
                # Test model with test and training data after each epoch
                train_result = self.Test(training_data)
                test_result = self.Test(test_data)

                if self.print_logs:
                    print(f"Training set accuracy: {100*train_result:.2f}%")
                    print(f"Test set accuracy: {100*test_result:.2f}%")

                results.append((train_result, test_result))
            
        if slow_training:
            badchars = [' ', ':']
            with open(f"results/training_results_{str(datetime.now()).translate({ord(x): '_' for x in badchars})}_{self.n_hidden}_{len(training_data)}_{self.momentum}.data", "wb") as f:
                pickle.dump(results, f)

            if self.print_logs:
                print("Training results written to file.")

            return results
        else:
            return None


    def hidden_activations(self, features):
        '''
        Computes the output from the hidden layer (number of perceptrons given by self.n_hidden)
        '''
        hidden_activations = [0] * self.n_hidden # speeds up run-time if we pre-allocate arrays (instead of use append)
        features = np.reshape(features, (features.shape[0], 1)) # reshape features array from eg. (785,) to (785, 1)
        for i in range(self.n_hidden):
            z = np.dot(self.weights[0][i], features)[0] #weights*features dot product, note both ndarrays -> returns ndarray
            hidden_activations[i] = self.sigma(z)

        return hidden_activations


    def output_activations(self, hidden_activations):
        '''
        Computes the output from the hidden layer (number of perceptrons given by self.n_hidden)
        '''
        output_activations = [0] * self.n_output # speeds up run-time if we pre-allocate arrays (instead of use append)
        for i in range(self.n_output):
            z = np.dot(self.weights[1][i], hidden_activations) #weights*features dot product, note np.dot return is a float 
            output_activations[i] = self.sigma(z)

        return output_activations 


    def output_error(self, outputs, targets):
        '''
        Computes the error for each output unit
        '''
        output_errors = [0] * self.n_output
        for i in range(self.n_output):
            output_errors[i] = outputs[i] * (1.0 - outputs[i]) * (targets[i] - outputs[i])

        return output_errors


    def hidden_error(self, hidden_activations, output_errors):
        '''
        Computes the error for each hidden unit
        '''
        hidden_errors = [0] * self.n_hidden
        for j in range(self.n_hidden):
            weighted_out_errors = 0
            for k in range(self.n_output):
                weighted_out_errors += self.weights[1][k][j] * output_errors[k] # this can be sped up using numpy properly

            hidden_errors[j] = hidden_activations[j] * (1 - hidden_activations[j]) * weighted_out_errors

        return hidden_errors


    def update_ho_weights(self, hidden_activations, output_errors, ho_delta_hist):
        '''
        Update hidden->output unit weights
        '''
        ## w_kj <- w_kj + eta * d_k * h_j
        # self.weights[1] output units
        # self.weights[1][0] weights from n_hidden that attach to the 0th output unit
        # in class note notation: self.weights[1][k][j]
        subset_size = len(hidden_activations)
        for k in range(self.n_output):
            ho_delta = self.learning_rate * np.sum([np.multiply(output_errors[i][k], hidden_activations[i]) for i in range(len(hidden_activations))], 0) / subset_size
            ho_delta_hist.append(ho_delta)
            self.weights[1][k] = np.add(self.weights[1][k], ho_delta + np.multiply(self.momentum, ho_delta_hist.pop()))


    def update_ih_weights(self, training_set, hidden_errors, ih_delta_hist):
        '''
        Update input->hidden unit weights
        '''
        ## w_ji <- w_ji + eta * d_j * x_i
        # self.weights[0] hidden units
        # self.weights[0][0] weights from input that attach to the 0th hidden unit
        # in class note notation: self.weights[0][j][i]
        subset_size = len(training_set)
        for j in range(self.n_hidden):
            ih_delta = self.learning_rate * np.sum([np.multiply(hidden_errors[i][j], training_set[i][0]) for i in range(len(training_set))], 0) / subset_size
            ih_delta_hist.append(ih_delta)
            self.weights[0][j] = np.add(self.weights[0][j], ih_delta + np.multiply(self.momentum, ih_delta_hist.pop()))


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

    fname = "test.data"
    with open("data/" + fname, "rb") as f:
        test_data = pickle.load(f)

    fname = "qrter_train.data"
    with open("data/" + fname, "rb") as f:
        qrter_train = pickle.load(f)

    nn_net.Train(qrter_train, test_data, True)
    print(f"Network training completed in {(time.time() - nn_start)/60.0:.2f} minutes")

    badchars = [' ', ':']
    with open(f"results/nn_qrtr_{str(datetime.now()).translate({ord(x): '_' for x in badchars})}.model", "wb") as f:
        pickle.dump(nn_net, f)


if __name__ == '__main__':
    main()