#!/anaconda3/bin/python
import pdb

import numpy as np

class Network(object):
    '''
        sizes: The number of neurons in each layer.
                ex) [2, 3, 1] - 2 first, 3 second, 1 third
    '''
    def __init__(self, sizes):
        self.n_layers = len(sizes)
        self.sizes = sizes

        ## 
        self.biases = [ np.random.randn(y, 1) for y in sizes[1:]]

        ## Array of matrices with weights that connect to each layer
        ## weight[1]_jk is the weight for the path between the
        ## k-th neuron in layer 2 and the jth neuron in layer 3
        ## note this is 'backwards'  for activation function
        ## a` = s(wa+b)
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]

    def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    '''
    train: (x, y) pairs
    epochs: number of epochs to train for
    mbatch_size: size of the mini batches
    eta: learning rate
    test: optional, prints partial progress after each epoch training
    '''
    def SGD(self, train, epochs, mbatch_size, eta, test):
        ntest = len(test)
        n = len(train)

        for j in range(epochs):
            ## shuffle training data on each epoch
            np.random.shuffle(train)
            print(train)
            ## build a set of mini-batches (random set of data from training data)
            mbatches = [train[k:k+mbatch_size] for k in range(0, n, mbatch_size)]
            ##apply a step of gradient descent for each mini batch
            for mbatch in mbatches:
                try:
                    self.update_mbatch(mbatch, eta)
                except TypeError:
                    pass

            if test:
                print("Eoch %s: %s / %s" % (j, self.evaluate(test), ntest))
            else:
                print("Epoch %s complete" % j)
    
    def update_mbatch(self, mbatch, eta):
        print("updating")
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nalba_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mbatch:
            delta_nabla_b, detla_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(eta/len(mbatch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mbatch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        delta = self.cost_derivative(activations[-1], y)*sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta)* sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test):
        test = [(np.argmax(self.feedforward(x)), y) for (x, y) in test]
        return sum(int(x == y) for (x, y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    
    def sigmoid_prime(z):
        return sigmoid(z)*(1-sigmoid(z))
