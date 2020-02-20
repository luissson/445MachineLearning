import math
from collections import defaultdict
import pdb
import numpy as np
import pandas as pd
import os
import sys
from operator import methodcaller
import ast
from scipy.stats import norm


def naive_bayes_train(training_data):
    '''
    Gaussian Naive Bayes Implementation
    Aglorithm:
        I. Compute the probability for each class
        II. Compute the mean of each feature for each class
        III. Compute the standard deviation of each feature for each class
    '''
    inputs_mean = defaultdict(list)
    inputs_std = defaultdict(list)
    inputs_params = defaultdict(list)

    classes = training_data['class'].unique()
    classes.sort()
    feature_names = training_data.columns.values[:-1] # ignore the last column as it's the class label

    class_count = training_df.groupby('class').count()[feature_names[0]].values
    num_rows = training_data.shape[0]

    inputs_params['classes'] = classes
    for j, label in enumerate(classes):
        inputs_mean[label] = [training_data[training_data['class']==label][x].mean() for x in feature_names]
        inputs_std[label] = [training_data[training_data['class']==label][x].std() if training_data[training_data['class']==label][x].std() > 0.01 else 0.01 for x in feature_names] # lower bound std at 0.01

        # Now we build the data structure used to compute probabilities
        # We store P(class) and Gaussian params to compute P(x_i|class)
        inputs_params[label] = class_count[j] / num_rows
        for i in range(len(feature_names)):
            print(f"Class {label}, attribute {feature_names[i]}, mean = {inputs_mean[label][i]:.2f}, std = {inputs_std[label][i]:.2f}")
            inputs_params[(feature_names[i], label)] = (inputs_mean[label][i], inputs_std[label][i]) # eg (x1, label) = (x1_mean, x1_std)

    return inputs_params


def naive_bayes_test(test_data, training_params):
    '''
    Takes a dictionary of training_params to compute the probability a given input from test_data belongs to a class
    training_params[(feature name, class)] = (feature mean, feature std)
    training_params[class] = count of class / number of rows
    '''
    results = [()] * test_data.shape[0] # store results for each test object
    for i, row in test_data.iterrows():
        feature_probs = defaultdict(list)
        row_prob = []
        for label in training_params['classes']:
            for j, feature in enumerate(row[:-1]):
                feature_name = 'x' + str(j+1)
                mean, std = training_params[(feature_name, label)]
                try:
                    feature_probs[label].append(math.log(norm(mean, std).pdf(feature), 10))
                except ValueError:
                    feature_probs[label].append(-500.0) # log10(0) = -inf, -500.0 is close

            row_prob.append(math.log(training_params[label], 10) + np.sum(feature_probs[label]))

        pred = training_params['classes'][np.argmax(row_prob)]
        prob = row_prob[np.argmax(row_prob)]
        ties = sum(np.isclose(row_prob, row_prob[np.argmax(row_prob)])) # isclose is used for float comparisons
        actual = row[-1]
        acc = int(np.isclose(pred, actual)) / ties
        results[i] = (i, pred, prob, actual, acc)

        print("ID={:5d}, predicted={:3f}, probability = {:.4f}, true={:3f}, accuracy={:4.2f}".format(*results[i]))
    accuracies = [x[4] for x in results]
    overall_acc = sum(accuracies) / len(accuracies)
    print(f"classification accuracy={overall_acc:6.4f}")


def parse_raw_data(raw_data):
    '''
    Assumes data is white-space delimited, with rows seperated by '\n' character
    Returns a pandas dataframe populated from raw_data
    '''
    data = raw_data.split('\n')
    data = list(map(methodcaller("split", " "), data))
    for i in range(len(data)):
        data[i] = [ast.literal_eval(elem) for elem in data[i] if len(elem) > 0]

    headers = [f"x{i+1}" for i in range(len(data[0]) - 1)]
    headers.append("class")

    df = pd.DataFrame(data, columns=headers)
    df = df.dropna()

    return df


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Error! expecting 2 arguments: <path to training data> <path to test data>")
    
    with open(sys.argv[1], 'r') as training_file, open(sys.argv[2], 'r') as test_file:
        training_raw = training_file.read()
        test_raw = test_file.read()
    
    training_df = parse_raw_data(training_raw)
    test_df = parse_raw_data(test_raw)

    training_params = naive_bayes_train(training_df)

    naive_bayes_test(test_df, training_params)
