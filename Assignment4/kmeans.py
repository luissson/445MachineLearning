import pandas as pd
import numpy as np
from collections import defaultdict
from random import randrange
import pdb
from operator import methodcaller
import ast


def convert_raw_data(raw_data):
    data = raw_data.split('\n')
    data = list(map(methodcaller("split", ","), data))
    for i in range(len(data)):
        data[i] = [ast.literal_eval(elem) for elem in data[i] if len(elem) > 0]

    headers = [f"x{i+1}" for i in range(len(data[0]) - 1)]
    headers.append("class")

    df = pd.DataFrame(data, columns=headers)
    df = df.dropna()

    return df


def kmeans(training_features, training_labels, num_centers):
    print("Starting k-means clustering")
    # k means algorithm
    num_features = len(training_features)
    # initially select features as centers randomly
    centers = []
    for _ in range(num_centers):
        r = randrange(num_features)
        centers.append(training_features.loc[r])

    mins = []
    # loop until centroids do not change
    num_iterations = 1
    while(True):
        # calculate square distance of each feature to each center
        sq_distances = [] # stores the square distance of each feature to each center
        for center in centers:
            sq_distances.append(((training_features - center)**2).sum(axis=1))
        sq_distances = pd.DataFrame(sq_distances)

        # associate each feature with a cluster center
        mins.append(sq_distances.idxmin())

        # compare centroid membership between iterations
        if len(mins) > 1 and (mins.pop(0)).equals(mins[0]):
            break

        # update cluster centers using cluster members
        for i, _ in enumerate(centers):
            # for each center, get member features
            members = mins[0].loc[mins[0] == i].index.tolist()
            centers[i] = training_features.ix[members].mean()
        
        num_iterations += 1

    print(f"Finished clustering after {num_iterations} iterations")
    print()
    print("******************")
    print("Cluster Statistics")
    print("******************")
    print()
    print("*** Mean Square Errors ***")
    mean_sq_error(training_features, centers, mins)
    print()
    print("*** Mean Square Separation ***")
    mean_sq_separation(centers)
    print()
    print("*** Mean Entropy ***")
    cluster_counts, cluster_labels = avg_mean_entropy(training_labels, mins[0], num_centers)
    print()

    return centers, cluster_labels


def avg_mean_entropy(training_labels, membership, num_clusters):
    # entropy for a cluster = sum_j p_j * log_2 p_j, j = class label
    labels = training_labels.unique()
    num_labels = len(labels)
    num_entities = len(membership)

    cluster_counts = {}
    cluster_probs = {}
    cluster_entropies = {}
    cluster_labels = {}

    for i in range(num_clusters):
        cluster_counts[i] = training_labels.ix[membership == i].value_counts()
        cluster_probs[i] = cluster_counts[i] / cluster_counts[i].sum() # number of entities with 'label' in a cluster
        cluster_entropies[i] = (-cluster_probs[i] * np.log2(cluster_probs[i])).sum()
        cluster_labels[i] = cluster_counts[i].idxmax()
        print(f"Cluster {i+1} Entropy: {cluster_entropies[i]:.2f}")

    mean_entropy = sum([cluster_counts[i].sum() * cluster_entropies[i] / num_entities for i in range(num_clusters)])
    print()
    print(f"Mean Entropy: {mean_entropy:.2f}")

    return cluster_counts, cluster_labels
    

def mean_sq_separation(centers):
    k = len(centers)
    mss = []
    sq_distances = []
    for i, center in enumerate(centers):
        # sum the distances between each pair of clusters
        sq_distance = [((center - x)**2).sum() for j, x in enumerate(centers) if j > i ] 
        sq_distances.append(sum(sq_distance))

    mss = 2 * sum(sq_distances) / (k*(k-1))

    print(f"Mean-Square-Separation: {mss:.2f}")


def mean_sq_error(training_features, centers, membership):
    mses = []
    for i, center in enumerate(centers):
        # sum 
        members = membership[0].loc[membership[0] == i].index.tolist()
        #training_features.ix[members]
        sq_distances = ((training_features.ix[members] - center)**2).sum(axis=1)
        mses.append(sq_distances.sum() / len(sq_distances))
        print(f"Cluster {i+1} MSE: {mses[i]:.2f}")
    
    avg_mse = sum(mses) / len(mses)
    print()
    print(f"Average Mean-Square-Error: {avg_mse:.2f}")


def test(test_features, test_labels, centers, cluster_labels):
    # for each test feature, find the nearest cluster center
    sq_distances = []
    for center in centers:
        sq_distances.append(((test_features - center)**2).sum(axis=1))
    sq_distances = pd.DataFrame(sq_distances)

    # associate each feature with a cluster center
    mins = (sq_distances.idxmin())

    # compare that cluster label with the test label
    for i in range(len(centers)):
        mins.replace(mins[mins==i], cluster_labels[i], inplace=True)
    
    acc = (mins == test_labels).sum() / len(test_labels)
    print(f"Classification accuracy: {acc:.2f}")

    conf_matrix = {}
    for label in test_labels.unique():
        label_idxs = test_labels[test_labels == label].index.values # get map from index to label
        conf_matrix[label] = mins.ix[label_idxs].value_counts() # use index->label map to get corresponding predictions, get prediction label

    return acc, conf_matrix

def main(num_centers):
    # load data
    with open("optdigits/optdigits.train", "r") as train_file, \
            open("optdigits/optdigits.test", "r") as test_file:
        raw_train = train_file.read()
        raw_test = test_file.read()

    test_data = convert_raw_data(raw_test)
    test_labels = test_data['class']
    test_features = test_data.drop(labels='class', axis=1)

    training_data = convert_raw_data(raw_train)
    training_labels = training_data['class']
    training_features = training_data.drop(labels='class', axis=1)

    centers, cluster_labels = kmeans(training_features, training_labels, num_centers)

    test(test_features, test_labels, centers, cluster_labels)



if __name__ == '__main__':
    main(30)