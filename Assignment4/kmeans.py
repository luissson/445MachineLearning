import pandas as pd
from random import randrange
import pdb
from operator import methodcaller
import ast


def convert_raw_data(raw_data):
    data = raw_data.split('\n')
    data = list(map(methodcaller("split", ","), data))
    for i in range(len(data)):
        data[i] = [ast.literal_eval(elem) for elem in data[i] if len(elem) > 0]
    del data[-1]

    headers = [f"x{i+1}" for i in range(len(data[0]) - 1)]
    headers.append("class")

    df = pd.DataFrame(data, columns=headers)
    df = df.dropna()

    return df


def sq_distance():
    pass

def main(num_centers):

    # load data
    with open("optdigits/optdigits.train", "r") as train_file:
        raw_train = train_file.read()

    training_data = convert_raw_data(raw_train)
    training_labels = training_data['class']
    training_features = training_data.drop(labels='class', axis=1)

    num_instances = len(training_features)

    # k means algorithm

    # initially select features as centers randomly
    centers = []
    for _ in range(num_centers):
        r = randrange(num_instances)
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
        
        print(f"Finished iteration #{num_iterations}")
        num_iterations += 1



if __name__ == '__main__':
    main(10)