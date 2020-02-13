from collections import Counter
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *
from test import write_csv

data = {}


def build_data(root):
    all_words = []
    files = [os.path.join(root, file) for file in os.listdir(root)]

    global data

    for file in files:
        with open(file,errors='ignore') as f:
            for line in f:
                words = line.split()
                all_words += words

    frequent = Counter(all_words)

    all_keys = list(frequent)

    for key in all_keys:
        if not key.isalpha():
            del frequent[key]

    frequent = frequent.most_common(2500)

    count = 0
    for word in frequent:
        data[word[0]] = count
        count += 1


def feature_extraction(root):
    files = [os.path.join(root, file) for file in os.listdir(root)]
    matrix = np.zeros((len(files), 2500))
    labels = np.zeros(len(files))
    file_count = 0

    for file in files:
        with open(file,errors='ignore') as file_obj:
            for index, line in enumerate(file_obj):
                if index == 2:
                    line = line.split()
                    for word in line:
                        if word in data:
                            matrix[file_count, data[word]] = line.count(word)

        labels[file_count] = 0
        if 'spam' in file:
            labels[file_count] = 1
        file_count += 1
    return matrix, labels


def call(training_data, testing_data, number, flag):
    # Building word data
    build_data(training_data)

    print('Extracting features')
    training_feature, training_labels = feature_extraction(training_data)
    testing_features, testing_labels = feature_extraction(testing_data)

    model = MultinomialNB()
    model.fit(training_feature, training_labels)

    # Predicting
    predicted_labels = model.predict(testing_features)

    with open("prior.txt") as p:
        prior = p.read().split()

    nd = prior[2]
    table = [[nd, accuracy_score(testing_labels, predicted_labels), precision_score(testing_labels, predicted_labels),
              recall_score(testing_labels, predicted_labels), f1_score(testing_labels, predicted_labels)]]
    if not flag:
        write_csv("data/enron" + str(number) + ".csv", table)
    else:
        write_csv("data/enron" + str(number) + "T.csv", table)
    print('Accuracy:', accuracy_score(testing_labels, predicted_labels))
    print('Precision:', precision_score(testing_labels, predicted_labels))
    print('Recall:', recall_score(testing_labels, predicted_labels))
    print("F1:", f1_score(testing_labels, predicted_labels))
