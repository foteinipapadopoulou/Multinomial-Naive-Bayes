import json
import math
from collections import Counter
import glob
import os
from train import remove_digits
import csv


Ham = 0
Spam = 1
word_spam = "spam"


def load_data_json(s):
    with open(s, "r") as fh:
        x = json.loads(fh.read())
    return dict(x)


def write_csv(name, t):
    fields = ['N.of docs', 'Accuracy', 'Precision', 'Recall', 'F1']
    with open(name, "a", newline='') as csv_file:
        csvwriter = csv.writer(csv_file, delimiter=";")
        #csvwriter.writerow(fields)
        csvwriter.writerows(t)


def calltest(path, number,flag):
    vocabulary = {}
    probabilities = {}

    with open("prior.txt") as p:
        prior = p.read().split()

    nd = prior[2]
    #Positive -> Ham, Negative-> Spam
    #tp->True positive,tn->True negative
    #fp->False positive ,fn->False negative
    tp, tn, fp, fn = 0, 0, 0, 0
    tp1, tn1, fp1, fn1 = 0, 0, 0, 0

    vocabulary[Ham] = load_data_json("vocHam.json")
    vocabulary[Spam] = load_data_json("vocSpam.json")
    probabilities[Ham] = load_data_json("probHam.json")
    probabilities[Spam] = load_data_json("probSpam.json")
    number_of_documents = 0
    for filename in glob.glob(os.path.join(path, '*.txt')):
        number_of_documents += 1
        countw = Counter()
        # save the dictionary
        with open(filename, 'r') as file_to_read:
            words = Counter(file_to_read.read().split())
            countw += words

        # remove the numbers
        countw = remove_digits(countw)

        ####with Π#####
        scoreh = float(prior[Ham])
        scores = float(prior[Spam])
        for t in countw:
            if t in probabilities[Ham]:
                scoreh *= math.pow(float(probabilities[Ham][t]), countw[t])
            if t in probabilities[Spam]:
                scores *= math.pow(float(probabilities[Spam][t]), countw[t])
        if scoreh < scores:
            if filename.find(word_spam) == -1:
                fn1 += 1
            else:
                tn1 += 1
        else:
            if filename.find(word_spam) != -1:
                fp1 += 1
            else:
                tp1 += 1

        ####with log######
        score = [math.log(float(prior[Ham])), math.log(float(prior[Spam]))]
        for t in countw:

                if t in probabilities[Ham]:
                    score[Ham] += math.log(float(probabilities[Ham][t]))* countw[t]

                if t in probabilities[Spam]:
                    score[Spam] += math.log(float(probabilities[Spam][t]))* countw[t]

        if score[Ham] < score[Spam]:
            if filename.find(word_spam) == -1:
                fn += 1
            else:
                tn += 1
        else:
            if filename.find(word_spam) != -1:
                fp += 1
            else:
                tp += 1
        score.clear()
    print("Tested on ", number_of_documents, " documents.")
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    print("--Test on algorithm with log--")
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("Accuracy =", accuracy)
    print("Precision= ", precision)
    print("Recall = ", recall)
    print("f1 = ", f1)
    table = [[nd, accuracy, precision, recall, f1]]
    if not flag:
        write_csv("mydata/withlog/enron"+str(number)+"_withlog.csv", table)
    else:
        write_csv("mydata/withlog/enron" + str(number) + "_withlogT.csv", table)
    accuracy1 = (tp1 + tn1) / (tp1 + tn1 + fp1 + fn1)
    precision1 = tp1 / (tp1 + fp1)
    recall1 = tp1 / (tp1 + fn1)
    f11 = 2 * (precision1 * recall1) / (precision1 + recall1)
    print("--Test on algorithm with Π(multiplication of possibilities)--")
    print("Accuracy =", accuracy1)
    print("Precision= ", precision1)
    print("Recall = ", recall1)
    print("f1 = ", f11)
    table1 = [[nd, accuracy1, precision1, recall1, f11]]
    if not flag:
        write_csv("mydata/withΠ/enron"+str(number)+"_withΠ.csv", table1)
    else:
        write_csv("mydata/withΠ/enron" + str(number) + "_withΠT.csv", table1)
