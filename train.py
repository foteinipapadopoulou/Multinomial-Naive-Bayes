import glob
import os
import json
from collections import Counter
import csv

Ham = 0
Spam = 1
word_spam = "spam"


def count_docs(p):
    n = 0  # number of documents in this folder
    for i in glob.glob(os.path.join(p, '*.txt')):
        n += 1
    return n


def count_docs_in_class(path, c):
    # number of spam/ham documents
    nc = 0  # type: int
    for filename in glob.glob(os.path.join(path, '*.txt')):
        if c == 'Spam':
            if filename.find(word_spam) != -1:
                nc += 1
        else:
            if filename.find(word_spam) == -1:
                nc += 1
    return nc


def remove_digits(count_words):
    to_remove = set()
    vocabul_to_remove = ['.', ',', ')', '(', "'", "Subject:", "to", "the", "_", "-",
                         "on", "at", "in", "of", "a", "an", "/"]
    for k in count_words:
        if k.isdigit():
            to_remove.add(k)
        if k in vocabul_to_remove:
            to_remove.add(k)
    for i in to_remove:
        del count_words[i]
    to_remove.clear()
    return count_words


def write_to_json(ct, f):
    with open(f + ".json", "w") as file_to_write:
        json.dump(ct, file_to_write)


def fill_vocabulary(v,  x):
    for i in v:
        v[i] = 0
    for i in x:
        if i in v:
            v[i] += x[i]
    return v


def calltrain(path):
    # Always in 1st index of array we have elements about Ham and in the 2nd about Spam
    nc = [0, 0]  # holds number of documents in spam class and ham class
    number_of_words = {}  # holds number of words that has each spam class and ham class
    vocabulary = {}  # holds occurrences of each word in spams and hams
    condprob = {}  # holds probabilities of each word in spams and hams
    cl = [0, 1]
    n = count_docs(path)  # holds number of documents
    nc[Ham] = count_docs_in_class(path, 'Ham')
    nc[Spam] = count_docs_in_class(path, 'Spam')
    print("Total number of documents(ham+spam)=", n)
    print("Spam= ", nc[Spam])
    print("Ham= ", nc[Ham])
    fields = ['Ham', 'Spam']
    with open("mydata/num.csv", "a", newline='') as csv_file:
        csvwriter = csv.writer(csv_file, delimiter=";")
        #csvwriter.writerow(fields)
        csvwriter.writerows([[nc[Ham],nc[Spam]]])

    # possibilities of spam mails and ham emails [p(ham),p(spam)]
    prior = [nc[Ham] / n, nc[Spam] / n]
    # write possibilities to a txt called "prior"
    with open("prior.txt", "w") as prob:
        text = str(prior[Ham]) + " " + str(prior[Spam])+" "+str(n)
        prob.write(text)
    # save the occurrences of each document and finally for all spams and hams
    co = {}
    co[Ham] = Counter()
    co[Spam] = Counter()

    for filename in glob.glob(os.path.join(path, '*.txt')):
        if filename.find(word_spam) != -1:
            with open(filename, 'r',errors='ignore') as file_to_read:
                words = Counter(file_to_read.read().split())
                co[Spam] += words
        else:
            with open(filename, 'r',errors='ignore') as file_to_read:
                words = Counter(file_to_read.read().split())
                co[Ham] += words
    co[Ham].most_common()
    # remove the digits
    co[Ham] = remove_digits(co[Ham])
    co[Spam] = remove_digits(co[Spam])

    # union the words of spams and hams
    voc = set(co[Ham]).union(set(co[Spam]))

    for c in cl:
        vocabulary[c] = Counter(voc)
        vocabulary[c] = fill_vocabulary(vocabulary[c], co[c])
        number_of_words[c] = sum(value for value in vocabulary[c].values())
        condprob[c] = dict.fromkeys(vocabulary[c].elements(), 0)

    vocabulary[Ham] = dict(vocabulary[Ham])
    vocabulary[Spam] = dict(vocabulary[Spam])
    write_to_json(vocabulary[Ham], "vocHam")
    write_to_json(vocabulary[Spam], "vocSpam")

    for c in cl:
        for t in vocabulary[c].keys():
            condprob[c][t] = (vocabulary[c][t]+1)/(number_of_words[c] + len(voc))

    write_to_json(condprob[Spam], "probSpam")
    write_to_json(condprob[Ham], "probHam")
    print("All completed")
