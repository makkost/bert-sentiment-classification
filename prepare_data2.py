import json
import random
import sys
import os
from functools import reduce
from operator import add
from parse_args import parse_arguments

def mark_to_polarity(mark):
    if mark == "positive":
        return "positive"
    if mark == "negative":
        return "negative"
    if mark == "mixed":
        return "mixed"
    if mark == "objective":
        return "neutral"

def split_data(lst, number_of_parts):
    list_copy = lst[:]
    for i in range(number_of_parts, 0, -1):
        part = list_copy[:len(list_copy) // i]
        list_copy = list_copy[len(list_copy) // i:]
        yield part

def load_corpora():
    split = 4
    output = "data"
    positive = []
    negative = []
    objective = []
    mixed = []
    with open("ru_dataset_4/mixed 105.csv", encoding = 'utf-8', mode = 'r') as f:
        mixed = [(sentence, "mixed") for sentence in f.readlines()]

    with open("ru_dataset_4/negative 1065.csv", encoding = 'utf-8', mode = 'r') as f:
        negative = [(sentence, "negative") for sentence in f.readlines()]

    with open("ru_dataset_4/neutral 51.csv", encoding = 'utf-8', mode = 'r') as f:
        objective = [(sentence, "neutral") for sentence in f.readlines()]

    with open("ru_dataset_4/positive 1065.csv", encoding = 'utf-8', mode = 'r') as f:
        positive = [(sentence, "positive") for sentence in f.readlines()]

    positive_chunks = list(split_data(positive, split))
    negative_chunks = list(split_data(negative, split))
    objective_chunks = list(split_data(objective, split))
    mixed_chunks = list(split_data(mixed, split))
    for index in range(0, split):
        test = positive_chunks[index] + negative_chunks[index] + objective_chunks[index] + mixed_chunks[index]
        random.shuffle(test)
        train = [item for sublist in positive_chunks[:index] + positive_chunks[index+1:] for item in sublist] +\
                [item for sublist in negative_chunks[:index] + negative_chunks[index+1:] for item in sublist] +\
                [item for sublist in objective_chunks[:index] + objective_chunks[index+1:] for item in sublist] +\
                [item for sublist in mixed_chunks[:index] + mixed_chunks[index+1:] for item in sublist]
        random.shuffle(train)
        os.makedirs("{}/set{}".format(output, index), exist_ok = True)
        with open("{}/set{}/train.tsv".format(output, index), encoding = 'utf-8', mode = 'w') as f:
            f.write("phrase\tlabel\n")
            for phrase, label in train:
                 f.write("{}\t{}\n".format(phrase.strip(), label))
                 
        with open("{}/set{}/dev.tsv".format(output, index), encoding = 'utf-8', mode = 'w') as f:
            f.write("phrase\tlabel\n")
            for phrase, label in test:
                 f.write("{}\t{}\n".format(phrase.strip(), label))
                 
        with open("{}/set{}/test.tsv".format(output, index), encoding = 'utf-8', mode = 'w') as f:
            f.write("phrase\tlabel\n")
            for phrase, label in test:
                 f.write("{}\t{}\n".format(phrase.strip(), label))

if __name__ == '__main__':
    sys.exit(load_corpora())