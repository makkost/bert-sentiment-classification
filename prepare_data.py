import json
import random
import sys
import os
from functools import reduce
from operator import add
from parse_args import parse_arguments

import codecs
sys.stdout.reconfigure(encoding='utf-8')

def mark_to_polarity(mark):
    if mark == "positive":
        return "positive"
    if mark == "negative":
        return "negative"
    if mark == "mixed":
        return "mixed"
    if mark == "objective":
        return "neutral"

def mark_to_polarity2(mark):
    if mark == "+":
        return "positive"
    if mark == "-":
        return "negative"
    if mark == "*":
        return "mixed"
    if mark == "0":
        return "neutral"

def split_data(lst, number_of_parts):
    list_copy = lst[:]
    for i in range(number_of_parts, 0, -1):
        part = list_copy[:len(list_copy) // i]
        list_copy = list_copy[len(list_copy) // i:]
        yield part

def load_corpora(args):
    split = args.split
    output = args.output
    texts = list()
    for corpora_file in args.path:
        with open(corpora_file, encoding = 'utf-8', mode = 'r') as f:
            texts.extend(json.load(f))
            
    sentences = reduce(add, [text['sentences'] for text in texts], [])
    
    positive = [(sentence['sentence'], mark_to_polarity(sentence['mark'])) for sentence in sentences if sentence['mark'] == "positive"]
    negative = [(sentence['sentence'], mark_to_polarity(sentence['mark'])) for sentence in sentences if sentence['mark'] == "negative"]
    objective = [(sentence['sentence'], mark_to_polarity(sentence['mark'])) for sentence in sentences if sentence['mark'] == "objective"]
    mixed = [(sentence['sentence'], mark_to_polarity(sentence['mark'])) for sentence in sentences if sentence['mark'] == "mixed"]

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


def load_corpora2(args):
    split = args.split
    output = args.output
    texts = list()
    for corpora_file in args.path:
        with open(corpora_file, encoding = 'utf-8', mode = 'r') as f:
            texts.extend(f.readlines())

    sentences = reduce(add, [[text.strip().split(' ', 1)] for text in texts if text[0] in ['+', '-', '0', '*']], [])

    positive = [sentence[1] for sentence in sentences if sentence[0] == '+']
    negative = [sentence[1] for sentence in sentences if sentence[0] == '-']
    objective = [sentence[1] for sentence in sentences if sentence[0] == '0']
    mixed = [sentence[1] for sentence in sentences if sentence[0] == '*']

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
        print(set(test) & set(train))
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
    arguments = parse_arguments()
    sys.exit(load_corpora2(arguments))