import sys
import os
import numpy as np
from functools import reduce
from sklearn.metrics import precision_recall_fscore_support
from parse_args import parse_arguments

LABELS = ["negative", "mixed", "neutral", "positive"]

def get_prediction(predictions):
    if max(predictions) == predictions[2]:
        return 'neutral'    
    elif max(predictions) == predictions[0]:
        return 'negative'
    elif max(predictions) == predictions[3]:
        return 'positive'
    elif max(predictions) == predictions[1]:
        return 'mixed'

def summarize(args):
    totals = []
    split = args.split
    for i in range(0, split):
        dataset = "{}/set{}".format(*args.path, i)
        data = []
        with open("{}/test.tsv".format(dataset), encoding = 'utf-8', mode = 'r') as f:
            data = [{"label": line.strip().split("\t")[1]} for line in f.readlines()[1:]]

        with open("{}/output/test_results.tsv".format(dataset), encoding = 'utf-8', mode = 'r') as f:
            for num, line in enumerate(f.readlines(), start=0):
                data[num]["prediction"] = get_prediction(line.strip().split("\t"))

        y_true = np.array([entry["label"] for entry in data])
        y_pred = np.array([entry["prediction"] for entry in data])
        
        with open("{}/output/summary.txt".format(dataset), encoding = 'utf-8', mode = 'w') as f:
            f.write("             precision    recall   f1-score   support\n")
            labeled = precision_recall_fscore_support(y_true, y_pred, average=None, labels=LABELS)
            for num, label in enumerate(LABELS, start=0):
                f.write("{:10}{:10}{:10}{:10}{:10}\n".format(label, *np.around([sub[num] for sub in labeled], 2)))
                
            macro = precision_recall_fscore_support(y_true, y_pred, average='macro')
            f.write("{:10}{:10}{:10}{:10}{:10}\n".format("macro", *np.around(macro[:-1], 2), len(data)))
            weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            f.write("{:10}{:10}{:10}{:10}{:10}\n".format("weighted", *np.around(weighted[:-1], 2), len(data)))
            total = {"labeled": labeled,
                     "macro": macro[:-1],
                     "weighted": weighted[:-1]}
            totals.append(total)
    
    with open("{}/total_summary.txt".format(*args.path), encoding = 'utf-8', mode = 'w') as f:
        f.write("             precision    recall   f1-score   support\n")
        labeled = np.array(reduce(np.add, [sub["labeled"] for sub in totals])) / split
        for num, label in enumerate(LABELS, start=0):
            f.write("{:10}{:10}{:10}{:10}{:10}\n".format(label, *np.around([sub[num] for sub in labeled], 2)))
            
        macro = np.array(reduce(np.add, [sub["macro"] for sub in totals])) / split
        f.write("{:10}{:10}{:10}{:10}{:10}\n".format("macro", *np.around(macro, 2), len(data)))
        weighted = np.array(reduce(np.add, [sub["weighted"] for sub in totals])) / split
        f.write("{:10}{:10}{:10}{:10}{:10}\n".format("weighted", *np.around(weighted, 2), len(data)))

if __name__ == '__main__':
    arguments = parse_arguments()
    sys.exit(summarize(arguments))