import sys
import os
import re
import numpy as np
from functools import reduce
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from parse_args import parse_arguments

LABELS = ["neutral", "positive", "negative", "mixed"]

def get_label(num):
    return LABELS[int(num)]

def summarize(args):
    totals = []
    split = args.split
    for i in range(0, split):
        dataset = "{}/set{}".format(*args.path, i)
        data = []
        with open("{}/output/test_results.tsv".format(dataset), encoding = 'utf-8', mode = 'r') as f:
            for num, line in enumerate(f.readlines(), start=0):
                sent, pred = line.strip().split("\t")
                data.append({"label": get_label(sent), "prediction": get_label(pred)})

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

def confusion(args):
    totals = np.zeros((len(LABELS), len(LABELS)))
    split = args.split
    for i in range(0, split):
        dataset = "{}/set{}".format(*args.path, i)
        data = []
        # with open("{}/test.tsv".format(dataset), encoding = 'utf-8', mode = 'r') as f:
        #     data = [{"text": line.strip().split("\t")[0]} for line in f.readlines()[1:]]

        with open("{}/test_result.tsv".format(dataset), encoding = 'utf-8', mode = 'r') as f:
            for num, line in enumerate(f.readlines(), start=0):
                sent, pred = line.strip().split("\t")
                data.append({"label": get_label(sent), "prediction": get_label(pred)})

        # with open("{}/output/test_results.tsv".format(dataset), encoding = 'utf-8', mode = 'r') as f:
        #     for num, line in enumerate(f.readlines(), start=0):
        #         sent, pred = line.strip().split("\t")
        #         data[num]["label"] = get_label(sent)
        #         data[num]["prediction"] = get_label(pred)

        y_true = np.array([entry["label"] for entry in data])
        y_pred = np.array([entry["prediction"] for entry in data])
        
        with open("{}/confusion.txt".format(dataset), encoding = 'utf-8', mode = 'w') as f:
            labeled = confusion_matrix(y_true, y_pred, labels=LABELS)
            print(labeled)
            f.write("\tpredicted\n")
            f.write(" \t{}\t{}\t{}\t{}\n".format(*LABELS))
            for num, label in enumerate(LABELS, start=0):
                f.write("{}\t{}\t{}\t{}\t{}\n".format(label, *labeled[num]))

            totals = totals + labeled
        
        # with open("{}/output/errors.txt".format(dataset), encoding = 'utf-8', mode = 'w') as f:
        #     f.write("text\tlabel\tprediction\n")
        #     for line in data:
        #         if line["label"] != line["prediction"]:
        #             f.write("{}\t{}\t{}\n".format(line["text"], line["label"], line["prediction"]))
    print(totals)
    with open("{}/total_confusion.txt".format(*args.path), encoding = 'utf-8', mode = 'w') as f:
        f.write("\tpredicted\n")
        f.write(" \t{}\t{}\t{}\t{}\n".format(*LABELS))
        for num, label in enumerate(LABELS, start=0):
            f.write("{}\t{}\t{}\t{}\t{}\n".format(label, *np.around(totals[num] / 424.0, 2)))

if __name__ == '__main__':
    arguments = parse_arguments()
    sys.exit(summarize(arguments))
